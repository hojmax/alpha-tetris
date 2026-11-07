from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from simple_parsing import parse

from tetris_bot.action_space import LEGACY_NUM_ACTIONS
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    PROJECT_ROOT,
)
from tetris_bot.ml.config import TrainingConfig
from tetris_bot.ml.loss import compute_loss
from tetris_bot.ml.network import ROW_FILL_COUNT_FEATURES
from tetris_bot.ml.optimizer import OptimizerBundle
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    ResidualFusionBlock,
    get_preload_mode,
    init_wandb_run,
    normalize_policy_arrays,
    pick_device,
    validate_common_offline_args,
)

logger = structlog.get_logger()

REQUIRED_NPZ_BASE_KEYS = (
    "boards",
    "current_pieces",
    "hold_pieces",
    "hold_available",
    "next_queue",
    "placement_counts",
    "combos",
    "back_to_back",
    "next_hidden_piece_probs",
    "policy_targets",
    "value_targets",
    "action_masks",
)

BASE_AUX_FEATURES = 7 + 8 + 1 + 35 + 1 + 1 + 1 + 7


@dataclass(frozen=True)
class ExtraFeatureGroup:
    name: str
    npz_key: str
    width: int
    normalize_by: float = 1.0


CORE_EXTRA_FEATURE_GROUPS = (
    ExtraFeatureGroup(
        name="column_heights",
        npz_key="column_heights",
        width=BOARD_WIDTH,
    ),
    ExtraFeatureGroup(
        name="max_column_height",
        npz_key="max_column_heights",
        width=1,
    ),
    ExtraFeatureGroup(
        name="row_fill_counts",
        npz_key="row_fill_counts",
        width=ROW_FILL_COUNT_FEATURES,
    ),
    ExtraFeatureGroup(
        name="total_blocks",
        npz_key="total_blocks",
        width=1,
    ),
    ExtraFeatureGroup(
        name="bumpiness",
        npz_key="bumpiness",
        width=1,
    ),
    ExtraFeatureGroup(
        name="holes",
        npz_key="holes",
        width=1,
    ),
    ExtraFeatureGroup(
        name="overhang_fields",
        npz_key="overhang_fields",
        width=1,
    ),
)
MOVE_NUMBER_EXTRA_FEATURE_GROUP_NAME = "move_number"
MOVE_NUMBER_EXTRA_FEATURE_GROUP_NPZ_KEY = "move_numbers"


@dataclass
class ScriptArgs:
    data_path: Path = (
        PROJECT_ROOT / "training_data_v22.npz"
    )  # Path to offline replay buffer NPZ
    device: str = "auto"  # auto/cpu/cuda/mps
    seed: int = 123
    max_examples: int = 0  # 0 = use all examples in NPZ
    train_fraction: float = 0.9
    steps: int = 10000
    batch_size: int = 1024
    eval_interval: int = 20
    eval_examples: int = 32_768  # Max examples to use per train/val eval pass
    eval_batch_size: int = 2048
    log_train_metrics_every: int = 1  # Batch metric logging cadence
    preload_to_gpu: bool = True  # Preload selected dataset tensors to GPU
    preload_to_ram: bool = False  # Preload selected dataset tensors to CPU RAM
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    value_loss_weight: float = 1.0

    conv_filters: list[int] = field(default_factory=lambda: [4, 8])
    fc_hidden: int = 128
    conv_kernel_size: int = 3
    conv_padding: int = 1
    aux_hidden: int = 24
    num_fusion_blocks: int = 0

    max_placements: int = (
        TrainingConfig.max_placements
    )  # For normalizing move_numbers feature (if enabled)
    include_move_number_feature: bool = False

    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "feature-ablation"]
    )


@dataclass
class FeatureVariant:
    name: str
    wandb_prefix: str
    included_groups: tuple[ExtraFeatureGroup, ...]
    aux_features: int


@dataclass
class OfflineTensorDataset:
    boards: torch.Tensor
    aux: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    action_masks: torch.Tensor
    storage_device: torch.device


@dataclass
class OfflineDataSource:
    npz: np.lib.npyio.NpzFile
    selected_global_indices: np.ndarray
    tensor_data: OfflineTensorDataset | None
    included_groups: tuple[ExtraFeatureGroup, ...]


class GatedFeatureAblationTetrisNet(nn.Module):
    def __init__(
        self,
        conv_filters: list[int],
        fc_hidden: int,
        conv_kernel_size: int,
        conv_padding: int,
        aux_hidden: int,
        num_fusion_blocks: int,
        aux_features: int,
    ):
        super().__init__()
        if len(conv_filters) != 2:
            raise ValueError(
                f"Expected exactly 2 convolutional filters, got {len(conv_filters)}"
            )
        if aux_features <= 0:
            raise ValueError("aux_features must be > 0")
        if num_fusion_blocks < 0:
            raise ValueError("num_fusion_blocks must be >= 0")

        conv0 = conv_filters[0]
        conv1 = conv_filters[1]
        conv_flat_size = BOARD_HEIGHT * BOARD_WIDTH * conv1
        fusion_hidden = fc_hidden

        self.conv1 = nn.Conv2d(
            1, conv0, kernel_size=conv_kernel_size, padding=conv_padding
        )
        self.bn1 = nn.BatchNorm2d(conv0)
        self.conv2 = nn.Conv2d(
            conv0,
            conv1,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.bn2 = nn.BatchNorm2d(conv1)

        self.board_proj = nn.Linear(conv_flat_size, fusion_hidden)

        self.aux_fc = nn.Linear(aux_features, aux_hidden)
        self.aux_ln = nn.LayerNorm(aux_hidden)
        self.gate_fc = nn.Linear(aux_hidden, fusion_hidden)
        self.aux_proj = nn.Linear(aux_hidden, fusion_hidden)

        self.fusion_ln = nn.LayerNorm(fusion_hidden)
        self.fusion_blocks = nn.ModuleList(
            [ResidualFusionBlock(fusion_hidden) for _ in range(num_fusion_blocks)]
        )

        self.policy_head = nn.Linear(fusion_hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(fusion_hidden, 1)

    def forward(
        self,
        board: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        board_h = self.board_proj(x.view(x.size(0), -1))

        aux_h = F.relu(self.aux_ln(self.aux_fc(aux_features)))
        gate = torch.sigmoid(self.gate_fc(aux_h))
        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))
        for block in self.fusion_blocks:
            fused = block(fused)
        return self.policy_head(fused), self.value_head(fused)


def get_extra_feature_groups(
    include_move_number_feature: bool,
    max_placements: int,
) -> tuple[ExtraFeatureGroup, ...]:
    groups: list[ExtraFeatureGroup] = list(CORE_EXTRA_FEATURE_GROUPS)
    if include_move_number_feature:
        groups.append(
            ExtraFeatureGroup(
                name=MOVE_NUMBER_EXTRA_FEATURE_GROUP_NAME,
                npz_key=MOVE_NUMBER_EXTRA_FEATURE_GROUP_NPZ_KEY,
                width=1,
                normalize_by=float(max_placements),
            )
        )
    return tuple(groups)


def build_feature_variants(
    extra_feature_groups: tuple[ExtraFeatureGroup, ...],
) -> list[FeatureVariant]:
    variants: list[FeatureVariant] = [
        FeatureVariant(
            name="no_extra_features",
            wandb_prefix="no_extra_features",
            included_groups=tuple(),
            aux_features=BASE_AUX_FEATURES,
        )
    ]

    all_width = sum(group.width for group in extra_feature_groups)
    variants.append(
        FeatureVariant(
            name="all_extra_features",
            wandb_prefix="all_extra_features",
            included_groups=extra_feature_groups,
            aux_features=BASE_AUX_FEATURES + all_width,
        )
    )

    for group in extra_feature_groups:
        included_groups = tuple(g for g in extra_feature_groups if g.name != group.name)
        variants.append(
            FeatureVariant(
                name=f"all_without_{group.name}",
                wandb_prefix=f"all_without_{group.name}",
                included_groups=included_groups,
                aux_features=BASE_AUX_FEATURES + sum(g.width for g in included_groups),
            )
        )

    return variants


def validate_args(args: ScriptArgs) -> None:
    validate_common_offline_args(args)
    if len(args.conv_filters) != 2:
        raise ValueError("conv_filters must contain exactly two values")
    if args.conv_kernel_size <= 0:
        raise ValueError("conv_kernel_size must be > 0")
    if args.aux_hidden <= 0:
        raise ValueError("aux_hidden must be > 0")
    if args.num_fusion_blocks < 0:
        raise ValueError("num_fusion_blocks must be >= 0")


def ensure_required_keys(
    data: np.lib.npyio.NpzFile,
    extra_feature_groups: tuple[ExtraFeatureGroup, ...],
) -> None:
    required_keys: list[str] = list(REQUIRED_NPZ_BASE_KEYS)
    required_keys.extend(group.npz_key for group in extra_feature_groups)
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"NPZ is missing required keys: {missing}")


def validate_shapes(
    data: np.lib.npyio.NpzFile,
    extra_feature_groups: tuple[ExtraFeatureGroup, ...],
) -> int:
    n = int(data["boards"].shape[0])
    if data["boards"].shape[1:] != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(
            f"boards must have shape (N, {BOARD_HEIGHT}, {BOARD_WIDTH}), "
            f"got {data['boards'].shape}"
        )
    if data["current_pieces"].shape[1] != 7:
        raise ValueError("current_pieces must have shape (N, 7)")
    if data["hold_pieces"].shape[1] != 8:
        raise ValueError("hold_pieces must have shape (N, 8)")
    if data["next_queue"].shape[1:] != (5, 7):
        raise ValueError("next_queue must have shape (N, 5, 7)")
    if data["next_hidden_piece_probs"].shape[1] != 7:
        raise ValueError("next_hidden_piece_probs must have shape (N, 7)")
    policy_width = int(data["policy_targets"].shape[1])
    if policy_width not in {NUM_ACTIONS, LEGACY_NUM_ACTIONS}:
        raise ValueError(
            "policy_targets must have shape "
            f"(N, {NUM_ACTIONS}) or legacy shape (N, {LEGACY_NUM_ACTIONS})"
        )
    if data["action_masks"].shape[1] != policy_width:
        raise ValueError(
            "action_masks width must match policy_targets width "
            f"(got {data['action_masks'].shape[1]} vs {policy_width})"
        )

    for group in extra_feature_groups:
        group_shape = data[group.npz_key].shape
        if group.width == 1:
            expected_shape = (n,)
            if group_shape != expected_shape:
                raise ValueError(
                    f"{group.npz_key} must have shape {expected_shape}, "
                    f"got {group_shape}"
                )
            continue
        expected_shape = (n, group.width)
        if group_shape != expected_shape:
            raise ValueError(
                f"{group.npz_key} must have shape {expected_shape}, got {group_shape}"
            )

    per_example_keys: list[str] = [
        "current_pieces",
        "hold_pieces",
        "hold_available",
        "next_queue",
        "placement_counts",
        "combos",
        "back_to_back",
        "next_hidden_piece_probs",
        "policy_targets",
        "value_targets",
        "action_masks",
    ]
    per_example_keys.extend(group.npz_key for group in extra_feature_groups)

    for key in per_example_keys:
        if int(data[key].shape[0]) != n:
            raise ValueError(
                f"{key} has inconsistent first dimension: {data[key].shape[0]} vs {n}"
            )
    return n


def build_base_aux_batch_from_npz(
    data: np.lib.npyio.NpzFile,
    global_indices: np.ndarray,
    current_pieces: np.ndarray | None = None,
) -> np.ndarray:
    if current_pieces is None:
        current_pieces = data["current_pieces"][global_indices].astype(
            np.float32, copy=False
        )
    hold_pieces = data["hold_pieces"][global_indices].astype(np.float32, copy=False)
    hold_available = (
        data["hold_available"][global_indices].astype(np.float32).reshape(-1, 1)
    )
    next_queue = (
        data["next_queue"][global_indices]
        .astype(np.float32)
        .reshape(len(global_indices), -1)
    )
    placement_counts = (
        data["placement_counts"][global_indices].astype(np.float32).reshape(-1, 1)
    )
    combos = data["combos"][global_indices].astype(np.float32).reshape(-1, 1)
    back_to_back = (
        data["back_to_back"][global_indices].astype(np.float32).reshape(-1, 1)
    )
    next_hidden_piece_probs = data["next_hidden_piece_probs"][global_indices].astype(
        np.float32, copy=False
    )
    return np.concatenate(
        [
            current_pieces,
            hold_pieces,
            hold_available,
            next_queue,
            placement_counts,
            combos,
            back_to_back,
            next_hidden_piece_probs,
        ],
        axis=1,
    )


def build_extra_aux_batch_from_npz(
    data: np.lib.npyio.NpzFile,
    global_indices: np.ndarray,
    included_groups: tuple[ExtraFeatureGroup, ...],
) -> np.ndarray | None:
    if not included_groups:
        return None

    feature_arrays: list[np.ndarray] = []
    for group in included_groups:
        values = data[group.npz_key][global_indices].astype(np.float32, copy=False)
        if group.normalize_by != 1.0:
            values = values / group.normalize_by
        if group.width == 1:
            values = values.reshape(-1, 1)
        feature_arrays.append(values)
    return np.concatenate(feature_arrays, axis=1)


def build_aux_batch_from_npz(
    data: np.lib.npyio.NpzFile,
    global_indices: np.ndarray,
    included_groups: tuple[ExtraFeatureGroup, ...],
    current_pieces: np.ndarray | None = None,
) -> np.ndarray:
    base_aux = build_base_aux_batch_from_npz(data, global_indices, current_pieces)
    extra_aux = build_extra_aux_batch_from_npz(data, global_indices, included_groups)
    if extra_aux is None:
        return base_aux
    return np.concatenate([base_aux, extra_aux], axis=1)


def build_torch_batch_from_npz(
    data: np.lib.npyio.NpzFile,
    global_indices: np.ndarray,
    device: torch.device,
    included_groups: tuple[ExtraFeatureGroup, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards_np = data["boards"][global_indices].astype(np.float32, copy=False)
    current_pieces_np = data["current_pieces"][global_indices].astype(
        np.float32, copy=False
    )
    aux_np = build_aux_batch_from_npz(
        data,
        global_indices,
        included_groups,
        current_pieces_np,
    )
    raw_policy_targets_np = data["policy_targets"][global_indices]
    value_targets_np = data["value_targets"][global_indices].astype(
        np.float32, copy=False
    )
    raw_action_masks_np = data["action_masks"][global_indices]
    policy_targets_np, action_masks_np = normalize_policy_arrays(
        current_pieces=current_pieces_np,
        policy_targets=raw_policy_targets_np,
        action_masks=raw_action_masks_np,
    )

    boards = torch.from_numpy(boards_np).unsqueeze(1).to(device, non_blocking=True)
    aux = torch.from_numpy(aux_np).to(device, non_blocking=True)
    policy_targets = torch.from_numpy(policy_targets_np).to(device, non_blocking=True)
    value_targets = torch.from_numpy(value_targets_np).to(device, non_blocking=True)
    action_masks = torch.from_numpy(action_masks_np).to(device, non_blocking=True)
    return boards, aux, policy_targets, value_targets, action_masks


def build_tensor_dataset(
    data: np.lib.npyio.NpzFile,
    selected_global_indices: np.ndarray,
    mode: str,
    train_device: torch.device,
    included_groups: tuple[ExtraFeatureGroup, ...],
) -> OfflineTensorDataset:
    boards_np = data["boards"][selected_global_indices].astype(np.float32, copy=False)
    current_pieces_np = data["current_pieces"][selected_global_indices].astype(
        np.float32, copy=False
    )
    aux_np = build_aux_batch_from_npz(
        data,
        selected_global_indices,
        included_groups,
        current_pieces_np,
    ).astype(np.float32, copy=False)
    raw_policy_targets_np = data["policy_targets"][selected_global_indices]
    value_targets_np = data["value_targets"][selected_global_indices].astype(
        np.float32, copy=False
    )
    raw_action_masks_np = data["action_masks"][selected_global_indices]
    policy_targets_np, action_masks_np = normalize_policy_arrays(
        current_pieces=current_pieces_np,
        policy_targets=raw_policy_targets_np,
        action_masks=raw_action_masks_np,
    )

    boards = torch.from_numpy(boards_np).unsqueeze(1)
    aux = torch.from_numpy(aux_np)
    policy_targets = torch.from_numpy(policy_targets_np)
    value_targets = torch.from_numpy(value_targets_np)
    action_masks = torch.from_numpy(action_masks_np)

    if mode == "gpu":
        storage_device = train_device
        boards = boards.to(storage_device, non_blocking=True)
        aux = aux.to(storage_device, non_blocking=True)
        policy_targets = policy_targets.to(storage_device, non_blocking=True)
        value_targets = value_targets.to(storage_device, non_blocking=True)
        action_masks = action_masks.to(storage_device, non_blocking=True)
    else:
        storage_device = torch.device("cpu")

    return OfflineTensorDataset(
        boards=boards,
        aux=aux,
        policy_targets=policy_targets,
        value_targets=value_targets,
        action_masks=action_masks,
        storage_device=storage_device,
    )


def build_torch_batch(
    source: OfflineDataSource,
    local_indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if source.tensor_data is None:
        global_indices = source.selected_global_indices[local_indices]
        return build_torch_batch_from_npz(
            source.npz, global_indices, device, source.included_groups
        )

    tensor_data = source.tensor_data
    gather_indices = torch.from_numpy(local_indices.astype(np.int64, copy=False)).to(
        tensor_data.storage_device, non_blocking=True
    )
    boards = tensor_data.boards.index_select(0, gather_indices)
    aux = tensor_data.aux.index_select(0, gather_indices)
    policy_targets = tensor_data.policy_targets.index_select(0, gather_indices)
    value_targets = tensor_data.value_targets.index_select(0, gather_indices)
    action_masks = tensor_data.action_masks.index_select(0, gather_indices)

    if tensor_data.storage_device == device:
        return boards, aux, policy_targets, value_targets, action_masks

    return (
        boards.to(device, non_blocking=True),
        aux.to(device, non_blocking=True),
        policy_targets.to(device, non_blocking=True),
        value_targets.to(device, non_blocking=True),
        action_masks.to(device, non_blocking=True),
    )


def select_subset(indices: np.ndarray, max_examples: int, seed: int) -> np.ndarray:
    if max_examples <= 0 or max_examples >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_examples, replace=False)


def tensor_dataset_bytes(dataset: OfflineTensorDataset) -> int:
    tensors = (
        dataset.boards,
        dataset.aux,
        dataset.policy_targets,
        dataset.value_targets,
        dataset.action_masks,
    )
    return sum(t.numel() * t.element_size() for t in tensors)


def evaluate_losses(
    model: nn.Module,
    source: OfflineDataSource,
    local_indices: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
    value_loss_weight: float,
) -> dict[str, float]:
    total_sum = 0.0
    policy_sum = 0.0
    value_sum = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(local_indices), eval_batch_size):
            batch_indices = local_indices[start : start + eval_batch_size]
            boards, aux, policy_targets, value_targets, action_masks = (
                build_torch_batch(source, batch_indices, device)
            )
            total_loss, policy_loss, value_loss = compute_loss(
                model=model,
                boards=boards,
                aux_features=aux,
                policy_targets=policy_targets,
                value_targets=value_targets,
                action_masks=action_masks,
                value_loss_weight=value_loss_weight,
            )

            n = len(batch_indices)
            total_sum += total_loss.item() * n
            policy_sum += policy_loss.item() * n
            value_sum += value_loss.item() * n
            sample_count += n

    if sample_count == 0:
        raise ValueError("Cannot evaluate with empty local_indices")

    return {
        "total_loss": total_sum / sample_count,
        "policy_loss": policy_sum / sample_count,
        "value_loss": value_sum / sample_count,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_offline_variant(
    variant: FeatureVariant,
    model: nn.Module,
    source: OfflineDataSource,
    train_local_indices: np.ndarray,
    train_eval_local_indices: np.ndarray,
    val_eval_local_indices: np.ndarray,
    args: ScriptArgs,
    device: torch.device,
    schedule_seed: int,
) -> dict:
    optimizer = OptimizerBundle(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    rng = np.random.default_rng(schedule_seed)
    start_time = time.perf_counter()
    train_compute_seconds_total = 0.0
    eval_seconds_total = 0.0
    window_train_seconds = 0.0
    window_batches = 0
    window_examples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def record_eval(step: int) -> None:
        nonlocal eval_seconds_total
        eval_start = time.perf_counter()
        train_metrics = evaluate_losses(
            model=model,
            source=source,
            local_indices=train_eval_local_indices,
            device=device,
            eval_batch_size=args.eval_batch_size,
            value_loss_weight=args.value_loss_weight,
        )
        val_metrics = evaluate_losses(
            model=model,
            source=source,
            local_indices=val_eval_local_indices,
            device=device,
            eval_batch_size=args.eval_batch_size,
            value_loss_weight=args.value_loss_weight,
        )
        eval_seconds = time.perf_counter() - eval_start
        eval_seconds_total += eval_seconds
        elapsed_sec = time.perf_counter() - start_time
        eval_examples = len(train_eval_local_indices) + len(val_eval_local_indices)
        eval_examples_per_sec = (
            eval_examples / eval_seconds if eval_seconds > 0 else 0.0
        )
        train_examples_seen = step * args.batch_size
        epochs_seen = train_examples_seen / len(train_local_indices)
        train_batches_per_sec = (
            step / train_compute_seconds_total
            if train_compute_seconds_total > 0
            else 0.0
        )
        wall_batches_per_sec = step / elapsed_sec if elapsed_sec > 0 else 0.0
        row = {
            "step": step,
            "elapsed_sec": elapsed_sec,
            "eval_seconds": eval_seconds,
            "eval_examples_per_sec": eval_examples_per_sec,
            "train_total_loss": train_metrics["total_loss"],
            "train_policy_loss": train_metrics["policy_loss"],
            "train_value_loss": train_metrics["value_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_policy_loss": val_metrics["policy_loss"],
            "val_value_loss": val_metrics["value_loss"],
        }
        history.append(row)
        log_data = {
            "offline_step": step,
            "ablation_variant": variant.name,
            f"variants/{variant.wandb_prefix}/eval_train_total_loss": row[
                "train_total_loss"
            ],
            f"variants/{variant.wandb_prefix}/eval_train_policy_loss": row[
                "train_policy_loss"
            ],
            f"variants/{variant.wandb_prefix}/eval_train_value_loss": row[
                "train_value_loss"
            ],
            f"variants/{variant.wandb_prefix}/eval_val_total_loss": row[
                "val_total_loss"
            ],
            f"variants/{variant.wandb_prefix}/eval_val_policy_loss": row[
                "val_policy_loss"
            ],
            f"variants/{variant.wandb_prefix}/eval_val_value_loss": row[
                "val_value_loss"
            ],
            f"comparison/curves/eval_val_total_loss/{variant.wandb_prefix}": row[
                "val_total_loss"
            ],
            f"comparison/curves/eval_val_policy_loss/{variant.wandb_prefix}": row[
                "val_policy_loss"
            ],
            f"comparison/curves/eval_val_value_loss/{variant.wandb_prefix}": row[
                "val_value_loss"
            ],
            f"variants/{variant.wandb_prefix}/eval_seconds": eval_seconds,
            f"variants/{variant.wandb_prefix}/eval_examples_per_sec": eval_examples_per_sec,
            f"variants/{variant.wandb_prefix}/elapsed_sec": elapsed_sec,
            f"variants/{variant.wandb_prefix}/train_compute_seconds_total": train_compute_seconds_total,
            f"variants/{variant.wandb_prefix}/eval_seconds_total": eval_seconds_total,
            f"variants/{variant.wandb_prefix}/train_examples_seen": train_examples_seen,
            f"variants/{variant.wandb_prefix}/epochs_seen": epochs_seen,
            f"variants/{variant.wandb_prefix}/train_batches_per_sec": train_batches_per_sec,
            f"variants/{variant.wandb_prefix}/wall_batches_per_sec": wall_batches_per_sec,
        }
        if device.type == "cuda":
            log_data[f"variants/{variant.wandb_prefix}/gpu_mem_allocated_mb"] = (
                torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
            )
            log_data[f"variants/{variant.wandb_prefix}/gpu_mem_reserved_mb"] = (
                torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
            )
            log_data[f"variants/{variant.wandb_prefix}/gpu_mem_max_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            )
        wandb.log(log_data)
        logger.info(
            "Offline eval",
            variant=variant.name,
            step=step,
            train_total_loss=row["train_total_loss"],
            val_total_loss=row["val_total_loss"],
            train_policy_loss=row["train_policy_loss"],
            val_policy_loss=row["val_policy_loss"],
            train_value_loss=row["train_value_loss"],
            val_value_loss=row["val_value_loss"],
            eval_seconds=eval_seconds,
            eval_examples_per_sec=eval_examples_per_sec,
            train_batches_per_sec=train_batches_per_sec,
            wall_batches_per_sec=wall_batches_per_sec,
        )

    record_eval(step=0)

    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        positions = rng.integers(0, len(train_local_indices), size=args.batch_size)
        batch_indices = train_local_indices[positions]
        boards, aux, policy_targets, value_targets, action_masks = build_torch_batch(
            source, batch_indices, device
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, policy_loss, value_loss = compute_loss(
            model=model,
            boards=boards,
            aux_features=aux,
            policy_targets=policy_targets,
            value_targets=value_targets,
            action_masks=action_masks,
            value_loss_weight=args.value_loss_weight,
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip_norm
        )
        optimizer.step()
        step_seconds = time.perf_counter() - step_start
        train_compute_seconds_total += step_seconds
        window_train_seconds += step_seconds
        window_batches += 1
        window_examples += args.batch_size

        if step % args.log_train_metrics_every == 0 or step == args.steps:
            elapsed_sec = time.perf_counter() - start_time
            train_examples_seen = step * args.batch_size
            epochs_seen = train_examples_seen / len(train_local_indices)
            window_batches_per_sec = (
                window_batches / window_train_seconds
                if window_train_seconds > 0
                else 0.0
            )
            window_examples_per_sec = (
                window_examples / window_train_seconds
                if window_train_seconds > 0
                else 0.0
            )
            train_batches_per_sec = (
                step / train_compute_seconds_total
                if train_compute_seconds_total > 0
                else 0.0
            )
            wall_batches_per_sec = step / elapsed_sec if elapsed_sec > 0 else 0.0
            log_data = {
                "offline_step": step,
                "ablation_variant": variant.name,
                f"variants/{variant.wandb_prefix}/train_batch_total_loss": total_loss.item(),
                f"variants/{variant.wandb_prefix}/train_batch_policy_loss": policy_loss.item(),
                f"variants/{variant.wandb_prefix}/train_batch_value_loss": value_loss.item(),
                f"variants/{variant.wandb_prefix}/train_step_seconds": step_seconds,
                f"variants/{variant.wandb_prefix}/grad_norm": float(grad_norm.item()),
                f"variants/{variant.wandb_prefix}/learning_rate": optimizer.param_groups[
                    0
                ]["lr"],
                f"variants/{variant.wandb_prefix}/train_examples_seen": train_examples_seen,
                f"variants/{variant.wandb_prefix}/epochs_seen": epochs_seen,
                f"variants/{variant.wandb_prefix}/train_batches_per_sec": train_batches_per_sec,
                f"variants/{variant.wandb_prefix}/train_examples_per_sec": (
                    train_examples_seen / train_compute_seconds_total
                    if train_compute_seconds_total > 0
                    else 0.0
                ),
                f"variants/{variant.wandb_prefix}/wall_batches_per_sec": wall_batches_per_sec,
                f"variants/{variant.wandb_prefix}/window_batches_per_sec": window_batches_per_sec,
                f"variants/{variant.wandb_prefix}/window_examples_per_sec": window_examples_per_sec,
                f"variants/{variant.wandb_prefix}/elapsed_sec": elapsed_sec,
                f"variants/{variant.wandb_prefix}/train_compute_seconds_total": train_compute_seconds_total,
                f"variants/{variant.wandb_prefix}/eval_seconds_total": eval_seconds_total,
                f"variants/{variant.wandb_prefix}/progress_fraction": step / args.steps,
            }
            if device.type == "cuda":
                log_data[f"variants/{variant.wandb_prefix}/gpu_mem_allocated_mb"] = (
                    torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                )
                log_data[f"variants/{variant.wandb_prefix}/gpu_mem_reserved_mb"] = (
                    torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
                )
                log_data[
                    f"variants/{variant.wandb_prefix}/gpu_mem_max_allocated_mb"
                ] = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            wandb.log(log_data)
            window_train_seconds = 0.0
            window_batches = 0
            window_examples = 0

        if step % args.eval_interval == 0 or step == args.steps:
            record_eval(step=step)

    elapsed_sec = time.perf_counter() - start_time
    return {
        "variant_name": variant.name,
        "wandb_prefix": variant.wandb_prefix,
        "included_groups": [group.name for group in variant.included_groups],
        "num_parameters": count_parameters(model),
        "elapsed_sec": elapsed_sec,
        "final": history[-1],
        "history": history,
    }


def normalize_args_for_wandb(
    args: ScriptArgs,
    variants: list[FeatureVariant],
    extra_feature_groups: tuple[ExtraFeatureGroup, ...],
) -> dict:
    normalized = asdict(args)
    normalized["data_path"] = str(args.data_path)
    normalized["extra_feature_groups"] = [group.name for group in extra_feature_groups]
    normalized["feature_variants"] = [
        {
            "name": variant.name,
            "wandb_prefix": variant.wandb_prefix,
            "included_groups": [group.name for group in variant.included_groups],
            "aux_features": variant.aux_features,
        }
        for variant in variants
    ]
    return normalized


def extract_variant_series(
    results: list[dict], history_key: str
) -> tuple[list[int], list[list[float]], list[str]]:
    if not results:
        return [], [], []
    steps = [int(row["step"]) for row in results[0]["history"]]
    line_values_by_variant: list[list[float]] = []
    variant_names: list[str] = []
    for result in results:
        variant_steps = [int(row["step"]) for row in result["history"]]
        if variant_steps != steps:
            raise ValueError(
                "Variant eval step schedules differ; expected identical schedules for "
                "comparison charts"
            )
        line_values_by_variant.append(
            [float(row[history_key]) for row in result["history"]]
        )
        variant_names.append(str(result["variant_name"]))
    return steps, line_values_by_variant, variant_names


def log_main_comparison_chart(
    results: list[dict],
    history_key: str,
    chart_key: str,
    chart_title: str,
) -> None:
    steps, line_values_by_variant, variant_names = extract_variant_series(
        results=results,
        history_key=history_key,
    )
    if not steps:
        return
    wandb.log(
        {
            chart_key: wandb.plot.line_series(
                xs=steps,
                ys=line_values_by_variant,
                keys=variant_names,
                title=chart_title,
                xname="offline_step",
            )
        }
    )


def log_overlay_chart(
    results: list[dict],
    history_key: str,
    chart_key: str,
    chart_title: str,
) -> None:
    steps, line_values_by_variant, variant_names = extract_variant_series(
        results=results,
        history_key=history_key,
    )
    if not steps:
        return
    figure = go.Figure()
    for variant_name, line_values in zip(variant_names, line_values_by_variant):
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=line_values,
                mode="lines",
                name=variant_name,
                hovertemplate=(
                    "variant=%{fullData.name}<br>"
                    "offline_step=%{x}<br>"
                    f"{history_key}=%{{y:.6f}}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=chart_title,
        xaxis_title="offline_step",
        yaxis_title=history_key,
        legend_title_text="feature_variant",
        template="plotly_white",
    )
    wandb.log({chart_key: wandb.Plotly(figure)})


def rank_val_total_loss(result: dict) -> float:
    val_loss = float(result["final"]["val_total_loss"])
    if np.isfinite(val_loss):
        return val_loss
    return float("inf")


def main(args: ScriptArgs) -> None:
    validate_args(args)
    if not args.data_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.data_path}")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {args.data_path}")

    extra_feature_groups = get_extra_feature_groups(
        args.include_move_number_feature, args.max_placements
    )
    variants = build_feature_variants(extra_feature_groups)

    device_str = pick_device(args.device)
    device = torch.device(device_str)
    preload_mode = get_preload_mode(args)
    if preload_mode == "gpu" and device.type == "cpu":
        raise ValueError("preload_to_gpu requires a non-CPU device")
    logger.info("Using device", device=device_str)

    run = init_wandb_run(
        args, normalize_args_for_wandb(args, variants, extra_feature_groups)
    )
    wandb.define_metric("offline_step")
    wandb.define_metric("variants/*", step_metric="offline_step")
    wandb.define_metric(
        "comparison/curves/eval_val_total_loss/*", step_metric="offline_step"
    )
    wandb.define_metric(
        "comparison/curves/eval_val_policy_loss/*", step_metric="offline_step"
    )
    wandb.define_metric(
        "comparison/curves/eval_val_value_loss/*", step_metric="offline_step"
    )

    npz = np.load(args.data_path, mmap_mode="r")
    try:
        ensure_required_keys(npz, extra_feature_groups)
        total_examples = validate_shapes(npz, extra_feature_groups)
        selected_global_indices = np.arange(total_examples, dtype=np.int64)

        split_rng = np.random.default_rng(args.seed)
        split_rng.shuffle(selected_global_indices)
        if args.max_examples > 0:
            selected_global_indices = selected_global_indices[: args.max_examples]

        num_selected = len(selected_global_indices)
        split_point = int(num_selected * args.train_fraction)
        if split_point <= 0 or split_point >= num_selected:
            raise ValueError(
                "Invalid train/val split; adjust max_examples or train_fraction"
            )
        train_local_indices = np.arange(split_point, dtype=np.int64)
        val_local_indices = np.arange(split_point, num_selected, dtype=np.int64)

        train_eval_local_indices = select_subset(
            train_local_indices,
            max_examples=args.eval_examples,
            seed=args.seed + 1,
        )
        val_eval_local_indices = select_subset(
            val_local_indices,
            max_examples=args.eval_examples,
            seed=args.seed + 2,
        )

        logger.info(
            "Dataset split",
            total_examples=total_examples,
            used_examples=num_selected,
            train_examples=len(train_local_indices),
            val_examples=len(val_local_indices),
            train_eval_examples=len(train_eval_local_indices),
            val_eval_examples=len(val_eval_local_indices),
            preload_mode=preload_mode,
            feature_variants=[variant.name for variant in variants],
            extra_feature_groups=[group.name for group in extra_feature_groups],
        )
        wandb.log(
            {
                "dataset/total_examples": total_examples,
                "dataset/used_examples": num_selected,
                "dataset/train_examples": len(train_local_indices),
                "dataset/val_examples": len(val_local_indices),
                "dataset/train_eval_examples": len(train_eval_local_indices),
                "dataset/val_eval_examples": len(val_eval_local_indices),
                "dataset/preload_mode": preload_mode,
                "dataset/num_feature_variants": len(variants),
                "dataset/base_aux_features": BASE_AUX_FEATURES,
            }
        )

        results: list[dict] = []
        schedule_seed = args.seed + 12345
        for idx, variant in enumerate(variants):
            logger.info(
                "Starting variant",
                variant=variant.name,
                included_groups=[group.name for group in variant.included_groups],
                aux_features=variant.aux_features,
                variant_index=idx,
                num_variants=len(variants),
            )

            preload_start = time.perf_counter()
            tensor_data: OfflineTensorDataset | None = None
            if preload_mode != "none":
                tensor_data = build_tensor_dataset(
                    data=npz,
                    selected_global_indices=selected_global_indices,
                    mode=preload_mode,
                    train_device=device,
                    included_groups=variant.included_groups,
                )
            preload_sec = time.perf_counter() - preload_start
            source = OfflineDataSource(
                npz=npz,
                selected_global_indices=selected_global_indices,
                tensor_data=tensor_data,
                included_groups=variant.included_groups,
            )

            torch.manual_seed(args.seed + idx)
            model = GatedFeatureAblationTetrisNet(
                conv_filters=args.conv_filters,
                fc_hidden=args.fc_hidden,
                conv_kernel_size=args.conv_kernel_size,
                conv_padding=args.conv_padding,
                aux_hidden=args.aux_hidden,
                num_fusion_blocks=args.num_fusion_blocks,
                aux_features=variant.aux_features,
            ).to(device)

            log_data = {
                "offline_step": 0,
                "ablation_variant": variant.name,
                f"variants/{variant.wandb_prefix}/num_parameters": count_parameters(
                    model
                ),
                f"variants/{variant.wandb_prefix}/aux_features": variant.aux_features,
                f"variants/{variant.wandb_prefix}/extra_aux_features": (
                    variant.aux_features - BASE_AUX_FEATURES
                ),
                f"variants/{variant.wandb_prefix}/num_extra_feature_groups": len(
                    variant.included_groups
                ),
                f"variants/{variant.wandb_prefix}/preload_seconds": preload_sec,
            }
            if tensor_data is not None:
                log_data[f"variants/{variant.wandb_prefix}/preload_bytes"] = (
                    tensor_dataset_bytes(tensor_data)
                )
            wandb.log(log_data)

            result = train_offline_variant(
                variant=variant,
                model=model,
                source=source,
                train_local_indices=train_local_indices,
                train_eval_local_indices=train_eval_local_indices,
                val_eval_local_indices=val_eval_local_indices,
                args=args,
                device=device,
                schedule_seed=schedule_seed,
            )
            results.append(result)
            logger.info(
                "Completed variant",
                variant=variant.name,
                final_val_total_loss=result["final"]["val_total_loss"],
                elapsed_sec=result["elapsed_sec"],
            )

        if all(not np.isfinite(rank_val_total_loss(result)) for result in results):
            raise ValueError("All variants produced non-finite final val_total_loss")
        winner = min(results, key=rank_val_total_loss)
        comparison_log = {
            "comparison/winner": winner["variant_name"],
            "comparison/winner_is_no_extra_features": (
                1 if winner["variant_name"] == "no_extra_features" else 0
            ),
        }
        for result in results:
            prefix = result["wandb_prefix"]
            comparison_log[f"comparison/final_val_total_loss/{prefix}"] = result[
                "final"
            ]["val_total_loss"]
            comparison_log[f"comparison/final_val_policy_loss/{prefix}"] = result[
                "final"
            ]["val_policy_loss"]
            comparison_log[f"comparison/final_val_value_loss/{prefix}"] = result[
                "final"
            ]["val_value_loss"]
            comparison_log[f"comparison/final_train_total_loss/{prefix}"] = result[
                "final"
            ]["train_total_loss"]
        wandb.log(comparison_log)

        log_main_comparison_chart(
            results=results,
            history_key="val_total_loss",
            chart_key="comparison/charts/eval_val_total_loss_main",
            chart_title="Eval Val Total Loss by Feature Variant (Main)",
        )
        log_main_comparison_chart(
            results=results,
            history_key="val_policy_loss",
            chart_key="comparison/charts/eval_val_policy_loss_main",
            chart_title="Eval Val Policy Loss by Feature Variant (Main)",
        )
        log_main_comparison_chart(
            results=results,
            history_key="val_value_loss",
            chart_key="comparison/charts/eval_val_value_loss_main",
            chart_title="Eval Val Value Loss by Feature Variant (Main)",
        )
        log_overlay_chart(
            results=results,
            history_key="val_total_loss",
            chart_key="charts/eval_val_total_loss_overlay",
            chart_title="Eval Val Total Loss by Feature Variant",
        )
        log_overlay_chart(
            results=results,
            history_key="val_policy_loss",
            chart_key="charts/eval_val_policy_loss_overlay",
            chart_title="Eval Val Policy Loss by Feature Variant",
        )
        log_overlay_chart(
            results=results,
            history_key="val_value_loss",
            chart_key="charts/eval_val_value_loss_overlay",
            chart_title="Eval Val Value Loss by Feature Variant",
        )

        run.summary["winner_by_final_val_total_loss"] = winner["variant_name"]
        run.summary["winner_final_val_total_loss"] = winner["final"]["val_total_loss"]
        run.summary["num_variants"] = len(results)
        for result in results:
            result_key = str(result["wandb_prefix"])
            run.summary[f"{result_key}_final_val_total_loss"] = result["final"][
                "val_total_loss"
            ]
            run.summary[f"{result_key}_final_val_policy_loss"] = result["final"][
                "val_policy_loss"
            ]
            run.summary[f"{result_key}_final_val_value_loss"] = result["final"][
                "val_value_loss"
            ]

        logger.info(
            "Offline feature ablation comparison complete",
            winner=winner["variant_name"],
            winner_final_val_total_loss=winner["final"]["val_total_loss"],
            variants=[result["variant_name"] for result in results],
        )
    finally:
        npz.close()
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
