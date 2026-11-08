from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from simple_parsing import parse

from tetris_bot.action_space import (
    LEGACY_NUM_ACTIONS,
    adapt_legacy_action_masks,
    adapt_legacy_policy_targets,
)
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
)
from tetris_bot.ml.loss import compute_loss
from tetris_bot.ml.network import (
    AUX_FEATURES,
    BOARD_STATS_FEATURES,
    PIECE_AUX_FEATURES,
    ROW_FILL_COUNT_FEATURES,
)
from tetris_bot.ml.optimizer import OptimizerBundle

logger = structlog.get_logger()


class _PreloadArgs(Protocol):
    preload_to_gpu: bool
    preload_to_ram: bool


class _TrainOfflineArgs(_PreloadArgs, Protocol):
    learning_rate: float
    weight_decay: float
    steps: int
    batch_size: int
    value_loss_weight: float
    grad_clip_norm: float
    log_train_metrics_every: int
    eval_interval: int
    eval_batch_size: int


class _OfflineScriptArgs(_TrainOfflineArgs, Protocol):
    data_path: Path
    seed: int
    max_examples: int
    train_fraction: float
    eval_examples: int
    wandb_project: str
    wandb_run_name: str | None
    wandb_entity: str | None
    wandb_tags: list[str]


REQUIRED_NPZ_KEYS = (
    "boards",
    "current_pieces",
    "hold_pieces",
    "hold_available",
    "next_queue",
    "placement_counts",
    "combos",
    "back_to_back",
    "next_hidden_piece_probs",
    "column_heights",
    "max_column_heights",
    "row_fill_counts",
    "total_blocks",
    "bumpiness",
    "holes",
    "overhang_fields",
    "policy_targets",
    "value_targets",
    "action_masks",
)


@dataclass
class ScriptArgs:
    data_path: Path  # Path to offline replay buffer NPZ
    device: str = "auto"  # auto/cpu/cuda/mps
    seed: int = 123
    max_examples: int = 0  # 0 = use all examples in NPZ
    train_fraction: float = 0.9
    steps: int = 20000
    batch_size: int = 1024
    eval_interval: int = 100
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

    match_aux_hidden_min: int = 8
    match_aux_hidden_max: int = 256
    match_aux_hidden_step: int = 8
    match_fusion_hidden_min: int = 32
    match_fusion_hidden_max: int = 384
    match_fusion_hidden_step: int = 8
    match_num_fusion_blocks_options: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3]
    )
    match_param_tolerance: float = 0.01  # Relative tolerance
    match_flop_tolerance: float = (  # Relative tolerance on cache-weighted FLOPs
        0.01
    )
    match_flop_weight: float = 1.0  # Score weight for cache-weighted FLOP error
    cache_hit_rate_for_matching: float = (
        0.96  # Expected board-cache hit rate for effective FLOP matching
    )

    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "architecture-compare"]
    )


@dataclass
class MatchedGatedConfig:
    aux_hidden: int
    fusion_hidden: int
    num_fusion_blocks: int
    params: int
    miss_only_flops: int
    hit_path_flops: int
    full_flops: int
    effective_flops: float
    param_rel_error: float
    flop_rel_error: float
    score: float


@dataclass
class FlopBreakdown:
    miss_only: int
    hit_path: int
    full: int
    effective: float


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


@dataclass
class OfflineDatasetSetup:
    source: OfflineDataSource
    train_local_indices: np.ndarray
    val_local_indices: np.ndarray
    train_eval_local_indices: np.ndarray
    val_eval_local_indices: np.ndarray
    total_examples: int
    num_selected: int
    preload_sec: float


class ResidualFusionBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(x))
        h = self.fc1(h)
        h = F.relu(self.ln2(h))
        h = self.fc2(h)
        return x + h


class BaselineConcatFCTetrisNet(nn.Module):
    def __init__(
        self,
        conv_filters: list[int],
        fc_hidden: int,
        conv_kernel_size: int,
        conv_padding: int,
    ):
        super().__init__()
        if len(conv_filters) != 2:
            raise ValueError(
                f"Expected exactly 2 convolutional filters, got {len(conv_filters)}"
            )
        conv0 = conv_filters[0]
        conv1 = conv_filters[1]

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

        conv_flat_size = BOARD_HEIGHT * BOARD_WIDTH * conv1
        self.fc1 = nn.Linear(conv_flat_size + AUX_FEATURES, fc_hidden)
        self.ln1 = nn.LayerNorm(fc_hidden)
        self.policy_head = nn.Linear(fc_hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(fc_hidden, 1)

    def forward(
        self, board: torch.Tensor, aux_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, aux_features], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        return self.policy_head(x), self.value_head(x)


class GatedFusionTetrisNet(nn.Module):
    def __init__(
        self,
        conv_filters: list[int],
        conv_kernel_size: int,
        conv_padding: int,
        aux_hidden: int,
        fusion_hidden: int,
        num_fusion_blocks: int,
    ):
        super().__init__()
        if len(conv_filters) != 2:
            raise ValueError(
                f"Expected exactly 2 convolutional filters, got {len(conv_filters)}"
            )
        if num_fusion_blocks < 0:
            raise ValueError("num_fusion_blocks must be >= 0")

        conv0 = conv_filters[0]
        conv1 = conv_filters[1]
        conv_flat_size = BOARD_HEIGHT * BOARD_WIDTH * conv1

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

        self.board_proj = nn.Linear(
            conv_flat_size + BOARD_STATS_FEATURES, fusion_hidden
        )
        self.aux_fc = nn.Linear(PIECE_AUX_FEATURES, aux_hidden)
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
        piece_aux = aux_features[:, :PIECE_AUX_FEATURES]
        board_stats = aux_features[:, PIECE_AUX_FEATURES:]
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)

        board_h = self.board_proj(torch.cat([x, board_stats], dim=1))
        aux_h = F.relu(self.aux_ln(self.aux_fc(piece_aux)))
        gate = torch.sigmoid(self.gate_fc(aux_h))

        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))
        for block in self.fusion_blocks:
            fused = block(fused)

        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)
        return policy_logits, value


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def conv_out_size(size: int, kernel_size: int, padding: int) -> int:
    return size + 2 * padding - kernel_size + 1


def conv_output_hw(kernel_size: int, padding: int) -> tuple[int, int, int, int]:
    h1 = conv_out_size(BOARD_HEIGHT, kernel_size, padding)
    w1 = conv_out_size(BOARD_WIDTH, kernel_size, padding)
    h2 = conv_out_size(h1, kernel_size, padding)
    w2 = conv_out_size(w1, kernel_size, padding)
    if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
        raise ValueError(
            "Invalid conv output shape; adjust conv_kernel_size/conv_padding "
            f"(got h2={h2}, w2={w2})"
        )
    return h1, w1, h2, w2


def baseline_flop_breakdown(args: ScriptArgs, cache_hit_rate: float) -> FlopBreakdown:
    conv0, conv1 = args.conv_filters
    k = args.conv_kernel_size
    h1, w1, h2, w2 = conv_output_hw(k, args.conv_padding)
    conv_flat = h2 * w2 * conv1

    miss_only = 0
    miss_only += h1 * w1 * conv0 * (2 * 1 * k * k + 1)
    miss_only += 2 * h1 * w1 * conv0
    miss_only += h2 * w2 * conv1 * (2 * conv0 * k * k + 1)
    miss_only += 2 * h2 * w2 * conv1
    miss_only += h1 * w1 * conv0 + h2 * w2 * conv1
    # Cached board embedding in Rust includes FC(board) + bias
    miss_only += args.fc_hidden * (2 * conv_flat + 1)

    hit_path = 0
    # Aux contribution and merge with cached board embedding
    hit_path += args.fc_hidden * (2 * AUX_FEATURES + 1)
    hit_path += args.fc_hidden
    # LayerNorm + ReLU and heads
    hit_path += 5 * args.fc_hidden
    hit_path += args.fc_hidden
    hit_path += NUM_ACTIONS * (2 * args.fc_hidden + 1)
    hit_path += 2 * args.fc_hidden + 1

    full = miss_only + hit_path
    effective = hit_path + (1.0 - cache_hit_rate) * miss_only
    return FlopBreakdown(
        miss_only=miss_only,
        hit_path=hit_path,
        full=full,
        effective=effective,
    )


def gated_flop_breakdown(
    args: ScriptArgs,
    aux_hidden: int,
    fusion_hidden: int,
    num_fusion_blocks: int,
    cache_hit_rate: float,
) -> FlopBreakdown:
    conv0, conv1 = args.conv_filters
    k = args.conv_kernel_size
    h1, w1, h2, w2 = conv_output_hw(k, args.conv_padding)
    conv_flat = h2 * w2 * conv1

    miss_only = 0
    miss_only += h1 * w1 * conv0 * (2 * 1 * k * k + 1)
    miss_only += 2 * h1 * w1 * conv0
    miss_only += h2 * w2 * conv1 * (2 * conv0 * k * k + 1)
    miss_only += 2 * h2 * w2 * conv1
    miss_only += h1 * w1 * conv0 + h2 * w2 * conv1
    # Cached board path includes board projection (conv_flat + board_stats).
    miss_only += fusion_hidden * (2 * (conv_flat + BOARD_STATS_FEATURES) + 1)

    hit_path = 0
    hit_path += aux_hidden * (2 * PIECE_AUX_FEATURES + 1)
    hit_path += 5 * aux_hidden
    hit_path += aux_hidden
    hit_path += fusion_hidden * (2 * aux_hidden + 1)
    hit_path += fusion_hidden * (2 * aux_hidden + 1)
    # board_h * (1 + gate) + aux_proj
    hit_path += 3 * fusion_hidden
    hit_path += 5 * fusion_hidden
    hit_path += fusion_hidden

    for _ in range(num_fusion_blocks):
        hit_path += 5 * fusion_hidden
        hit_path += fusion_hidden
        hit_path += fusion_hidden * (2 * fusion_hidden + 1)
        hit_path += 5 * fusion_hidden
        hit_path += fusion_hidden
        hit_path += fusion_hidden * (2 * fusion_hidden + 1)
        hit_path += fusion_hidden

    hit_path += NUM_ACTIONS * (2 * fusion_hidden + 1)
    hit_path += 2 * fusion_hidden + 1

    full = miss_only + hit_path
    effective = hit_path + (1.0 - cache_hit_rate) * miss_only
    return FlopBreakdown(
        miss_only=miss_only,
        hit_path=hit_path,
        full=full,
        effective=effective,
    )


def gated_parameter_count(
    args: ScriptArgs,
    aux_hidden: int,
    fusion_hidden: int,
    num_fusion_blocks: int,
) -> int:
    conv0, conv1 = args.conv_filters
    k = args.conv_kernel_size
    _, _, h2, w2 = conv_output_hw(k, args.conv_padding)
    conv_flat = h2 * w2 * conv1

    params = 0
    params += conv0 * k * k + conv0
    params += 2 * conv0
    params += conv1 * conv0 * k * k + conv1
    params += 2 * conv1

    params += (conv_flat + BOARD_STATS_FEATURES) * fusion_hidden + fusion_hidden
    params += PIECE_AUX_FEATURES * aux_hidden + aux_hidden
    params += 2 * aux_hidden
    params += aux_hidden * fusion_hidden + fusion_hidden
    params += aux_hidden * fusion_hidden + fusion_hidden
    params += 2 * fusion_hidden

    for _ in range(num_fusion_blocks):
        params += 2 * fusion_hidden
        params += fusion_hidden * fusion_hidden + fusion_hidden
        params += 2 * fusion_hidden
        params += fusion_hidden * fusion_hidden + fusion_hidden

    params += fusion_hidden * NUM_ACTIONS + NUM_ACTIONS
    params += fusion_hidden + 1
    return params


def find_matched_gated_config(
    args: ScriptArgs,
    target_params: int,
    target_effective_flops: float,
) -> MatchedGatedConfig:
    best: MatchedGatedConfig | None = None
    for num_blocks in args.match_num_fusion_blocks_options:
        for aux_hidden in range(
            args.match_aux_hidden_min,
            args.match_aux_hidden_max + 1,
            args.match_aux_hidden_step,
        ):
            for fusion_hidden in range(
                args.match_fusion_hidden_min,
                args.match_fusion_hidden_max + 1,
                args.match_fusion_hidden_step,
            ):
                params = gated_parameter_count(
                    args, aux_hidden, fusion_hidden, num_blocks
                )
                flops = gated_flop_breakdown(
                    args=args,
                    aux_hidden=aux_hidden,
                    fusion_hidden=fusion_hidden,
                    num_fusion_blocks=num_blocks,
                    cache_hit_rate=args.cache_hit_rate_for_matching,
                )
                param_rel_error = abs(params - target_params) / target_params
                flop_rel_error = (
                    abs(flops.effective - target_effective_flops)
                    / target_effective_flops
                )
                score = param_rel_error + args.match_flop_weight * flop_rel_error
                candidate = MatchedGatedConfig(
                    aux_hidden=aux_hidden,
                    fusion_hidden=fusion_hidden,
                    num_fusion_blocks=num_blocks,
                    params=params,
                    miss_only_flops=flops.miss_only,
                    hit_path_flops=flops.hit_path,
                    full_flops=flops.full,
                    effective_flops=flops.effective,
                    param_rel_error=param_rel_error,
                    flop_rel_error=flop_rel_error,
                    score=score,
                )
                if best is None or candidate.score < best.score:
                    best = candidate

    if best is None:
        raise ValueError("No gated architecture candidates were evaluated")

    if best.param_rel_error > args.match_param_tolerance:
        raise ValueError(
            "Unable to match parameter count within tolerance. "
            f"Best relative error={best.param_rel_error:.6f}, "
            f"tolerance={args.match_param_tolerance:.6f}. "
            "Expand match search ranges."
        )
    if best.flop_rel_error > args.match_flop_tolerance:
        raise ValueError(
            "Unable to match cache-weighted FLOPs within tolerance. "
            f"Best relative error={best.flop_rel_error:.6f}, "
            f"tolerance={args.match_flop_tolerance:.6f}. "
            "Expand match search ranges."
        )
    return best


def validate_args(args: ScriptArgs) -> None:
    validate_common_offline_args(args)
    if len(args.conv_filters) != 2:
        raise ValueError("conv_filters must contain exactly two values")
    if args.conv_kernel_size <= 0:
        raise ValueError("conv_kernel_size must be > 0")
    if args.match_aux_hidden_step <= 0:
        raise ValueError("match_aux_hidden_step must be > 0")
    if args.match_fusion_hidden_step <= 0:
        raise ValueError("match_fusion_hidden_step must be > 0")
    if args.match_aux_hidden_min <= 0 or args.match_aux_hidden_max <= 0:
        raise ValueError("match_aux_hidden bounds must be > 0")
    if args.match_fusion_hidden_min <= 0 or args.match_fusion_hidden_max <= 0:
        raise ValueError("match_fusion_hidden bounds must be > 0")
    if args.match_aux_hidden_min > args.match_aux_hidden_max:
        raise ValueError("match_aux_hidden_min must be <= match_aux_hidden_max")
    if args.match_fusion_hidden_min > args.match_fusion_hidden_max:
        raise ValueError("match_fusion_hidden_min must be <= match_fusion_hidden_max")
    if any(blocks < 0 for blocks in args.match_num_fusion_blocks_options):
        raise ValueError(
            "match_num_fusion_blocks_options must contain only >= 0 values"
        )
    if args.match_param_tolerance <= 0:
        raise ValueError("match_param_tolerance must be > 0")
    if args.match_flop_tolerance <= 0:
        raise ValueError("match_flop_tolerance must be > 0")
    if args.match_flop_weight <= 0:
        raise ValueError("match_flop_weight must be > 0")
    if not 0.0 <= args.cache_hit_rate_for_matching <= 1.0:
        raise ValueError("cache_hit_rate_for_matching must be in [0, 1]")


def ensure_required_keys(data: np.lib.npyio.NpzFile) -> None:
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in data]
    if missing:
        raise KeyError(f"NPZ is missing required keys: {missing}")


def validate_shapes(data: np.lib.npyio.NpzFile) -> int:
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
    if data["column_heights"].shape != (n, BOARD_WIDTH):
        raise ValueError(f"column_heights must have shape ({n}, {BOARD_WIDTH})")
    if data["max_column_heights"].shape != (n,):
        raise ValueError("max_column_heights must have shape (N,)")
    if data["row_fill_counts"].shape != (n, ROW_FILL_COUNT_FEATURES):
        raise ValueError(
            f"row_fill_counts must have shape ({n}, {ROW_FILL_COUNT_FEATURES})"
        )
    if data["total_blocks"].shape != (n,):
        raise ValueError("total_blocks must have shape (N,)")
    if data["bumpiness"].shape != (n,):
        raise ValueError("bumpiness must have shape (N,)")
    if data["holes"].shape != (n,):
        raise ValueError("holes must have shape (N,)")
    if data["overhang_fields"].shape != (n,):
        raise ValueError("overhang_fields must have shape (N,)")
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

    per_example_keys = (
        "current_pieces",
        "hold_pieces",
        "hold_available",
        "next_queue",
        "placement_counts",
        "combos",
        "back_to_back",
        "next_hidden_piece_probs",
        "column_heights",
        "max_column_heights",
        "row_fill_counts",
        "total_blocks",
        "bumpiness",
        "holes",
        "overhang_fields",
        "policy_targets",
        "value_targets",
        "action_masks",
    )
    for key in per_example_keys:
        if int(data[key].shape[0]) != n:
            raise ValueError(
                f"{key} has inconsistent first dimension: {data[key].shape[0]} vs {n}"
            )
    return n


def get_preload_mode(args: _PreloadArgs) -> str:
    if args.preload_to_gpu and args.preload_to_ram:
        raise ValueError("preload_to_gpu and preload_to_ram cannot both be true")
    if args.preload_to_gpu:
        return "gpu"
    if args.preload_to_ram:
        return "ram"
    return "none"


def build_aux_batch_from_npz(
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
    column_heights = data["column_heights"][global_indices].astype(
        np.float32, copy=False
    )
    max_column_heights = (
        data["max_column_heights"][global_indices].astype(np.float32).reshape(-1, 1)
    )
    row_fill_counts = data["row_fill_counts"][global_indices].astype(
        np.float32, copy=False
    )
    total_blocks = (
        data["total_blocks"][global_indices].astype(np.float32).reshape(-1, 1)
    )
    bumpiness = data["bumpiness"][global_indices].astype(np.float32).reshape(-1, 1)
    holes = data["holes"][global_indices].astype(np.float32).reshape(-1, 1)
    overhang_fields = (
        data["overhang_fields"][global_indices].astype(np.float32).reshape(-1, 1)
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
            column_heights,
            max_column_heights,
            row_fill_counts,
            total_blocks,
            bumpiness,
            holes,
            overhang_fields,
        ],
        axis=1,
    )


def normalize_policy_arrays(
    current_pieces: np.ndarray,
    policy_targets: np.ndarray,
    action_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if policy_targets.shape != action_masks.shape:
        raise ValueError(
            "policy_targets/action_masks shape mismatch: "
            f"{policy_targets.shape} vs {action_masks.shape}"
        )
    if policy_targets.shape[1] == NUM_ACTIONS:
        return (
            policy_targets.astype(np.float32, copy=False),
            action_masks.astype(np.bool_, copy=False),
        )
    if policy_targets.shape[1] != LEGACY_NUM_ACTIONS:
        raise ValueError(
            "Unsupported action width for policy_targets/action_masks: "
            f"{policy_targets.shape[1]}"
        )
    return (
        adapt_legacy_policy_targets(current_pieces, policy_targets),
        adapt_legacy_action_masks(current_pieces, action_masks),
    )


def build_torch_batch_from_npz(
    data: np.lib.npyio.NpzFile,
    global_indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards_np = data["boards"][global_indices]
    current_pieces_np = data["current_pieces"][global_indices].astype(
        np.float32, copy=False
    )
    aux_np = build_aux_batch_from_npz(data, global_indices, current_pieces_np)
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
) -> OfflineTensorDataset:
    boards_np = data["boards"][selected_global_indices]
    current_pieces_np = data["current_pieces"][selected_global_indices].astype(
        np.float32, copy=False
    )
    aux_np = build_aux_batch_from_npz(
        data,
        selected_global_indices,
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
        return build_torch_batch_from_npz(source.npz, global_indices, device)

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


def validate_common_offline_args(args: _OfflineScriptArgs) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be > 0")
    if args.log_train_metrics_every <= 0:
        raise ValueError("log_train_metrics_every must be > 0")
    if args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
    if args.eval_examples <= 0:
        raise ValueError("eval_examples must be > 0")
    if args.max_examples < 0:
        raise ValueError("max_examples must be >= 0")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if args.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be > 0")


def setup_offline_dataset(
    npz: np.lib.npyio.NpzFile,
    seed: int,
    max_examples: int,
    train_fraction: float,
    eval_examples: int,
    preload_mode: str,
    device: torch.device,
) -> OfflineDatasetSetup:
    ensure_required_keys(npz)
    total_examples = validate_shapes(npz)
    selected_global_indices = np.arange(total_examples, dtype=np.int64)

    split_rng = np.random.default_rng(seed)
    split_rng.shuffle(selected_global_indices)
    if max_examples > 0:
        selected_global_indices = selected_global_indices[:max_examples]

    num_selected = len(selected_global_indices)
    split_point = int(num_selected * train_fraction)
    if split_point <= 0 or split_point >= num_selected:
        raise ValueError(
            "Invalid train/val split; adjust max_examples or train_fraction"
        )

    train_local_indices = np.arange(split_point, dtype=np.int64)
    val_local_indices = np.arange(split_point, num_selected, dtype=np.int64)

    preload_start = time.perf_counter()
    tensor_data: OfflineTensorDataset | None = None
    if preload_mode != "none":
        tensor_data = build_tensor_dataset(
            data=npz,
            selected_global_indices=selected_global_indices,
            mode=preload_mode,
            train_device=device,
        )
    preload_sec = time.perf_counter() - preload_start
    source = OfflineDataSource(
        npz=npz,
        selected_global_indices=selected_global_indices,
        tensor_data=tensor_data,
    )

    train_eval_local_indices = select_subset(
        train_local_indices,
        max_examples=eval_examples,
        seed=seed + 1,
    )
    val_eval_local_indices = select_subset(
        val_local_indices,
        max_examples=eval_examples,
        seed=seed + 2,
    )

    dataset_log: dict[str, float | int | str] = {
        "dataset/total_examples": total_examples,
        "dataset/used_examples": num_selected,
        "dataset/train_examples": len(train_local_indices),
        "dataset/val_examples": len(val_local_indices),
        "dataset/train_eval_examples": len(train_eval_local_indices),
        "dataset/val_eval_examples": len(val_eval_local_indices),
        "dataset/preload_mode": preload_mode,
        "dataset/preload_seconds": preload_sec,
    }
    if tensor_data is not None:
        dataset_log["dataset/preload_bytes"] = tensor_dataset_bytes(tensor_data)
    wandb.log(dataset_log)

    logger.info(
        "Dataset split",
        total_examples=total_examples,
        used_examples=num_selected,
        train_examples=len(train_local_indices),
        val_examples=len(val_local_indices),
        train_eval_examples=len(train_eval_local_indices),
        val_eval_examples=len(val_eval_local_indices),
        preload_mode=preload_mode,
        preload_seconds=preload_sec,
    )

    return OfflineDatasetSetup(
        source=source,
        train_local_indices=train_local_indices,
        val_local_indices=val_local_indices,
        train_eval_local_indices=train_eval_local_indices,
        val_eval_local_indices=val_eval_local_indices,
        total_examples=total_examples,
        num_selected=num_selected,
        preload_sec=preload_sec,
    )


def init_wandb_run(args: _OfflineScriptArgs, config: dict):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=config,
    )
    run = wandb.run
    if run is None:
        raise RuntimeError("wandb.init did not create a run")
    return run


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


def train_offline_model(
    model_name: str,
    wandb_prefix: str,
    model: nn.Module,
    source: OfflineDataSource,
    train_local_indices: np.ndarray,
    train_eval_local_indices: np.ndarray,
    val_eval_local_indices: np.ndarray,
    args: _TrainOfflineArgs,
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
            "train_batches_per_sec": train_batches_per_sec,
            "train_examples_per_sec": (
                train_examples_seen / train_compute_seconds_total
                if train_compute_seconds_total > 0
                else 0.0
            ),
            "wall_batches_per_sec": wall_batches_per_sec,
            "train_examples_seen": train_examples_seen,
            "epochs_seen": epochs_seen,
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
            f"{wandb_prefix}/eval_train_total_loss": row["train_total_loss"],
            f"{wandb_prefix}/eval_train_policy_loss": row["train_policy_loss"],
            f"{wandb_prefix}/eval_train_value_loss": row["train_value_loss"],
            f"{wandb_prefix}/eval_val_total_loss": row["val_total_loss"],
            f"{wandb_prefix}/eval_val_policy_loss": row["val_policy_loss"],
            f"{wandb_prefix}/eval_val_value_loss": row["val_value_loss"],
            f"{wandb_prefix}/eval_seconds": eval_seconds,
            f"{wandb_prefix}/eval_examples_per_sec": eval_examples_per_sec,
            f"{wandb_prefix}/elapsed_sec": elapsed_sec,
            f"{wandb_prefix}/train_compute_seconds_total": train_compute_seconds_total,
            f"{wandb_prefix}/eval_seconds_total": eval_seconds_total,
            f"{wandb_prefix}/train_examples_seen": train_examples_seen,
            f"{wandb_prefix}/epochs_seen": epochs_seen,
            f"{wandb_prefix}/train_batches_per_sec": train_batches_per_sec,
            f"{wandb_prefix}/wall_batches_per_sec": wall_batches_per_sec,
        }
        if device.type == "cuda":
            log_data[f"{wandb_prefix}/gpu_mem_allocated_mb"] = (
                torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
            )
            log_data[f"{wandb_prefix}/gpu_mem_reserved_mb"] = (
                torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
            )
            log_data[f"{wandb_prefix}/gpu_mem_max_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            )
        wandb.log(log_data)
        logger.info(
            "Offline eval",
            model=model_name,
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
                f"{wandb_prefix}/train_batch_total_loss": total_loss.item(),
                f"{wandb_prefix}/train_batch_policy_loss": policy_loss.item(),
                f"{wandb_prefix}/train_batch_value_loss": value_loss.item(),
                f"{wandb_prefix}/train_step_seconds": step_seconds,
                f"{wandb_prefix}/grad_norm": float(grad_norm.item()),
                f"{wandb_prefix}/learning_rate": optimizer.param_groups[0]["lr"],
                f"{wandb_prefix}/train_examples_seen": train_examples_seen,
                f"{wandb_prefix}/epochs_seen": epochs_seen,
                f"{wandb_prefix}/train_batches_per_sec": train_batches_per_sec,
                f"{wandb_prefix}/train_examples_per_sec": (
                    train_examples_seen / train_compute_seconds_total
                    if train_compute_seconds_total > 0
                    else 0.0
                ),
                f"{wandb_prefix}/wall_batches_per_sec": wall_batches_per_sec,
                f"{wandb_prefix}/window_batches_per_sec": window_batches_per_sec,
                f"{wandb_prefix}/window_examples_per_sec": window_examples_per_sec,
                f"{wandb_prefix}/elapsed_sec": elapsed_sec,
                f"{wandb_prefix}/train_compute_seconds_total": train_compute_seconds_total,
                f"{wandb_prefix}/eval_seconds_total": eval_seconds_total,
                f"{wandb_prefix}/progress_fraction": step / args.steps,
            }
            if device.type == "cuda":
                log_data[f"{wandb_prefix}/gpu_mem_allocated_mb"] = (
                    torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                )
                log_data[f"{wandb_prefix}/gpu_mem_reserved_mb"] = (
                    torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
                )
                log_data[f"{wandb_prefix}/gpu_mem_max_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                )
            wandb.log(log_data)
            window_train_seconds = 0.0
            window_batches = 0
            window_examples = 0

        if step % args.eval_interval == 0 or step == args.steps:
            record_eval(step=step)

    elapsed_sec = time.perf_counter() - start_time
    final_metrics = history[-1]
    return {
        "model_name": model_name,
        "num_parameters": count_parameters(model),
        "elapsed_sec": elapsed_sec,
        "final": final_metrics,
        "history": history,
    }


def normalize_args_for_wandb(args: ScriptArgs) -> dict:
    normalized = asdict(args)
    normalized["data_path"] = str(args.data_path)
    return normalized


def main(args: ScriptArgs) -> None:
    validate_args(args)
    if not args.data_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.data_path}")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {args.data_path}")

    device_str = pick_device(args.device)
    device = torch.device(device_str)
    preload_mode = get_preload_mode(args)
    if preload_mode == "gpu" and device.type == "cpu":
        raise ValueError("preload_to_gpu requires a non-CPU device")
    logger.info("Using device", device=device_str)

    run = init_wandb_run(args, normalize_args_for_wandb(args))
    wandb.define_metric("offline_step")
    wandb.define_metric("baseline/*", step_metric="offline_step")
    wandb.define_metric("gated/*", step_metric="offline_step")

    npz = np.load(args.data_path, mmap_mode="r")
    try:
        setup = setup_offline_dataset(
            npz=npz,
            seed=args.seed,
            max_examples=args.max_examples,
            train_fraction=args.train_fraction,
            eval_examples=args.eval_examples,
            preload_mode=preload_mode,
            device=device,
        )
        source = setup.source
        train_local_indices = setup.train_local_indices
        train_eval_local_indices = setup.train_eval_local_indices
        val_eval_local_indices = setup.val_eval_local_indices

        torch.manual_seed(args.seed)
        baseline_model = BaselineConcatFCTetrisNet(
            conv_filters=args.conv_filters,
            fc_hidden=args.fc_hidden,
            conv_kernel_size=args.conv_kernel_size,
            conv_padding=args.conv_padding,
        ).to(device)
        baseline_params = count_parameters(baseline_model)
        baseline_flops = baseline_flop_breakdown(
            args=args,
            cache_hit_rate=args.cache_hit_rate_for_matching,
        )

        matched = find_matched_gated_config(
            args=args,
            target_params=baseline_params,
            target_effective_flops=baseline_flops.effective,
        )

        torch.manual_seed(args.seed)
        gated_model = GatedFusionTetrisNet(
            conv_filters=args.conv_filters,
            conv_kernel_size=args.conv_kernel_size,
            conv_padding=args.conv_padding,
            aux_hidden=matched.aux_hidden,
            fusion_hidden=matched.fusion_hidden,
            num_fusion_blocks=matched.num_fusion_blocks,
        ).to(device)
        gated_params_actual = count_parameters(gated_model)

        if gated_params_actual != matched.params:
            raise ValueError(
                "Analytical gated parameter count does not match instantiated model: "
                f"formula={matched.params}, actual={gated_params_actual}"
            )

        logger.info(
            "Matched architectures",
            baseline_params=baseline_params,
            baseline_miss_only_flops=baseline_flops.miss_only,
            baseline_hit_path_flops=baseline_flops.hit_path,
            baseline_full_flops=baseline_flops.full,
            baseline_effective_flops=baseline_flops.effective,
            gated_params=matched.params,
            gated_miss_only_flops=matched.miss_only_flops,
            gated_hit_path_flops=matched.hit_path_flops,
            gated_full_flops=matched.full_flops,
            gated_effective_flops=matched.effective_flops,
            gated_aux_hidden=matched.aux_hidden,
            gated_fusion_hidden=matched.fusion_hidden,
            gated_num_fusion_blocks=matched.num_fusion_blocks,
            cache_hit_rate_for_matching=args.cache_hit_rate_for_matching,
            param_rel_error=matched.param_rel_error,
            cache_weighted_flop_rel_error=matched.flop_rel_error,
        )
        wandb.log(
            {
                "arch/baseline_params": baseline_params,
                "arch/baseline_miss_only_flops": baseline_flops.miss_only,
                "arch/baseline_hit_path_flops": baseline_flops.hit_path,
                "arch/baseline_full_flops": baseline_flops.full,
                "arch/baseline_effective_flops": baseline_flops.effective,
                "arch/gated_params": matched.params,
                "arch/gated_miss_only_flops": matched.miss_only_flops,
                "arch/gated_hit_path_flops": matched.hit_path_flops,
                "arch/gated_full_flops": matched.full_flops,
                "arch/gated_effective_flops": matched.effective_flops,
                "arch/gated_aux_hidden": matched.aux_hidden,
                "arch/gated_fusion_hidden": matched.fusion_hidden,
                "arch/gated_num_fusion_blocks": matched.num_fusion_blocks,
                "arch/cache_hit_rate_for_matching": args.cache_hit_rate_for_matching,
                "arch/param_rel_error": matched.param_rel_error,
                "arch/cache_weighted_flop_rel_error": matched.flop_rel_error,
            }
        )

        baseline_result = train_offline_model(
            model_name="baseline_concat_fc",
            wandb_prefix="baseline",
            model=baseline_model,
            source=source,
            train_local_indices=train_local_indices,
            train_eval_local_indices=train_eval_local_indices,
            val_eval_local_indices=val_eval_local_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )

        gated_result = train_offline_model(
            model_name="gated_fusion",
            wandb_prefix="gated",
            model=gated_model,
            source=source,
            train_local_indices=train_local_indices,
            train_eval_local_indices=train_eval_local_indices,
            val_eval_local_indices=val_eval_local_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )

        winner = min(
            [baseline_result, gated_result],
            key=lambda x: x["final"]["val_total_loss"],
        )["model_name"]

        wandb.log(
            {
                "comparison/baseline_final_val_total_loss": baseline_result["final"][
                    "val_total_loss"
                ],
                "comparison/gated_final_val_total_loss": gated_result["final"][
                    "val_total_loss"
                ],
                "comparison/baseline_final_val_policy_loss": baseline_result["final"][
                    "val_policy_loss"
                ],
                "comparison/gated_final_val_policy_loss": gated_result["final"][
                    "val_policy_loss"
                ],
                "comparison/baseline_final_val_value_loss": baseline_result["final"][
                    "val_value_loss"
                ],
                "comparison/gated_final_val_value_loss": gated_result["final"][
                    "val_value_loss"
                ],
                "comparison/winner_is_gated": 1 if winner == "gated_fusion" else 0,
            }
        )

        run.summary["winner_by_final_val_total_loss"] = winner
        run.summary["baseline_final_val_total_loss"] = baseline_result["final"][
            "val_total_loss"
        ]
        run.summary["gated_final_val_total_loss"] = gated_result["final"][
            "val_total_loss"
        ]
        run.summary["matched_gated_aux_hidden"] = matched.aux_hidden
        run.summary["matched_gated_fusion_hidden"] = matched.fusion_hidden
        run.summary["matched_gated_num_fusion_blocks"] = matched.num_fusion_blocks
        run.summary["matched_param_rel_error"] = matched.param_rel_error
        run.summary["matched_cache_weighted_flop_rel_error"] = matched.flop_rel_error
        run.summary["cache_hit_rate_for_matching"] = args.cache_hit_rate_for_matching
        run.summary["baseline_effective_flops"] = baseline_flops.effective
        run.summary["gated_effective_flops"] = matched.effective_flops

        logger.info(
            "Offline architecture comparison complete",
            winner=winner,
            baseline_final_val_loss=baseline_result["final"]["val_total_loss"],
            gated_final_val_loss=gated_result["final"]["val_total_loss"],
        )
    finally:
        npz.close()
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
