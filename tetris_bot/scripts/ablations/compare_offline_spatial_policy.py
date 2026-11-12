from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from simple_parsing import parse

from tetris_bot.action_space import (
    ACTION_TO_CANONICAL_FLAT_INDEX,
    ACTIVE_CANONICAL_ROTATION_COUNTS,
    FULL_GRID_PLACEMENT_SLOTS,
    NUM_PLACEMENT_ACTIONS,
    PIECE_VALID_ACTION_COUNTS,
    ROTATION_LABELS,
)
from tetris_bot.constants import (
    DEFAULT_CONFIG_PATH,
    NUM_ACTIONS,
    PIECE_NAMES,
)
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.network import (
    BOARD_STATS_FEATURES,
    PIECE_AUX_FEATURES,
    ResidualConvBlock,
    TetrisNet,
    _make_group_norm,
)
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    OfflineDataSource,
    count_parameters,
    get_preload_mode,
    init_wandb_run,
    pick_device,
    setup_offline_dataset,
    train_offline_model,
    validate_common_offline_args,
)
from tetris_bot.ml.replay_npz import open_replay_npz

logger = structlog.get_logger()
_DEFAULT_NETWORK = load_training_config(DEFAULT_CONFIG_PATH).network
PLACEMENT_GRID_SLOTS = FULL_GRID_PLACEMENT_SLOTS
ROTATION_ACTION_COUNTS = ACTIVE_CANONICAL_ROTATION_COUNTS


@dataclass
class ScriptArgs:
    data_path: Path
    device: str = "auto"
    seed: int = 123
    max_examples: int = 0
    train_fraction: float = 0.9
    steps: int = 20000
    batch_size: int = 1024
    eval_interval: int = 100
    eval_examples: int = 32_768
    eval_batch_size: int = 2048
    log_train_metrics_every: int = 1
    preload_to_gpu: bool = True
    preload_to_ram: bool = False
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    value_loss_weight: float = 1.0

    baseline_trunk_channels: int = _DEFAULT_NETWORK.trunk_channels
    baseline_num_conv_residual_blocks: int = _DEFAULT_NETWORK.num_conv_residual_blocks
    baseline_reduction_channels: int = _DEFAULT_NETWORK.reduction_channels
    baseline_fc_hidden: int = _DEFAULT_NETWORK.fc_hidden
    baseline_aux_hidden: int = _DEFAULT_NETWORK.aux_hidden
    baseline_num_fusion_blocks: int = _DEFAULT_NETWORK.num_fusion_blocks
    baseline_conv_kernel_size: int = _DEFAULT_NETWORK.conv_kernel_size
    baseline_conv_padding: int = _DEFAULT_NETWORK.conv_padding

    spatial_trunk_channels: int = _DEFAULT_NETWORK.trunk_channels
    spatial_num_conv_residual_blocks: int = _DEFAULT_NETWORK.num_conv_residual_blocks
    spatial_aux_hidden: int = _DEFAULT_NETWORK.aux_hidden
    spatial_decoder_channels: int = _DEFAULT_NETWORK.reduction_channels
    spatial_num_decoder_blocks: int = 1
    spatial_value_hidden: int = _DEFAULT_NETWORK.fc_hidden
    spatial_hold_hidden: int = _DEFAULT_NETWORK.aux_hidden
    spatial_conv_kernel_size: int = _DEFAULT_NETWORK.conv_kernel_size
    spatial_conv_padding: int = _DEFAULT_NETWORK.conv_padding

    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "spatial-policy-compare"]
    )


class SpatialPolicyDecoderTetrisNet(nn.Module):
    """Board-spatial policy decoder with a separate non-spatial hold head."""

    def __init__(
        self,
        trunk_channels: int,
        num_conv_residual_blocks: int,
        aux_hidden: int,
        decoder_channels: int,
        num_decoder_blocks: int,
        value_hidden: int,
        hold_hidden: int,
        conv_kernel_size: int,
        conv_padding: int,
    ):
        super().__init__()
        if trunk_channels <= 0:
            raise ValueError("trunk_channels must be > 0")
        if num_conv_residual_blocks < 0:
            raise ValueError("num_conv_residual_blocks must be >= 0")
        if aux_hidden <= 0:
            raise ValueError("aux_hidden must be > 0")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be > 0")
        if num_decoder_blocks < 0:
            raise ValueError("num_decoder_blocks must be >= 0")
        if value_hidden <= 0:
            raise ValueError("value_hidden must be > 0")
        if hold_hidden <= 0:
            raise ValueError("hold_hidden must be > 0")

        self.conv_initial = nn.Conv2d(
            1, trunk_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )
        self.bn_initial = _make_group_norm(trunk_channels)
        self.res_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    trunk_channels, kernel_size=conv_kernel_size, padding=conv_padding
                )
                for _ in range(num_conv_residual_blocks)
            ]
        )

        self.aux_fc = nn.Linear(PIECE_AUX_FEATURES, aux_hidden)
        self.aux_ln = nn.LayerNorm(aux_hidden)
        self.spatial_gate_fc = nn.Linear(aux_hidden, trunk_channels)
        self.spatial_bias_fc = nn.Linear(aux_hidden, trunk_channels)

        self.decoder_initial = nn.Conv2d(
            trunk_channels,
            decoder_channels,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.decoder_bn = _make_group_norm(decoder_channels)
        self.decoder_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    decoder_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                )
                for _ in range(num_decoder_blocks)
            ]
        )
        self.policy_head = nn.Conv2d(
            decoder_channels, len(ROTATION_LABELS), kernel_size=1
        )

        pooled_features = trunk_channels + aux_hidden + BOARD_STATS_FEATURES
        self.value_fc = nn.Linear(pooled_features, value_hidden)
        self.value_ln = nn.LayerNorm(value_hidden)
        self.value_head = nn.Linear(value_hidden, 1)
        self.hold_fc = nn.Linear(pooled_features, hold_hidden)
        self.hold_ln = nn.LayerNorm(hold_hidden)
        self.hold_head = nn.Linear(hold_hidden, 1)

        self.register_buffer(
            "active_flat_indices",
            torch.from_numpy(ACTION_TO_CANONICAL_FLAT_INDEX.copy()),
            persistent=False,
        )

    def forward(
        self,
        board: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board = board.float()
        piece_aux = aux_features[:, :PIECE_AUX_FEATURES]
        board_stats = aux_features[:, PIECE_AUX_FEATURES:]

        board_map = F.silu(self.bn_initial(self.conv_initial(board)))
        for block in self.res_blocks:
            board_map = block(board_map)

        aux_hidden = F.silu(self.aux_ln(self.aux_fc(piece_aux)))
        gate = (
            torch.sigmoid(self.spatial_gate_fc(aux_hidden)).unsqueeze(-1).unsqueeze(-1)
        )
        bias = self.spatial_bias_fc(aux_hidden).unsqueeze(-1).unsqueeze(-1)
        fused_spatial = board_map * (1.0 + gate) + bias

        policy_map = F.silu(self.decoder_bn(self.decoder_initial(fused_spatial)))
        for block in self.decoder_blocks:
            policy_map = block(policy_map)
        batch_size = board.shape[0]
        placement_logits = self.policy_head(policy_map).reshape(
            batch_size, PLACEMENT_GRID_SLOTS
        )
        action_logits = placement_logits[:, self.active_flat_indices]

        pooled_spatial = fused_spatial.mean(dim=(2, 3))
        pooled_features = torch.cat([pooled_spatial, aux_hidden, board_stats], dim=1)
        value_hidden = F.silu(self.value_ln(self.value_fc(pooled_features)))
        hold_hidden = F.silu(self.hold_ln(self.hold_fc(pooled_features)))
        value = self.value_head(value_hidden)
        hold_logit = self.hold_head(hold_hidden)

        policy_logits = torch.cat([action_logits, hold_logit], dim=1)
        if policy_logits.shape[1] != NUM_ACTIONS:
            raise ValueError(
                f"Spatial policy output width drifted: expected {NUM_ACTIONS}, "
                f"got {policy_logits.shape[1]}"
            )
        return policy_logits, value


def selection_metric(result: dict) -> float:
    final = result["final"]
    return float(final["val_policy_loss"]) + (float(final["val_value_loss"]) / 4.0)


def validate_args(args: ScriptArgs) -> None:
    validate_common_offline_args(args)
    if args.baseline_trunk_channels <= 0:
        raise ValueError("baseline_trunk_channels must be > 0")
    if args.baseline_num_conv_residual_blocks < 0:
        raise ValueError("baseline_num_conv_residual_blocks must be >= 0")
    if args.baseline_reduction_channels <= 0:
        raise ValueError("baseline_reduction_channels must be > 0")
    if args.baseline_fc_hidden <= 0:
        raise ValueError("baseline_fc_hidden must be > 0")
    if args.baseline_aux_hidden <= 0:
        raise ValueError("baseline_aux_hidden must be > 0")
    if args.baseline_num_fusion_blocks < 0:
        raise ValueError("baseline_num_fusion_blocks must be >= 0")
    if args.baseline_conv_kernel_size <= 0:
        raise ValueError("baseline_conv_kernel_size must be > 0")
    if args.spatial_trunk_channels <= 0:
        raise ValueError("spatial_trunk_channels must be > 0")
    if args.spatial_num_conv_residual_blocks < 0:
        raise ValueError("spatial_num_conv_residual_blocks must be >= 0")
    if args.spatial_aux_hidden <= 0:
        raise ValueError("spatial_aux_hidden must be > 0")
    if args.spatial_decoder_channels <= 0:
        raise ValueError("spatial_decoder_channels must be > 0")
    if args.spatial_num_decoder_blocks < 0:
        raise ValueError("spatial_num_decoder_blocks must be >= 0")
    if args.spatial_value_hidden <= 0:
        raise ValueError("spatial_value_hidden must be > 0")
    if args.spatial_hold_hidden <= 0:
        raise ValueError("spatial_hold_hidden must be > 0")
    if args.spatial_conv_kernel_size <= 0:
        raise ValueError("spatial_conv_kernel_size must be > 0")


def normalize_args_for_wandb(args: ScriptArgs) -> dict:
    normalized = asdict(args)
    normalized["data_path"] = str(args.data_path)
    normalized["placement_grid_slots"] = PLACEMENT_GRID_SLOTS
    normalized["num_placement_actions"] = NUM_PLACEMENT_ACTIONS
    return normalized


def log_mapping_metadata() -> dict[str, int]:
    metadata: dict[str, int] = {
        "mapping/num_placement_actions": NUM_PLACEMENT_ACTIONS,
        "mapping/placement_grid_slots": PLACEMENT_GRID_SLOTS,
        "mapping/hold_actions": 1,
    }
    for rotation_idx, count in enumerate(ROTATION_ACTION_COUNTS):
        metadata[f"mapping/rotation_{ROTATION_LABELS[rotation_idx]}_action_count"] = (
            count
        )
    for piece_idx, count in enumerate(PIECE_VALID_ACTION_COUNTS):
        metadata[f"mapping/piece_{PIECE_NAMES[piece_idx]}_valid_actions"] = count
    return metadata


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
    wandb.define_metric("current_gated/*", step_metric="offline_step")
    wandb.define_metric("spatial_policy/*", step_metric="offline_step")

    npz = open_replay_npz(args.data_path, mmap_mode="r")
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
        source: OfflineDataSource = setup.source

        torch.manual_seed(args.seed)
        baseline_model = TetrisNet(
            trunk_channels=args.baseline_trunk_channels,
            num_conv_residual_blocks=args.baseline_num_conv_residual_blocks,
            reduction_channels=args.baseline_reduction_channels,
            fc_hidden=args.baseline_fc_hidden,
            conv_kernel_size=args.baseline_conv_kernel_size,
            conv_padding=args.baseline_conv_padding,
            aux_hidden=args.baseline_aux_hidden,
            num_fusion_blocks=args.baseline_num_fusion_blocks,
        ).to(device)

        torch.manual_seed(args.seed)
        spatial_model = SpatialPolicyDecoderTetrisNet(
            trunk_channels=args.spatial_trunk_channels,
            num_conv_residual_blocks=args.spatial_num_conv_residual_blocks,
            aux_hidden=args.spatial_aux_hidden,
            decoder_channels=args.spatial_decoder_channels,
            num_decoder_blocks=args.spatial_num_decoder_blocks,
            value_hidden=args.spatial_value_hidden,
            hold_hidden=args.spatial_hold_hidden,
            conv_kernel_size=args.spatial_conv_kernel_size,
            conv_padding=args.spatial_conv_padding,
        ).to(device)

        baseline_params = count_parameters(baseline_model)
        spatial_params = count_parameters(spatial_model)
        mapping_metadata = log_mapping_metadata()
        wandb.log(
            {
                "arch/current_gated_params": baseline_params,
                "arch/spatial_policy_params": spatial_params,
                **mapping_metadata,
            }
        )
        logger.info(
            "Model setup",
            current_gated_params=baseline_params,
            spatial_policy_params=spatial_params,
            placement_grid_slots=PLACEMENT_GRID_SLOTS,
            placement_actions=NUM_PLACEMENT_ACTIONS,
            rotation_action_counts=ROTATION_ACTION_COUNTS,
            piece_valid_action_counts=PIECE_VALID_ACTION_COUNTS,
        )

        baseline_result = train_offline_model(
            model_name="current_gated_fusion",
            wandb_prefix="current_gated",
            model=baseline_model,
            source=source,
            train_local_indices=setup.train_local_indices,
            train_eval_local_indices=setup.train_eval_local_indices,
            val_eval_local_indices=setup.val_eval_local_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )
        spatial_result = train_offline_model(
            model_name="spatial_policy_decoder",
            wandb_prefix="spatial_policy",
            model=spatial_model,
            source=source,
            train_local_indices=setup.train_local_indices,
            train_eval_local_indices=setup.train_eval_local_indices,
            val_eval_local_indices=setup.val_eval_local_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )

        winner_by_total_loss = min(
            [baseline_result, spatial_result],
            key=lambda result: float(result["final"]["val_total_loss"]),
        )["model_name"]
        winner_by_selection_metric = min(
            [baseline_result, spatial_result],
            key=selection_metric,
        )["model_name"]

        comparison_log = {
            "comparison/current_gated_final_val_total_loss": baseline_result["final"][
                "val_total_loss"
            ],
            "comparison/spatial_policy_final_val_total_loss": spatial_result["final"][
                "val_total_loss"
            ],
            "comparison/current_gated_final_val_policy_loss": baseline_result["final"][
                "val_policy_loss"
            ],
            "comparison/spatial_policy_final_val_policy_loss": spatial_result["final"][
                "val_policy_loss"
            ],
            "comparison/current_gated_final_val_value_loss": baseline_result["final"][
                "val_value_loss"
            ],
            "comparison/spatial_policy_final_val_value_loss": spatial_result["final"][
                "val_value_loss"
            ],
            "comparison/current_gated_selection_metric": selection_metric(
                baseline_result
            ),
            "comparison/spatial_policy_selection_metric": selection_metric(
                spatial_result
            ),
            "comparison/winner_is_spatial_by_val_total_loss": (
                1 if winner_by_total_loss == "spatial_policy_decoder" else 0
            ),
            "comparison/winner_is_spatial_by_selection_metric": (
                1 if winner_by_selection_metric == "spatial_policy_decoder" else 0
            ),
        }
        wandb.log(comparison_log)

        run.summary["winner_by_final_val_total_loss"] = winner_by_total_loss
        run.summary["winner_by_selection_metric"] = winner_by_selection_metric
        run.summary["current_gated_final_val_total_loss"] = baseline_result["final"][
            "val_total_loss"
        ]
        run.summary["spatial_policy_final_val_total_loss"] = spatial_result["final"][
            "val_total_loss"
        ]
        run.summary["current_gated_selection_metric"] = selection_metric(
            baseline_result
        )
        run.summary["spatial_policy_selection_metric"] = selection_metric(
            spatial_result
        )
        run.summary["current_gated_params"] = baseline_params
        run.summary["spatial_policy_params"] = spatial_params

        logger.info(
            "Offline spatial-policy comparison complete",
            winner_by_total_loss=winner_by_total_loss,
            winner_by_selection_metric=winner_by_selection_metric,
            current_gated_val_total_loss=baseline_result["final"]["val_total_loss"],
            spatial_policy_val_total_loss=spatial_result["final"]["val_total_loss"],
            current_gated_selection_metric=selection_metric(baseline_result),
            spatial_policy_selection_metric=selection_metric(spatial_result),
        )
    finally:
        npz.close()
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
