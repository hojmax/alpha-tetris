from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from simple_parsing import parse

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
)
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.network import BOARD_STATS_FEATURES, PIECE_AUX_FEATURES
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    GatedFusionTetrisNet,
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
_DEFAULT_SELF_PLAY = load_training_config(DEFAULT_CONFIG_PATH).self_play


@dataclass
class ScriptArgs:
    data_path: Path = (
        PROJECT_ROOT / "training_runs/v37/training_data.npz"
    )  # Path to offline replay buffer NPZ
    device: str = "auto"  # auto/cpu/cuda/mps
    seed: int = 123
    max_examples: int = 0  # 0 = use all examples in NPZ
    train_fraction: float = 0.9
    steps: int = 20000
    batch_size: int = 1024
    eval_interval: int = 10000
    eval_examples: int = 32_768  # Max examples to use per train/val eval pass
    eval_batch_size: int = 2048
    log_train_metrics_every: int = 1  # Batch metric logging cadence
    preload_to_gpu: bool = True  # Preload selected dataset tensors to GPU
    preload_to_ram: bool = False  # Preload selected dataset tensors to CPU RAM
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    value_loss_weight: float = 1.0

    # Deep conv architecture
    trunk_channels: int = 16
    num_conv_residual_blocks: int = 1
    reduction_channels: int = 32

    # Current architecture (control)
    current_conv_filters: list[int] = field(default_factory=lambda: [4, 8])

    # Shared architecture params
    fc_hidden: int = 128
    aux_hidden: int = 24
    conv_kernel_size: int = 3
    conv_padding: int = 1
    max_placements: int = _DEFAULT_SELF_PLAY.max_placements

    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "conv-depth-compare"]
    )


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        return x + h


class DeepConvGatedFusionTetrisNet(nn.Module):
    def __init__(
        self,
        trunk_channels: int,
        num_conv_residual_blocks: int,
        reduction_channels: int,
        fc_hidden: int,
        aux_hidden: int,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
    ):
        super().__init__()
        # Initial conv: 1 -> trunk_channels, same spatial dims
        self.conv_initial = nn.Conv2d(
            1, trunk_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )
        self.bn_initial = nn.BatchNorm2d(trunk_channels)

        # Residual conv blocks at trunk_channels
        self.res_blocks = nn.ModuleList(
            [
                ResidualConvBlock(trunk_channels, conv_kernel_size, conv_padding)
                for _ in range(num_conv_residual_blocks)
            ]
        )

        # Stride-2 reduction: trunk_channels -> reduction_channels, halves spatial dims
        self.conv_reduce = nn.Conv2d(
            trunk_channels,
            reduction_channels,
            kernel_size=conv_kernel_size,
            stride=2,
            padding=conv_padding,
        )
        self.bn_reduce = nn.BatchNorm2d(reduction_channels)

        # Compute flattened conv output size: stride-2 on 20x10 -> 10x5
        reduced_h = (BOARD_HEIGHT + 2 * conv_padding - conv_kernel_size) // 2 + 1
        reduced_w = (BOARD_WIDTH + 2 * conv_padding - conv_kernel_size) // 2 + 1
        conv_flat_size = reduction_channels * reduced_h * reduced_w

        fusion_hidden = fc_hidden

        # Board projection: conv features + board stats -> fusion_hidden
        self.board_proj = nn.Linear(
            conv_flat_size + BOARD_STATS_FEATURES, fusion_hidden
        )

        # Aux-conditioned modulation (same as current architecture)
        self.aux_fc = nn.Linear(PIECE_AUX_FEATURES, aux_hidden)
        self.aux_ln = nn.LayerNorm(aux_hidden)
        self.gate_fc = nn.Linear(aux_hidden, fusion_hidden)
        self.aux_proj = nn.Linear(aux_hidden, fusion_hidden)

        # Post-fusion
        self.fusion_ln = nn.LayerNorm(fusion_hidden)

        self.policy_head = nn.Linear(fusion_hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(fusion_hidden, 1)

    def forward(
        self,
        board: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        piece_aux = aux_features[:, :PIECE_AUX_FEATURES]
        board_stats = aux_features[:, PIECE_AUX_FEATURES:]

        # Deep conv trunk
        x = F.relu(self.bn_initial(self.conv_initial(board)))
        for block in self.res_blocks:
            x = block(x)
        x = F.relu(self.bn_reduce(self.conv_reduce(x)))
        x = x.view(x.size(0), -1)

        # Board projection (cached path in inference)
        board_h = self.board_proj(torch.cat([x, board_stats], dim=1))

        # Gated fusion (identical to current architecture)
        aux_h = F.relu(self.aux_ln(self.aux_fc(piece_aux)))
        gate = torch.sigmoid(self.gate_fc(aux_h))

        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))

        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)
        return policy_logits, value


def count_conv_parameters(model: nn.Module) -> int:
    conv_param_names = {
        "conv_initial",
        "bn_initial",
        "res_blocks",
        "conv_reduce",
        "bn_reduce",
    }
    total = 0
    for name, param in model.named_parameters():
        top_name = name.split(".")[0]
        if top_name in conv_param_names:
            total += param.numel()
    return total


def validate_args(args: ScriptArgs) -> None:
    validate_common_offline_args(args)
    if len(args.current_conv_filters) != 2:
        raise ValueError("current_conv_filters must contain exactly two values")
    if args.trunk_channels <= 0:
        raise ValueError("trunk_channels must be > 0")
    if args.num_conv_residual_blocks < 0:
        raise ValueError("num_conv_residual_blocks must be >= 0")
    if args.reduction_channels <= 0:
        raise ValueError("reduction_channels must be > 0")


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
    wandb.define_metric("current/*", step_metric="offline_step")
    wandb.define_metric("deep_conv/*", step_metric="offline_step")

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
        source = setup.source
        train_local_indices = setup.train_local_indices
        train_eval_local_indices = setup.train_eval_local_indices
        val_eval_local_indices = setup.val_eval_local_indices

        # Build current production architecture (gated fusion, Conv(1->4->8), 0 fusion blocks)
        # train_offline_model expects an args object with certain fields. We create a
        # lightweight adapter since train_offline_model reads from ScriptArgs of the
        # compare_offline_architectures module. We pass our own args which has the same
        # field names for the fields used by train_offline_model.
        torch.manual_seed(args.seed)
        current_model = GatedFusionTetrisNet(
            conv_filters=args.current_conv_filters,
            conv_kernel_size=args.conv_kernel_size,
            conv_padding=args.conv_padding,
            aux_hidden=args.aux_hidden,
            fusion_hidden=args.fc_hidden,
            num_fusion_blocks=0,
        ).to(device)
        current_params = count_parameters(current_model)

        # Build deep conv variant
        torch.manual_seed(args.seed)
        deep_conv_model = DeepConvGatedFusionTetrisNet(
            trunk_channels=args.trunk_channels,
            num_conv_residual_blocks=args.num_conv_residual_blocks,
            reduction_channels=args.reduction_channels,
            fc_hidden=args.fc_hidden,
            aux_hidden=args.aux_hidden,
            conv_kernel_size=args.conv_kernel_size,
            conv_padding=args.conv_padding,
        ).to(device)
        deep_conv_params = count_parameters(deep_conv_model)
        deep_conv_conv_params = count_conv_parameters(deep_conv_model)

        # Verify conv output dimensions match
        with torch.no_grad():
            dummy_board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH, device=device)

            # Current conv path output
            x_curr = F.relu(current_model.bn1(current_model.conv1(dummy_board)))
            x_curr = F.relu(current_model.bn2(current_model.conv2(x_curr)))
            current_conv_flat = x_curr.view(1, -1).shape[1]

            # Deep conv path output
            x_deep = F.relu(
                deep_conv_model.bn_initial(deep_conv_model.conv_initial(dummy_board))
            )
            for block in deep_conv_model.res_blocks:
                x_deep = block(x_deep)
            x_deep = F.relu(
                deep_conv_model.bn_reduce(deep_conv_model.conv_reduce(x_deep))
            )
            deep_conv_flat = x_deep.view(1, -1).shape[1]

        # Count conv params for current model
        current_conv_param_names = {"conv1", "bn1", "conv2", "bn2"}
        current_conv_params = sum(
            p.numel()
            for name, p in current_model.named_parameters()
            if name.split(".")[0] in current_conv_param_names
        )

        logger.info(
            "Architecture comparison",
            current_params=current_params,
            current_conv_params=current_conv_params,
            current_conv_flat_size=current_conv_flat,
            deep_conv_params=deep_conv_params,
            deep_conv_conv_params=deep_conv_conv_params,
            deep_conv_flat_size=deep_conv_flat,
            conv_flat_match=current_conv_flat == deep_conv_flat,
            param_difference=deep_conv_params - current_params,
            param_ratio=deep_conv_params / current_params,
        )
        wandb.log(
            {
                "arch/current_params": current_params,
                "arch/current_conv_params": current_conv_params,
                "arch/current_conv_flat_size": current_conv_flat,
                "arch/deep_conv_params": deep_conv_params,
                "arch/deep_conv_conv_params": deep_conv_conv_params,
                "arch/deep_conv_flat_size": deep_conv_flat,
                "arch/conv_flat_match": 1 if current_conv_flat == deep_conv_flat else 0,
                "arch/param_difference": deep_conv_params - current_params,
                "arch/param_ratio": deep_conv_params / current_params,
                "arch/trunk_channels": args.trunk_channels,
                "arch/num_conv_residual_blocks": args.num_conv_residual_blocks,
                "arch/reduction_channels": args.reduction_channels,
            }
        )

        current_result = train_offline_model(
            model_name="current_gated_fusion",
            wandb_prefix="current",
            model=current_model,
            source=source,
            train_local_indices=train_local_indices,
            train_eval_local_indices=train_eval_local_indices,
            val_eval_local_indices=val_eval_local_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )

        deep_conv_result = train_offline_model(
            model_name="deep_conv_gated_fusion",
            wandb_prefix="deep_conv",
            model=deep_conv_model,
            source=source,
            train_local_indices=train_local_indices,
            train_eval_local_indices=train_eval_local_indices,
            val_eval_local_indices=val_eval_local_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )

        winner = min(
            [current_result, deep_conv_result],
            key=lambda x: x["final"]["val_total_loss"],
        )["model_name"]

        wandb.log(
            {
                "comparison/current_final_val_total_loss": current_result["final"][
                    "val_total_loss"
                ],
                "comparison/deep_conv_final_val_total_loss": deep_conv_result["final"][
                    "val_total_loss"
                ],
                "comparison/current_final_val_policy_loss": current_result["final"][
                    "val_policy_loss"
                ],
                "comparison/deep_conv_final_val_policy_loss": deep_conv_result["final"][
                    "val_policy_loss"
                ],
                "comparison/current_final_val_value_loss": current_result["final"][
                    "val_value_loss"
                ],
                "comparison/deep_conv_final_val_value_loss": deep_conv_result["final"][
                    "val_value_loss"
                ],
                "comparison/winner_is_deep_conv": (
                    1 if winner == "deep_conv_gated_fusion" else 0
                ),
            }
        )

        run.summary["winner_by_final_val_total_loss"] = winner
        run.summary["current_final_val_total_loss"] = current_result["final"][
            "val_total_loss"
        ]
        run.summary["deep_conv_final_val_total_loss"] = deep_conv_result["final"][
            "val_total_loss"
        ]
        run.summary["current_params"] = current_params
        run.summary["deep_conv_params"] = deep_conv_params
        run.summary["deep_conv_conv_params"] = deep_conv_conv_params
        run.summary["conv_flat_match"] = current_conv_flat == deep_conv_flat

        logger.info(
            "Offline conv depth comparison complete",
            winner=winner,
            current_final_val_loss=current_result["final"]["val_total_loss"],
            deep_conv_final_val_loss=deep_conv_result["final"]["val_total_loss"],
        )
    finally:
        npz.close()
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
