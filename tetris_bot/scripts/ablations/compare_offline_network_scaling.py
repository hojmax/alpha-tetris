from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog
import torch
import wandb
from simple_parsing import parse

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_CONFIG_PATH,
    NUM_ACTIONS,
)
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.network import BOARD_STATS_FEATURES, PIECE_AUX_FEATURES, TetrisNet
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    FlopBreakdown,
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

    trunk_channels: int = 16
    num_conv_residual_blocks: int = 1
    reduction_channels: int = 32
    fc_hidden: int = 128
    aux_hidden: int = 24
    num_fusion_blocks: int = 0
    conv_kernel_size: int = 3
    conv_padding: int = 1
    max_placements: int = _DEFAULT_SELF_PLAY.max_placements

    board_trunk_multiplier: int = 2  # Multiply trunk_channels and reduction_channels
    post_fusion_multiplier: int = 2  # Multiply fc_hidden size
    cache_hit_rate_for_effective_flops: float = 0.96

    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "network-scaling"]
    )


@dataclass(frozen=True)
class ScalingVariant:
    name: str
    wandb_prefix: str
    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    fc_hidden: int


def validate_args(args: ScriptArgs) -> None:
    validate_common_offline_args(args)
    if args.trunk_channels <= 0:
        raise ValueError("trunk_channels must be > 0")
    if args.num_conv_residual_blocks < 0:
        raise ValueError("num_conv_residual_blocks must be >= 0")
    if args.reduction_channels <= 0:
        raise ValueError("reduction_channels must be > 0")
    if args.fc_hidden <= 0:
        raise ValueError("fc_hidden must be > 0")
    if args.aux_hidden <= 0:
        raise ValueError("aux_hidden must be > 0")
    if args.num_fusion_blocks < 0:
        raise ValueError("num_fusion_blocks must be >= 0")
    if args.conv_kernel_size <= 0:
        raise ValueError("conv_kernel_size must be > 0")
    if args.board_trunk_multiplier <= 0:
        raise ValueError("board_trunk_multiplier must be > 0")
    if args.post_fusion_multiplier <= 0:
        raise ValueError("post_fusion_multiplier must be > 0")
    if not 0.0 <= args.cache_hit_rate_for_effective_flops <= 1.0:
        raise ValueError("cache_hit_rate_for_effective_flops must be in [0, 1]")


def build_variants(args: ScriptArgs) -> list[ScalingVariant]:
    trunk_scaled = args.trunk_channels * args.board_trunk_multiplier
    reduction_scaled = args.reduction_channels * args.board_trunk_multiplier
    post_fusion_scaled_hidden = args.fc_hidden * args.post_fusion_multiplier

    variants = [
        ScalingVariant(
            name="default",
            wandb_prefix="default",
            trunk_channels=args.trunk_channels,
            num_conv_residual_blocks=args.num_conv_residual_blocks,
            reduction_channels=args.reduction_channels,
            fc_hidden=args.fc_hidden,
        ),
        ScalingVariant(
            name="double_board_trunk",
            wandb_prefix="double_board_trunk",
            trunk_channels=trunk_scaled,
            num_conv_residual_blocks=args.num_conv_residual_blocks,
            reduction_channels=reduction_scaled,
            fc_hidden=args.fc_hidden,
        ),
        ScalingVariant(
            name="double_post_fusion",
            wandb_prefix="double_post_fusion",
            trunk_channels=args.trunk_channels,
            num_conv_residual_blocks=args.num_conv_residual_blocks,
            reduction_channels=args.reduction_channels,
            fc_hidden=post_fusion_scaled_hidden,
        ),
    ]

    seen: set[tuple[int, int, int, int]] = set()
    for variant in variants:
        key = (
            variant.trunk_channels,
            variant.num_conv_residual_blocks,
            variant.reduction_channels,
            variant.fc_hidden,
        )
        if key in seen:
            raise ValueError(
                "Variant definitions are not unique; adjust multipliers "
                "so default, double_board_trunk, and double_post_fusion differ"
            )
        seen.add(key)

    return variants


def get_final_train_batches_per_sec(result: dict) -> float:
    final = result["final"]
    train_batches_per_sec = final.get("train_batches_per_sec")
    if train_batches_per_sec is not None:
        return float(train_batches_per_sec)

    wall_batches_per_sec = final.get("wall_batches_per_sec")
    if wall_batches_per_sec is not None:
        return float(wall_batches_per_sec)

    elapsed_sec = final.get("elapsed_sec", result["elapsed_sec"])
    if elapsed_sec <= 0:
        return 0.0
    return float(final["step"]) / float(elapsed_sec)


def stride2_output_hw(kernel_size: int, padding: int) -> tuple[int, int]:
    reduced_h = (BOARD_HEIGHT + 2 * padding - kernel_size) // 2 + 1
    reduced_w = (BOARD_WIDTH + 2 * padding - kernel_size) // 2 + 1
    if reduced_h <= 0 or reduced_w <= 0:
        raise ValueError(
            "Invalid stride-2 conv output shape; adjust conv_kernel_size/conv_padding "
            f"(got reduced_h={reduced_h}, reduced_w={reduced_w})"
        )
    return reduced_h, reduced_w


def variant_flop_breakdown(
    variant: ScalingVariant,
    args: ScriptArgs,
) -> FlopBreakdown:
    trunk = variant.trunk_channels
    reduction = variant.reduction_channels
    k = args.conv_kernel_size
    p = args.conv_padding
    reduced_h, reduced_w = stride2_output_hw(k, p)
    conv_flat = reduction * reduced_h * reduced_w
    fusion_hidden = variant.fc_hidden

    miss_only = 0
    # conv_initial: 1 -> trunk, same spatial
    miss_only += BOARD_HEIGHT * BOARD_WIDTH * trunk * (2 * 1 * k * k + 1)
    miss_only += 2 * BOARD_HEIGHT * BOARD_WIDTH * trunk  # BN
    # residual blocks: trunk -> trunk, same spatial
    for _ in range(variant.num_conv_residual_blocks):
        miss_only += BOARD_HEIGHT * BOARD_WIDTH * trunk * (2 * trunk * k * k + 1)
        miss_only += 2 * BOARD_HEIGHT * BOARD_WIDTH * trunk  # BN
        miss_only += BOARD_HEIGHT * BOARD_WIDTH * trunk * (2 * trunk * k * k + 1)
        miss_only += 2 * BOARD_HEIGHT * BOARD_WIDTH * trunk  # BN
    # conv_reduce: trunk -> reduction, stride-2
    miss_only += reduced_h * reduced_w * reduction * (2 * trunk * k * k + 1)
    miss_only += 2 * reduced_h * reduced_w * reduction  # BN
    # board_proj: conv_flat + board_stats -> fusion_hidden
    miss_only += fusion_hidden * (2 * (conv_flat + BOARD_STATS_FEATURES) + 1)

    hit_path = 0
    hit_path += args.aux_hidden * (2 * PIECE_AUX_FEATURES + 1)
    hit_path += 5 * args.aux_hidden
    hit_path += args.aux_hidden
    hit_path += fusion_hidden * (2 * args.aux_hidden + 1)
    hit_path += fusion_hidden * (2 * args.aux_hidden + 1)
    hit_path += 3 * fusion_hidden
    hit_path += 5 * fusion_hidden
    hit_path += fusion_hidden

    for _ in range(args.num_fusion_blocks):
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
    effective = hit_path + (1.0 - args.cache_hit_rate_for_effective_flops) * miss_only
    return FlopBreakdown(
        miss_only=miss_only,
        hit_path=hit_path,
        full=full,
        effective=effective,
    )


def normalize_args_for_wandb(
    args: ScriptArgs,
    variants: list[ScalingVariant],
) -> dict:
    normalized = asdict(args)
    normalized["data_path"] = str(args.data_path)
    normalized["variants"] = [
        {
            "name": variant.name,
            "wandb_prefix": variant.wandb_prefix,
            "trunk_channels": variant.trunk_channels,
            "num_conv_residual_blocks": variant.num_conv_residual_blocks,
            "reduction_channels": variant.reduction_channels,
            "fc_hidden": variant.fc_hidden,
        }
        for variant in variants
    ]
    return normalized


def main(args: ScriptArgs) -> None:
    validate_args(args)
    if not args.data_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.data_path}")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {args.data_path}")

    variants = build_variants(args)
    device_str = pick_device(args.device)
    device = torch.device(device_str)
    preload_mode = get_preload_mode(args)
    if preload_mode == "gpu" and device.type == "cpu":
        raise ValueError("preload_to_gpu requires a non-CPU device")
    logger.info("Using device", device=device_str, variants=[v.name for v in variants])

    run = init_wandb_run(args, normalize_args_for_wandb(args, variants))
    wandb.define_metric("offline_step")
    wandb.define_metric("variants/*", step_metric="offline_step")

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

        results: list[dict] = []
        for index, variant in enumerate(variants):
            torch.manual_seed(args.seed)
            model = TetrisNet(
                trunk_channels=variant.trunk_channels,
                num_conv_residual_blocks=variant.num_conv_residual_blocks,
                reduction_channels=variant.reduction_channels,
                fc_hidden=variant.fc_hidden,
                conv_kernel_size=args.conv_kernel_size,
                conv_padding=args.conv_padding,
                aux_hidden=args.aux_hidden,
                num_fusion_blocks=args.num_fusion_blocks,
            ).to(device)

            num_parameters = count_parameters(model)
            flops = variant_flop_breakdown(variant, args)
            wandb.log(
                {
                    f"variants/{variant.wandb_prefix}/num_parameters": num_parameters,
                    f"variants/{variant.wandb_prefix}/trunk_channels": (
                        variant.trunk_channels
                    ),
                    f"variants/{variant.wandb_prefix}/num_conv_residual_blocks": (
                        variant.num_conv_residual_blocks
                    ),
                    f"variants/{variant.wandb_prefix}/reduction_channels": (
                        variant.reduction_channels
                    ),
                    f"variants/{variant.wandb_prefix}/fc_hidden": variant.fc_hidden,
                    f"variants/{variant.wandb_prefix}/aux_hidden": args.aux_hidden,
                    f"variants/{variant.wandb_prefix}/num_fusion_blocks": (
                        args.num_fusion_blocks
                    ),
                    f"variants/{variant.wandb_prefix}/miss_only_flops": flops.miss_only,
                    f"variants/{variant.wandb_prefix}/hit_path_flops": flops.hit_path,
                    f"variants/{variant.wandb_prefix}/full_flops": flops.full,
                    f"variants/{variant.wandb_prefix}/effective_flops": flops.effective,
                    "arch/cache_hit_rate_for_effective_flops": (
                        args.cache_hit_rate_for_effective_flops
                    ),
                }
            )
            logger.info(
                "Training variant",
                variant=variant.name,
                variant_index=index,
                num_variants=len(variants),
                trunk_channels=variant.trunk_channels,
                reduction_channels=variant.reduction_channels,
                fc_hidden=variant.fc_hidden,
                params=num_parameters,
                effective_flops=flops.effective,
            )

            result = train_offline_model(
                model_name=variant.name,
                wandb_prefix=f"variants/{variant.wandb_prefix}",
                model=model,
                source=source,
                train_local_indices=train_local_indices,
                train_eval_local_indices=train_eval_local_indices,
                val_eval_local_indices=val_eval_local_indices,
                args=args,
                device=device,
                schedule_seed=args.seed + 12345,
            )
            result["variant"] = variant
            result["flops"] = flops
            results.append(result)

        winner = min(results, key=lambda x: x["final"]["val_total_loss"])
        default = next(
            result for result in results if result["variant"].wandb_prefix == "default"
        )

        comparison_log: dict[str, float | int | str] = {
            "comparison/winner": winner["variant"].name,
            "comparison/winner_is_default": (
                1 if winner["variant"].wandb_prefix == "default" else 0
            ),
        }

        for result in results:
            prefix = result["variant"].wandb_prefix
            final = result["final"]
            final_train_batches_per_sec = get_final_train_batches_per_sec(result)
            comparison_log[f"comparison/final_val_total_loss/{prefix}"] = final[
                "val_total_loss"
            ]
            comparison_log[f"comparison/final_val_policy_loss/{prefix}"] = final[
                "val_policy_loss"
            ]
            comparison_log[f"comparison/final_val_value_loss/{prefix}"] = final[
                "val_value_loss"
            ]
            comparison_log[f"comparison/final_train_total_loss/{prefix}"] = final[
                "train_total_loss"
            ]
            comparison_log[f"comparison/final_elapsed_sec/{prefix}"] = result[
                "elapsed_sec"
            ]
            comparison_log[f"comparison/final_train_batches_per_sec/{prefix}"] = (
                final_train_batches_per_sec
            )
            comparison_log[f"comparison/final_train_examples_per_sec/{prefix}"] = (
                final_train_batches_per_sec * args.batch_size
            )
            comparison_log[f"comparison/num_parameters/{prefix}"] = result[
                "num_parameters"
            ]
            comparison_log[f"comparison/effective_flops/{prefix}"] = result[
                "flops"
            ].effective

        default_final = default["final"]
        default_batches_per_sec = get_final_train_batches_per_sec(default)
        default_val_total_loss = default_final["val_total_loss"]

        for result in results:
            prefix = result["variant"].wandb_prefix
            if prefix == "default":
                continue
            final = result["final"]
            final_train_batches_per_sec = get_final_train_batches_per_sec(result)
            comparison_log[f"comparison/val_total_loss_delta_vs_default/{prefix}"] = (
                final["val_total_loss"] - default_val_total_loss
            )
            comparison_log[f"comparison/train_speedup_vs_default/{prefix}"] = (
                final_train_batches_per_sec / default_batches_per_sec
                if default_batches_per_sec > 0
                else 0.0
            )

        wandb.log(comparison_log)

        run.summary["winner_by_final_val_total_loss"] = winner["variant"].name
        run.summary["num_variants"] = len(results)
        run.summary["cache_hit_rate_for_effective_flops"] = (
            args.cache_hit_rate_for_effective_flops
        )

        for result in results:
            prefix = result["variant"].wandb_prefix
            run.summary[f"{prefix}_final_val_total_loss"] = result["final"][
                "val_total_loss"
            ]
            run.summary[f"{prefix}_final_train_batches_per_sec"] = (
                get_final_train_batches_per_sec(result)
            )
            run.summary[f"{prefix}_num_parameters"] = result["num_parameters"]
            run.summary[f"{prefix}_effective_flops"] = result["flops"].effective

        logger.info(
            "Offline network scaling comparison complete",
            winner=winner["variant"].name,
            default_final_val_total_loss=default["final"]["val_total_loss"],
            winner_final_val_total_loss=winner["final"]["val_total_loss"],
        )
    finally:
        npz.close()
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
