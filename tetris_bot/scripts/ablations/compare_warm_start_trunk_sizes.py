from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import cast

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import MCTSConfig, evaluate_model
from tetris_bot.constants import BENCHMARKS_DIR, CONFIG_FILENAME, TRAINING_DATA_FILENAME
from tetris_bot.ml.config import (
    NetworkConfig,
    TrainingConfig,
    load_training_config,
)
from tetris_bot.ml.network import TetrisNet
from tetris_bot.scripts.warm_start import (
    ScriptArgs as WarmStartArgs,
    build_output_config,
    resolve_eval_num_workers,
    run_warm_start,
)

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    source_run_dir: Path
    output_root: Path | None = None
    trunk_channels: list[int] = field(default_factory=list)
    trunk_multipliers: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    residual_block_counts: list[int] = field(default_factory=list)
    scale_residual_blocks_with_size: bool = True
    scale_reduction_channels_with_trunk: bool = True
    resume_from_source_offline_state: bool = False
    device: str = "auto"
    seed: int = 123
    epochs_per_round: float = 2.0
    early_stopping_patience: int = 20
    max_rounds: int = 0
    max_examples: int = 0
    batch_size: int | None = None
    learning_rate: float | None = None
    warmup_epochs: float = 3.0
    lr_min_factor: float = 0.1
    weight_decay: float | None = None
    grad_clip_norm: float | None = None
    eval_examples: int = 32_768
    eval_batch_size: int = 2_048
    preload_to_gpu: bool = True
    preload_to_ram: bool = False
    num_eval_games: int = 20
    eval_seed_start: int = 0
    eval_num_workers: int = 0
    eval_num_simulations: int | None = None
    eval_max_placements: int | None = None
    mcts_seed: int | None = 0
    warm_start_wandb_project: str = "tetris-mcts-offline"
    warm_start_wandb_entity: str | None = None
    warm_start_wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "warm-start", "trunk-size-compare"]
    )
    benchmark_num_games: int = 20
    benchmark_repeats: int = 3
    benchmark_seed_start: int = 1000
    benchmark_num_workers: int = 0
    benchmark_num_simulations: int | None = None
    benchmark_max_placements: int | None = None
    benchmark_mcts_seed: int | None = 0


@dataclass(frozen=True)
class TrunkVariant:
    label: str
    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    matches_source: bool


@dataclass(frozen=True)
class WarmStartVariantRecord:
    label: str
    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    matches_source: bool
    num_parameters: int
    output_run_dir: str
    summary_path: str
    incumbent_onnx_path: str
    warm_start_avg_attack: float
    warm_start_avg_moves: float
    warm_start_num_steps: int
    warm_start_offline_best: dict[str, object]


@dataclass(frozen=True)
class BenchmarkMetrics:
    elapsed_sec: float
    num_games: int
    avg_attack: float
    avg_lines: float
    avg_moves: float
    max_attack: int
    attack_per_piece: float
    avg_tree_nodes: float
    total_moves: int
    total_attack: int
    games_per_second: float
    moves_per_second: float
    game_results: list[dict[str, int]]


@dataclass(frozen=True)
class BenchmarkRow:
    variant: str
    repeat_index: int
    seed_start: int
    seed_end: int
    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    num_parameters: int
    elapsed_sec: float
    num_games: int
    avg_attack: float
    avg_lines: float
    avg_moves: float
    max_attack: int
    attack_per_piece: float
    avg_tree_nodes: float
    total_moves: int
    total_attack: int
    games_per_second: float
    moves_per_second: float
    game_results: list[dict[str, int]]


@dataclass
class BenchmarkAggregate:
    label: str
    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    num_parameters: int
    matches_source: bool
    output_run_dir: str
    incumbent_onnx_path: str
    warm_start_avg_attack: float
    warm_start_avg_moves: float
    warm_start_num_steps: int
    games_per_second_mean: float
    games_per_second_std: float
    moves_per_second_mean: float
    moves_per_second_std: float
    benchmark_avg_attack_mean: float
    benchmark_avg_attack_std: float
    benchmark_avg_moves_mean: float
    benchmark_avg_moves_std: float
    elapsed_sec_mean: float
    elapsed_sec_std: float
    speedup_vs_baseline: float = 0.0
    attack_delta_vs_baseline: float = 0.0
    pareto_optimal: bool = False


def validate_args(args: ScriptArgs) -> None:
    if not args.source_run_dir.exists():
        raise FileNotFoundError(
            f"Source run directory does not exist: {args.source_run_dir}"
        )
    if not args.source_run_dir.is_dir():
        raise NotADirectoryError(
            f"Source run directory is not a directory: {args.source_run_dir}"
        )
    config_path = args.source_run_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Source config not found: {config_path}")
    training_data_path = args.source_run_dir / TRAINING_DATA_FILENAME
    if not training_data_path.exists():
        raise FileNotFoundError(f"Source training data not found: {training_data_path}")
    if args.output_root is not None and args.output_root.exists():
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    if args.epochs_per_round <= 0:
        raise ValueError(f"epochs_per_round must be > 0 (got {args.epochs_per_round})")
    if args.early_stopping_patience <= 0:
        raise ValueError(
            f"early_stopping_patience must be > 0 (got {args.early_stopping_patience})"
        )
    if args.max_rounds < 0:
        raise ValueError(f"max_rounds must be >= 0 (got {args.max_rounds})")
    if args.max_examples < 0:
        raise ValueError(f"max_examples must be >= 0 (got {args.max_examples})")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 (got {args.batch_size})")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be > 0 when set (got {args.learning_rate})"
        )
    if args.weight_decay is not None and args.weight_decay < 0.0:
        raise ValueError(
            f"weight_decay must be >= 0 when set (got {args.weight_decay})"
        )
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0.0:
        raise ValueError(
            f"grad_clip_norm must be > 0 when set (got {args.grad_clip_norm})"
        )
    if args.eval_examples <= 0:
        raise ValueError(f"eval_examples must be > 0 (got {args.eval_examples})")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be > 0 (got {args.eval_batch_size})")
    if args.num_eval_games <= 0:
        raise ValueError(f"num_eval_games must be > 0 (got {args.num_eval_games})")
    if args.eval_num_workers < 0 or args.eval_num_workers == 1:
        raise ValueError(
            f"eval_num_workers must be 0 (auto) or >= 2 (got {args.eval_num_workers})"
        )
    if args.eval_num_simulations is not None and args.eval_num_simulations <= 0:
        raise ValueError(
            "eval_num_simulations must be > 0 when set "
            f"(got {args.eval_num_simulations})"
        )
    if args.eval_max_placements is not None and args.eval_max_placements <= 0:
        raise ValueError(
            f"eval_max_placements must be > 0 when set (got {args.eval_max_placements})"
        )
    if args.benchmark_num_games <= 0:
        raise ValueError(
            f"benchmark_num_games must be > 0 (got {args.benchmark_num_games})"
        )
    if args.benchmark_repeats <= 0:
        raise ValueError(
            f"benchmark_repeats must be > 0 (got {args.benchmark_repeats})"
        )
    if args.benchmark_num_workers < 0 or args.benchmark_num_workers == 1:
        raise ValueError(
            "benchmark_num_workers must be 0 (auto) or >= 2 "
            f"(got {args.benchmark_num_workers})"
        )
    if (
        args.benchmark_num_simulations is not None
        and args.benchmark_num_simulations <= 0
    ):
        raise ValueError(
            "benchmark_num_simulations must be > 0 when set "
            f"(got {args.benchmark_num_simulations})"
        )
    if args.benchmark_max_placements is not None and args.benchmark_max_placements <= 0:
        raise ValueError(
            "benchmark_max_placements must be > 0 when set "
            f"(got {args.benchmark_max_placements})"
        )
    if any(value <= 0 for value in args.trunk_channels):
        raise ValueError(f"trunk_channels must all be > 0 (got {args.trunk_channels})")
    if any(value < 0 for value in args.residual_block_counts):
        raise ValueError(
            f"residual_block_counts must all be >= 0 (got {args.residual_block_counts})"
        )
    if not args.trunk_channels:
        if not args.trunk_multipliers:
            raise ValueError(
                "Provide trunk_channels or trunk_multipliers to define variants"
            )
        if any(multiplier <= 0.0 for multiplier in args.trunk_multipliers):
            raise ValueError(
                f"trunk_multipliers must all be > 0 (got {args.trunk_multipliers})"
            )


def default_output_root(source_run_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        BENCHMARKS_DIR
        / "warm_start_trunk_sizes"
        / (f"{source_run_dir.name}_{timestamp}")
    )


def dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def resolve_trunk_channels(
    source_network: NetworkConfig,
    args: ScriptArgs,
) -> list[int]:
    if args.trunk_channels:
        variants = dedupe_preserve_order(list(args.trunk_channels))
    else:
        scaled = [
            max(1, int(round(source_network.trunk_channels * multiplier)))
            for multiplier in args.trunk_multipliers
        ]
        variants = dedupe_preserve_order(scaled)
    if len(variants) < 2:
        raise ValueError(
            "Need at least two unique trunk sizes to compare "
            f"(got {variants}; source trunk={source_network.trunk_channels})"
        )
    return variants


def reduction_channels_for_variant(
    source_network: NetworkConfig,
    *,
    trunk_channels: int,
    scale_reduction_channels_with_trunk: bool,
) -> int:
    if not scale_reduction_channels_with_trunk:
        return source_network.reduction_channels
    ratio = source_network.reduction_channels / source_network.trunk_channels
    return max(1, int(round(ratio * trunk_channels)))


def resolve_residual_block_counts(
    source_network: NetworkConfig,
    *,
    trunk_channels: list[int],
    args: ScriptArgs,
) -> list[int]:
    if args.residual_block_counts:
        if len(args.residual_block_counts) != len(trunk_channels):
            raise ValueError(
                "residual_block_counts must match the number of trunk variants "
                f"(got {len(args.residual_block_counts)} counts for {len(trunk_channels)} variants)"
            )
        return list(args.residual_block_counts)

    if not args.scale_residual_blocks_with_size:
        return [source_network.num_conv_residual_blocks] * len(trunk_channels)

    reference_channels = sorted({*trunk_channels, source_network.trunk_channels})
    source_index = reference_channels.index(source_network.trunk_channels)
    blocks_by_channel = {
        channel: max(0, source_network.num_conv_residual_blocks + index - source_index)
        for index, channel in enumerate(reference_channels)
    }
    return [blocks_by_channel[channel] for channel in trunk_channels]


def build_variants(
    source_config: TrainingConfig,
    args: ScriptArgs,
) -> list[TrunkVariant]:
    source_network = source_config.network
    resolved_trunk_channels = resolve_trunk_channels(source_network, args)
    resolved_residual_blocks = resolve_residual_block_counts(
        source_network,
        trunk_channels=resolved_trunk_channels,
        args=args,
    )
    variants: list[TrunkVariant] = []
    for trunk_channels, num_conv_residual_blocks in zip(
        resolved_trunk_channels, resolved_residual_blocks, strict=True
    ):
        reduction_channels = reduction_channels_for_variant(
            source_network,
            trunk_channels=trunk_channels,
            scale_reduction_channels_with_trunk=(
                args.scale_reduction_channels_with_trunk
            ),
        )
        variants.append(
            TrunkVariant(
                label=(
                    f"trunk{trunk_channels}_blocks{num_conv_residual_blocks}"
                    f"_reduce{reduction_channels}"
                ),
                trunk_channels=trunk_channels,
                num_conv_residual_blocks=num_conv_residual_blocks,
                reduction_channels=reduction_channels,
                matches_source=(
                    trunk_channels == source_network.trunk_channels
                    and (
                        num_conv_residual_blocks
                        == source_network.num_conv_residual_blocks
                    )
                    and reduction_channels == source_network.reduction_channels
                ),
            )
        )
    return variants


def build_variant_network_config(
    source_network: NetworkConfig,
    variant: TrunkVariant,
) -> NetworkConfig:
    network = copy.deepcopy(source_network)
    network.trunk_channels = variant.trunk_channels
    network.num_conv_residual_blocks = variant.num_conv_residual_blocks
    network.reduction_channels = variant.reduction_channels
    return network


def count_parameters(network: NetworkConfig) -> int:
    model = TetrisNet(**network.model_dump())
    return sum(parameter.numel() for parameter in model.parameters())


def make_warm_start_args(
    args: ScriptArgs,
    *,
    output_run_dir: Path,
    variant: TrunkVariant,
) -> WarmStartArgs:
    return WarmStartArgs(
        source_run_dir=args.source_run_dir,
        output_run_dir=output_run_dir,
        resume_from_source_offline_state=args.resume_from_source_offline_state,
        device=args.device,
        seed=args.seed,
        epochs_per_round=args.epochs_per_round,
        early_stopping_patience=args.early_stopping_patience,
        max_rounds=args.max_rounds,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        lr_min_factor=args.lr_min_factor,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        eval_examples=args.eval_examples,
        eval_batch_size=args.eval_batch_size,
        preload_to_gpu=args.preload_to_gpu,
        preload_to_ram=args.preload_to_ram,
        num_eval_games=args.num_eval_games,
        eval_seed_start=args.eval_seed_start,
        eval_num_workers=args.eval_num_workers,
        eval_num_simulations=args.eval_num_simulations,
        eval_max_placements=args.eval_max_placements,
        mcts_seed=args.mcts_seed,
        wandb_project=args.warm_start_wandb_project,
        wandb_run_name=f"{args.source_run_dir.name}-{variant.label}",
        wandb_entity=args.warm_start_wandb_entity,
        wandb_tags=[*args.warm_start_wandb_tags, variant.label],
    )


def build_benchmark_config(
    source_config: TrainingConfig,
    args: ScriptArgs,
) -> tuple[MCTSConfig, int, str, int, int, int | None]:
    num_workers_resolution = resolve_eval_num_workers(
        args.benchmark_num_workers,
        default_workers=source_config.self_play.num_workers,
    )
    num_simulations = (
        args.benchmark_num_simulations
        if args.benchmark_num_simulations is not None
        else source_config.self_play.num_simulations
    )
    max_placements = (
        args.benchmark_max_placements
        if args.benchmark_max_placements is not None
        else source_config.self_play.max_placements
    )
    mcts_seed = (
        args.benchmark_mcts_seed
        if args.benchmark_mcts_seed is not None
        else source_config.self_play.mcts_seed
    )
    config = MCTSConfig()
    config.num_simulations = num_simulations
    config.c_puct = source_config.self_play.c_puct
    config.temperature = source_config.self_play.temperature
    config.dirichlet_alpha = source_config.self_play.dirichlet_alpha
    config.dirichlet_epsilon = source_config.self_play.dirichlet_epsilon
    config.visit_sampling_epsilon = 0.0
    config.max_placements = max_placements
    config.reuse_tree = source_config.self_play.reuse_tree
    config.use_parent_value_for_unvisited_q = (
        source_config.self_play.use_parent_value_for_unvisited_q
    )
    config.fpu_reduction = source_config.self_play.fpu_reduction
    config.nn_value_weight = 1.0
    config.death_penalty = 0.0
    config.overhang_penalty_weight = 0.0
    config.seed = mcts_seed
    return (
        config,
        num_workers_resolution.num_workers,
        num_workers_resolution.source,
        num_simulations,
        max_placements,
        mcts_seed,
    )


def benchmark_once(
    *,
    model_path: Path,
    seeds: list[int],
    config: MCTSConfig,
    max_placements: int,
    num_workers: int,
) -> BenchmarkMetrics:
    start = time.perf_counter()
    result = evaluate_model(
        model_path=str(model_path),
        seeds=seeds,
        config=config,
        max_placements=max_placements,
        num_workers=num_workers,
        add_noise=False,
    )
    elapsed_sec = time.perf_counter() - start
    game_results = [(int(attack), int(moves)) for attack, moves in result.game_results]
    total_moves = sum(moves for _, moves in game_results)
    total_attack = sum(attack for attack, _ in game_results)
    return BenchmarkMetrics(
        elapsed_sec=elapsed_sec,
        num_games=int(result.num_games),
        avg_attack=float(result.avg_attack),
        avg_lines=float(result.avg_lines),
        avg_moves=float(result.avg_moves),
        max_attack=int(result.max_attack),
        attack_per_piece=float(result.attack_per_piece),
        avg_tree_nodes=float(result.avg_tree_nodes),
        total_moves=total_moves,
        total_attack=total_attack,
        games_per_second=(
            float(result.num_games) / elapsed_sec if elapsed_sec > 0 else 0
        ),
        moves_per_second=(total_moves / elapsed_sec if elapsed_sec > 0 else 0),
        game_results=[
            {"attack": attack, "moves": moves} for attack, moves in game_results
        ],
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_metric(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    if len(values) == 1:
        return values[0], 0.0
    return fmean(values), pstdev(values)


def pareto_optimal_labels(aggregates: list[BenchmarkAggregate]) -> set[str]:
    optimal: set[str] = set()
    for candidate in aggregates:
        dominated = False
        for other in aggregates:
            if other is candidate:
                continue
            if (
                other.benchmark_avg_attack_mean >= candidate.benchmark_avg_attack_mean
                and other.games_per_second_mean >= candidate.games_per_second_mean
                and (
                    other.benchmark_avg_attack_mean
                    > candidate.benchmark_avg_attack_mean
                    or other.games_per_second_mean > candidate.games_per_second_mean
                )
            ):
                dominated = True
                break
        if not dominated:
            optimal.add(candidate.label)
    return optimal


def build_markdown_table(aggregates: list[BenchmarkAggregate]) -> str:
    header = (
        "| variant | trunk | blocks | reduce | params | warm-start attack | "
        "bench attack | games/s | moves/s | speedup | attack delta | pareto |"
    )
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"
    rows = [header, separator]
    for aggregate in aggregates:
        rows.append(
            "| "
            f"{aggregate.label} | "
            f"{aggregate.trunk_channels} | "
            f"{aggregate.num_conv_residual_blocks} | "
            f"{aggregate.reduction_channels} | "
            f"{aggregate.num_parameters} | "
            f"{aggregate.warm_start_avg_attack:.2f} | "
            f"{aggregate.benchmark_avg_attack_mean:.2f} +/- {aggregate.benchmark_avg_attack_std:.2f} | "
            f"{aggregate.games_per_second_mean:.2f} +/- {aggregate.games_per_second_std:.2f} | "
            f"{aggregate.moves_per_second_mean:.1f} +/- {aggregate.moves_per_second_std:.1f} | "
            f"{aggregate.speedup_vs_baseline:.3f}x | "
            f"{aggregate.attack_delta_vs_baseline:+.2f} | "
            f"{'yes' if aggregate.pareto_optimal else 'no'} |"
        )
    return "\n".join(rows)


def main(args: ScriptArgs) -> None:
    validate_args(args)
    source_run_dir = args.source_run_dir.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else default_output_root(source_run_dir)
    )
    output_root.mkdir(parents=True, exist_ok=False)

    source_config = load_training_config(source_run_dir / CONFIG_FILENAME)
    variants = build_variants(source_config, args)
    logger.info(
        "Starting warm-start trunk comparison",
        source_run_dir=str(source_run_dir),
        output_root=str(output_root),
        variants=[variant.label for variant in variants],
        source_trunk_channels=source_config.network.trunk_channels,
        source_reduction_channels=source_config.network.reduction_channels,
    )

    variant_records: list[WarmStartVariantRecord] = []
    for variant_index, variant in enumerate(variants, start=1):
        variant_output_dir = output_root / "runs" / variant.label
        warm_start_args = make_warm_start_args(
            args,
            output_run_dir=variant_output_dir,
            variant=variant,
        )
        variant_output_config = build_output_config(
            source_run_dir=source_run_dir,
            output_run_dir=variant_output_dir,
        )
        variant_output_config.network = build_variant_network_config(
            source_config.network,
            variant,
        )
        num_parameters = count_parameters(variant_output_config.network)
        logger.info(
            "Running warm-start variant",
            variant=variant.label,
            variant_index=variant_index,
            num_variants=len(variants),
            trunk_channels=variant.trunk_channels,
            num_conv_residual_blocks=variant.num_conv_residual_blocks,
            reduction_channels=variant.reduction_channels,
            num_parameters=num_parameters,
            output_run_dir=str(variant_output_dir),
        )
        warm_start_result = run_warm_start(
            warm_start_args,
            output_config=variant_output_config,
        )
        warm_start_final_eval = cast(
            dict[str, object],
            warm_start_result.summary["final_eval"],
        )
        warm_start_num_steps = cast(int, warm_start_result.summary["num_steps"])
        warm_start_offline_best = cast(
            dict[str, object],
            warm_start_result.summary["offline_best"],
        )
        warm_start_avg_attack = cast(float, warm_start_final_eval["avg_attack"])
        warm_start_avg_moves = cast(float, warm_start_final_eval["avg_moves"])
        variant_records.append(
            WarmStartVariantRecord(
                label=variant.label,
                trunk_channels=variant.trunk_channels,
                num_conv_residual_blocks=variant.num_conv_residual_blocks,
                reduction_channels=variant.reduction_channels,
                matches_source=variant.matches_source,
                num_parameters=num_parameters,
                output_run_dir=str(warm_start_result.output_run_dir),
                summary_path=str(warm_start_result.summary_path),
                incumbent_onnx_path=str(warm_start_result.incumbent_onnx_path),
                warm_start_avg_attack=warm_start_avg_attack,
                warm_start_avg_moves=warm_start_avg_moves,
                warm_start_num_steps=warm_start_num_steps,
                warm_start_offline_best=warm_start_offline_best,
            )
        )

    (
        benchmark_config,
        benchmark_num_workers,
        benchmark_num_workers_source,
        benchmark_num_simulations,
        benchmark_max_placements,
        benchmark_mcts_seed,
    ) = build_benchmark_config(source_config, args)
    logger.info(
        "Starting benchmark phase",
        num_workers=benchmark_num_workers,
        num_workers_source=benchmark_num_workers_source,
        num_simulations=benchmark_num_simulations,
        max_placements=benchmark_max_placements,
        benchmark_num_games=args.benchmark_num_games,
        benchmark_repeats=args.benchmark_repeats,
    )

    benchmark_rows: list[BenchmarkRow] = []
    for repeat_index in range(args.benchmark_repeats):
        seed_start = args.benchmark_seed_start + (
            repeat_index * args.benchmark_num_games
        )
        seeds = list(range(seed_start, seed_start + args.benchmark_num_games))
        for variant_record in variant_records:
            logger.info(
                "Benchmarking variant",
                variant=variant_record.label,
                repeat_index=repeat_index,
                seeds=f"{seeds[0]}-{seeds[-1]}",
            )
            metrics = benchmark_once(
                model_path=Path(variant_record.incumbent_onnx_path),
                seeds=seeds,
                config=benchmark_config,
                max_placements=benchmark_max_placements,
                num_workers=benchmark_num_workers,
            )
            benchmark_rows.append(
                BenchmarkRow(
                    variant=variant_record.label,
                    repeat_index=repeat_index,
                    seed_start=seed_start,
                    seed_end=seeds[-1],
                    trunk_channels=variant_record.trunk_channels,
                    num_conv_residual_blocks=(variant_record.num_conv_residual_blocks),
                    reduction_channels=variant_record.reduction_channels,
                    num_parameters=variant_record.num_parameters,
                    elapsed_sec=metrics.elapsed_sec,
                    num_games=metrics.num_games,
                    avg_attack=metrics.avg_attack,
                    avg_lines=metrics.avg_lines,
                    avg_moves=metrics.avg_moves,
                    max_attack=metrics.max_attack,
                    attack_per_piece=metrics.attack_per_piece,
                    avg_tree_nodes=metrics.avg_tree_nodes,
                    total_moves=metrics.total_moves,
                    total_attack=metrics.total_attack,
                    games_per_second=metrics.games_per_second,
                    moves_per_second=metrics.moves_per_second,
                    game_results=metrics.game_results,
                )
            )

    baseline_record = next(
        (record for record in variant_records if record.matches_source),
        variant_records[0],
    )
    baseline_label = baseline_record.label
    aggregates: list[BenchmarkAggregate] = []
    for variant_record in variant_records:
        rows = [row for row in benchmark_rows if row.variant == variant_record.label]
        games_per_second_mean, games_per_second_std = summarize_metric(
            [row.games_per_second for row in rows]
        )
        moves_per_second_mean, moves_per_second_std = summarize_metric(
            [row.moves_per_second for row in rows]
        )
        benchmark_avg_attack_mean, benchmark_avg_attack_std = summarize_metric(
            [row.avg_attack for row in rows]
        )
        benchmark_avg_moves_mean, benchmark_avg_moves_std = summarize_metric(
            [row.avg_moves for row in rows]
        )
        elapsed_mean, elapsed_std = summarize_metric([row.elapsed_sec for row in rows])
        aggregates.append(
            BenchmarkAggregate(
                label=variant_record.label,
                trunk_channels=variant_record.trunk_channels,
                num_conv_residual_blocks=variant_record.num_conv_residual_blocks,
                reduction_channels=variant_record.reduction_channels,
                num_parameters=variant_record.num_parameters,
                matches_source=variant_record.matches_source,
                output_run_dir=variant_record.output_run_dir,
                incumbent_onnx_path=variant_record.incumbent_onnx_path,
                warm_start_avg_attack=variant_record.warm_start_avg_attack,
                warm_start_avg_moves=variant_record.warm_start_avg_moves,
                warm_start_num_steps=variant_record.warm_start_num_steps,
                games_per_second_mean=games_per_second_mean,
                games_per_second_std=games_per_second_std,
                moves_per_second_mean=moves_per_second_mean,
                moves_per_second_std=moves_per_second_std,
                benchmark_avg_attack_mean=benchmark_avg_attack_mean,
                benchmark_avg_attack_std=benchmark_avg_attack_std,
                benchmark_avg_moves_mean=benchmark_avg_moves_mean,
                benchmark_avg_moves_std=benchmark_avg_moves_std,
                elapsed_sec_mean=elapsed_mean,
                elapsed_sec_std=elapsed_std,
            )
        )

    aggregates_by_label = {aggregate.label: aggregate for aggregate in aggregates}
    baseline_games_per_second = aggregates_by_label[
        baseline_label
    ].games_per_second_mean
    baseline_attack = aggregates_by_label[baseline_label].benchmark_avg_attack_mean
    optimal_labels = pareto_optimal_labels(aggregates)
    for aggregate in aggregates:
        aggregate.speedup_vs_baseline = (
            aggregate.games_per_second_mean / baseline_games_per_second
            if baseline_games_per_second > 0
            else 0
        )
        aggregate.attack_delta_vs_baseline = (
            aggregate.benchmark_avg_attack_mean - baseline_attack
        )
        aggregate.pareto_optimal = aggregate.label in optimal_labels

    markdown_table = build_markdown_table(aggregates)
    summary_payload: dict[str, object] = {
        "source_run_dir": str(source_run_dir),
        "output_root": str(output_root),
        "source_network": source_config.network.model_dump(mode="json"),
        "source_self_play": {
            "num_simulations": source_config.self_play.num_simulations,
            "num_workers": source_config.self_play.num_workers,
            "max_placements": source_config.self_play.max_placements,
            "c_puct": source_config.self_play.c_puct,
            "reuse_tree": source_config.self_play.reuse_tree,
        },
        "requested_args": {
            **asdict(args),
            "source_run_dir": str(source_run_dir),
            "output_root": str(output_root),
        },
        "variants": [asdict(variant) for variant in variants],
        "warm_start_runs": [asdict(record) for record in variant_records],
        "benchmark": {
            "num_games": args.benchmark_num_games,
            "repeats": args.benchmark_repeats,
            "seed_start": args.benchmark_seed_start,
            "num_workers": benchmark_num_workers,
            "num_workers_source": benchmark_num_workers_source,
            "num_simulations": benchmark_num_simulations,
            "max_placements": benchmark_max_placements,
            "mcts_seed": benchmark_mcts_seed,
            "baseline_label": baseline_label,
            "pareto_optimal_labels": sorted(optimal_labels),
            "rows": [asdict(row) for row in benchmark_rows],
            "aggregates": [asdict(aggregate) for aggregate in aggregates],
        },
        "comparison_table_markdown": markdown_table,
    }
    summary_json_path = output_root / "comparison_summary.json"
    summary_markdown_path = output_root / "comparison_summary.md"
    benchmark_rows_path = output_root / "benchmark_rows.jsonl"
    write_json(summary_json_path, summary_payload)
    summary_markdown_path.write_text(markdown_table + "\n")
    write_jsonl(benchmark_rows_path, [asdict(row) for row in benchmark_rows])

    logger.info(
        "Warm-start trunk comparison complete",
        output_root=str(output_root),
        summary_json=str(summary_json_path),
        summary_markdown=str(summary_markdown_path),
        benchmark_rows=str(benchmark_rows_path),
        baseline_label=baseline_label,
        pareto_optimal=sorted(optimal_labels),
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
