"""Benchmark training throughput across different batch sizes.

Uses synthetic data to isolate training step performance from the data pipeline.
Measures forward + backward pass time, examples/sec, and memory usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
import torch
from simple_parsing import parse

from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
)
from tetris_bot.ml.loss import compute_loss
from tetris_bot.ml.network import AUX_FEATURES, TetrisNet
from tetris_bot.ml.optimizer import OptimizerBundle

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    """Batch size benchmark arguments."""

    batch_sizes: list[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048, 4096]
    )
    warmup_steps: int = 20  # Steps to discard before timing
    bench_steps: int = 100  # Steps to time
    device: str = "auto"  # auto/cpu/cuda/mps
    value_loss_weight: float = 30.0


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_synthetic_batch(
    batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = torch.randint(
        0,
        2,
        (batch_size, 1, BOARD_HEIGHT, BOARD_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    aux = torch.randn(batch_size, AUX_FEATURES, device=device)
    # Create a valid policy target: random sparse distribution over a subset of actions
    masks = torch.zeros(batch_size, NUM_ACTIONS, device=device)
    for i in range(batch_size):
        num_valid = torch.randint(10, 200, (1,)).item()
        valid_indices = torch.randperm(NUM_ACTIONS)[:num_valid]
        masks[i, valid_indices] = 1.0
    policy_targets = torch.rand(batch_size, NUM_ACTIONS, device=device) * masks
    policy_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True).clamp(
        min=1e-8
    )
    value_targets = torch.rand(batch_size, device=device) * 10.0
    return boards, aux, policy_targets, value_targets, masks


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_batch_size(
    batch_size: int,
    device: torch.device,
    warmup_steps: int,
    bench_steps: int,
    value_loss_weight: float,
) -> dict | None:
    try:
        batch = make_synthetic_batch(batch_size, device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
            logger.warning("OOM creating batch", batch_size=batch_size)
            return None
        raise

    model = TetrisNet(
        trunk_channels=16,
        num_conv_residual_blocks=1,
        reduction_channels=32,
        fc_hidden=128,
        conv_kernel_size=3,
        conv_padding=1,
    ).to(device)
    optimizer = OptimizerBundle(
        model,
        learning_rate=0.0005,
        weight_decay=1e-4,
    )
    model.train()

    boards, aux, policy_targets, value_targets, masks = batch

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        try:
            total_loss, _, _ = compute_loss(
                model,
                boards,
                aux,
                policy_targets,
                value_targets,
                masks,
                value_loss_weight,
            )
            total_loss.backward()
            optimizer.step()
        except RuntimeError as e:
            if (
                "out of memory" in str(e).lower()
                or "not enough memory" in str(e).lower()
            ):
                logger.warning("OOM during warmup", batch_size=batch_size)
                return None
            raise

    sync_device(device)

    # Timed run
    step_times: list[float] = []
    for _ in range(bench_steps):
        sync_device(device)
        t0 = time.perf_counter()

        optimizer.zero_grad()
        total_loss, policy_loss, value_loss = compute_loss(
            model, boards, aux, policy_targets, value_targets, masks, value_loss_weight
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sync_device(device)
        step_times.append(time.perf_counter() - t0)

    avg_ms = sum(step_times) / len(step_times) * 1000
    median_ms = sorted(step_times)[len(step_times) // 2] * 1000
    p95_ms = sorted(step_times)[int(len(step_times) * 0.95)] * 1000
    examples_per_sec = batch_size / (sum(step_times) / len(step_times))

    return {
        "batch_size": batch_size,
        "avg_step_ms": avg_ms,
        "median_step_ms": median_ms,
        "p95_step_ms": p95_ms,
        "examples_per_sec": examples_per_sec,
    }


def main(args: ScriptArgs) -> None:
    device = torch.device(get_best_device() if args.device == "auto" else args.device)
    logger.info("Benchmarking training step throughput", device=str(device))
    logger.info(
        "Config",
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        batch_sizes=args.batch_sizes,
    )

    results: list[dict] = []

    for bs in args.batch_sizes:
        logger.info("Benchmarking", batch_size=bs)
        result = benchmark_batch_size(
            bs, device, args.warmup_steps, args.bench_steps, args.value_loss_weight
        )
        if result is None:
            logger.warning("Skipped (OOM)", batch_size=bs)
            continue
        results.append(result)
        logger.info(
            "Result",
            batch_size=bs,
            avg_step_ms=f"{result['avg_step_ms']:.1f}",
            examples_per_sec=f"{result['examples_per_sec']:.0f}",
        )

    if not results:
        logger.error("No batch sizes completed successfully")
        return

    # Print summary table
    print("\n" + "=" * 72)
    print(
        f"{'Batch Size':>12} {'Avg (ms)':>10} {'Median (ms)':>12} {'P95 (ms)':>10} {'Examples/s':>12}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"{r['batch_size']:>12} "
            f"{r['avg_step_ms']:>10.1f} "
            f"{r['median_step_ms']:>12.1f} "
            f"{r['p95_step_ms']:>10.1f} "
            f"{r['examples_per_sec']:>12.0f}"
        )
    print("=" * 72)

    best = max(results, key=lambda r: r["examples_per_sec"])
    print(
        f"\nBest throughput: batch_size={best['batch_size']} ({best['examples_per_sec']:.0f} examples/sec)"
    )


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
