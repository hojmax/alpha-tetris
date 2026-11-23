"""Greedy fixed-seed sweep of c_puct x num_simulations on one trained model.

Appends one JSON row per (num_simulations, c_puct) to `--out` as it finishes, so a
partial run is already plottable; points already in the file are skipped.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT
from tetris_core.tetris_core import MCTSConfig, evaluate_model

logger = structlog.get_logger()

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "training_runs/new_run_plus_deterministic/final_weights/sync_step_3505140.onnx"
)


@dataclass
class Args:
    out: Path  # JSONL results file
    sims: list[int] = field(default_factory=lambda: [300, 1000])
    c_pucts: list[float] = field(default_factory=lambda: [0.75, 1.5, 3.0])
    policy_temperatures: list[float] = field(default_factory=lambda: [1.0])
    num_games: int = 20
    seed_start: int = 5000
    model_path: Path = DEFAULT_MODEL
    num_workers: int = 16
    max_placements: int = 50


def build_config(
    num_simulations: int, c_puct: float, policy_temperature: float, max_placements: int
) -> MCTSConfig:
    # Final incumbent search state: nn_value_weight at cap, zero penalties,
    # plain per-move sims (no visit budget), parent-value init as in training.
    config = MCTSConfig()
    config.num_simulations = num_simulations
    config.c_puct = c_puct
    config.policy_temperature = policy_temperature
    config.max_placements = max_placements
    config.nn_value_weight = 1.0
    config.death_penalty = 0.0
    config.overhang_penalty_weight = 0.0
    config.use_parent_value_for_unvisited_q = True
    config.reuse_tree = True
    config.count_reused_visits = False
    config.temperature = 0.0
    config.visit_sampling_epsilon = 0.0
    config.seed = 0
    return config


def done_points(out: Path) -> set[tuple[int, float, float]]:
    if not out.exists():
        return set()
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    return {
        (r["num_simulations"], r["c_puct"], r.get("policy_temperature", 1.0))
        for r in rows
    }


def main(args: Args) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_games))
    done = done_points(args.out)
    grid = [
        (sims, c_puct, temp)
        for sims in args.sims
        for temp in args.policy_temperatures
        for c_puct in args.c_pucts
    ]
    for num_simulations, c_puct, policy_temperature in grid:
        if (num_simulations, c_puct, policy_temperature) in done:
            continue
        start = time.perf_counter()
        result = evaluate_model(
            model_path=str(args.model_path),
            seeds=seeds,
            config=build_config(
                num_simulations, c_puct, policy_temperature, args.max_placements
            ),
            max_placements=args.max_placements,
            num_workers=args.num_workers,
            add_noise=False,
        )
        attacks = [int(a) for a, _ in result.game_results]
        row = {
            "num_simulations": num_simulations,
            "c_puct": c_puct,
            "policy_temperature": policy_temperature,
            "num_games": len(attacks),
            "seeds": seeds,
            "attacks": attacks,
            "avg_attack": float(np.mean(attacks)),
            "sem": float(np.std(attacks, ddof=1) / np.sqrt(len(attacks))),
            "seconds": time.perf_counter() - start,
            "model_path": str(args.model_path),
        }
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(
            "point done",
            sims=num_simulations,
            c_puct=c_puct,
            policy_temperature=policy_temperature,
            avg_attack=round(row["avg_attack"], 2),
            sem=round(row["sem"], 2),
            seconds=round(row["seconds"], 1),
        )


if __name__ == "__main__":
    main(parse(Args))
