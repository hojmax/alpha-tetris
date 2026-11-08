from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from tetris_core.tetris_core import MCTSConfig
from tetris_bot.constants import (
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    INCUMBENT_ONNX_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
)
from tetris_bot.ml.config import load_training_config


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return load_training_config(config_path).model_dump(mode="json")


def load_self_play_config(run_dir: Path) -> dict[str, Any]:
    data = load_run_config(run_dir)
    self_play = data.get("self_play", data)
    if not isinstance(self_play, dict):
        raise ValueError(
            f"self_play config must be a mapping: {run_dir / CONFIG_FILENAME}"
        )
    return dict(self_play)


def default_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME


def default_model_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_DIRNAME / INCUMBENT_ONNX_FILENAME


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint state must be a dict: {checkpoint_path}")
    return state


def apply_checkpoint_search_overrides(
    self_play_config: dict[str, Any],
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(self_play_config)
    nn_schedule = dict(resolved.get("nn_value_weight_schedule") or {})
    penalty_schedule = dict(resolved.get("penalty_schedule") or {})

    incumbent_nn_value_weight = checkpoint_state.get("incumbent_nn_value_weight")
    if incumbent_nn_value_weight is not None:
        restored_nn_value_weight = float(incumbent_nn_value_weight)
        if (
            not math.isfinite(restored_nn_value_weight)
            or restored_nn_value_weight < 0.0
        ):
            raise ValueError(
                "Checkpoint incumbent_nn_value_weight must be finite and >= 0 "
                f"(got {restored_nn_value_weight})"
            )
        nn_schedule["initial"] = restored_nn_value_weight

    incumbent_death_penalty = checkpoint_state.get("incumbent_death_penalty")
    incumbent_overhang_penalty_weight = checkpoint_state.get(
        "incumbent_overhang_penalty_weight"
    )
    if (
        incumbent_death_penalty is not None
        and incumbent_overhang_penalty_weight is not None
    ):
        restored_death_penalty = float(incumbent_death_penalty)
        restored_overhang_penalty_weight = float(incumbent_overhang_penalty_weight)
        if not math.isfinite(restored_death_penalty) or restored_death_penalty < 0.0:
            raise ValueError(
                "Checkpoint incumbent_death_penalty must be finite and >= 0 "
                f"(got {restored_death_penalty})"
            )
        if (
            not math.isfinite(restored_overhang_penalty_weight)
            or restored_overhang_penalty_weight < 0.0
        ):
            raise ValueError(
                "Checkpoint incumbent_overhang_penalty_weight must be finite and >= 0 "
                f"(got {restored_overhang_penalty_weight})"
            )
        penalty_schedule["death_penalty"] = restored_death_penalty
        penalty_schedule["overhang_penalty_weight"] = restored_overhang_penalty_weight
    else:
        nn_value_weight = float(nn_schedule.get("initial", 0.0))
        nn_value_weight_cap = float(nn_schedule.get("cap", nn_value_weight))
        if nn_value_weight >= nn_value_weight_cap:
            penalty_schedule["death_penalty"] = 0.0
            penalty_schedule["overhang_penalty_weight"] = 0.0

    resolved["nn_value_weight_schedule"] = nn_schedule
    resolved["penalty_schedule"] = penalty_schedule
    return resolved


def load_effective_self_play_config(
    run_dir: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    self_play_config = load_self_play_config(run_dir)
    resolved_checkpoint_path = (
        checkpoint_path
        if checkpoint_path is not None
        else default_checkpoint_path(run_dir)
    )
    if not resolved_checkpoint_path.exists():
        return self_play_config

    checkpoint_state = load_checkpoint_state(resolved_checkpoint_path)
    return apply_checkpoint_search_overrides(self_play_config, checkpoint_state)


def build_mcts_config(
    run_dir: Path,
    checkpoint_path: Path | None = None,
    *,
    track_value_history: bool = False,
) -> MCTSConfig:
    config_data = load_effective_self_play_config(run_dir, checkpoint_path)

    nn_schedule = config_data.get("nn_value_weight_schedule") or {}
    penalty_schedule = config_data.get("penalty_schedule") or {}

    config = MCTSConfig()
    config.num_simulations = int(config_data["num_simulations"])
    config.c_puct = float(config_data["c_puct"])
    config.temperature = float(config_data["temperature"])
    config.dirichlet_alpha = float(config_data["dirichlet_alpha"])
    config.dirichlet_epsilon = float(config_data["dirichlet_epsilon"])
    config.visit_sampling_epsilon = float(
        config_data.get("visit_sampling_epsilon", 0.0)
    )
    config.max_placements = int(config_data["max_placements"])
    config.reuse_tree = bool(config_data.get("reuse_tree", True))
    config.use_parent_value_for_unvisited_q = bool(
        config_data["use_parent_value_for_unvisited_q"]
    )
    config.count_reused_visits = bool(config_data.get("count_reused_visits", False))
    config.min_new_simulations = int(config_data.get("min_new_simulations", 0))
    config.nn_value_weight = float(nn_schedule.get("initial", config.nn_value_weight))
    config.death_penalty = float(
        penalty_schedule.get("death_penalty", config.death_penalty)
    )
    config.overhang_penalty_weight = float(
        penalty_schedule.get("overhang_penalty_weight", config.overhang_penalty_weight)
    )
    mcts_seed = config_data.get("mcts_seed")
    config.seed = int(mcts_seed) if mcts_seed is not None else None
    config.track_value_history = track_value_history
    return config
