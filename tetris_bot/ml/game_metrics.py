from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from tetris_bot.ml.aux_features import AUX_FEATURE_LAYOUT


def compute_batch_feature_metrics(
    aux: torch.Tensor,
    value_targets: torch.Tensor,
    overhang_fields: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, float]:
    layout = AUX_FEATURE_LAYOUT
    row_fill_counts = aux[:, layout.row_fill_counts]
    max_column_heights = aux[:, layout.max_column_height]
    total_blocks = aux[:, layout.total_blocks]
    bumpiness = aux[:, layout.bumpiness]
    holes = aux[:, layout.holes]

    means = {
        "batch/value_target_mean": value_targets.mean(),
        "batch/valid_actions_mean": masks.sum(dim=1).float().mean(),
        "batch/board_fill_mean": total_blocks.mean(),
        "batch/max_height_mean": max_column_heights.mean(),
        "batch/row_fill_mean": row_fill_counts.mean(),
        "batch/bumpiness_mean": bumpiness.mean(),
        "batch/holes_mean": holes.mean(),
        "batch/overhang_fields_mean": overhang_fields.mean(),
    }
    # One device->host transfer (a single sync) instead of one per metric.
    values = torch.stack(list(means.values())).tolist()
    return dict(zip(means.keys(), values, strict=True))


def average_completed_games(
    completed_games: Sequence[tuple[int, Mapping[str, float | int]]],
) -> dict[str, float]:
    """Average all per-game stats using the same metric names as individual game logging."""
    if not completed_games:
        return {}

    sums: dict[str, float] = {}
    episode_length_sum = 0.0

    for _, game_stats in completed_games:
        episode_length = float(game_stats["episode_length"])
        if episode_length <= 0.0:
            raise ValueError(
                "Invalid episode_length while averaging completed games: "
                f"{episode_length}"
            )
        episode_length_sum += episode_length
        for key, value in game_stats.items():
            sums[key] = sums.get(key, 0.0) + float(value)

    n = float(len(completed_games))
    metrics = {f"game/{key}": total / n for key, total in sums.items()}
    first_game_number = float(completed_games[0][0])
    last_game_number = float(completed_games[-1][0])
    metrics["game_number"] = last_game_number
    metrics["game/number"] = last_game_number
    metrics["game/window_size"] = n
    metrics["game/window_first_number"] = first_game_number
    metrics["game/window_last_number"] = last_game_number

    # Derived per-move rates (computed over pooled moves, not averaged per game)
    attack_sum = sums.get("total_attack", 0.0)
    lines_sum = sums.get("total_lines", 0.0)
    holds_sum = sums.get("holds", 0.0)
    metrics["game/attack_per_move"] = attack_sum / episode_length_sum
    metrics["game/lines_per_move"] = lines_sum / episode_length_sum
    metrics["game/hold_rate"] = holds_sum / episode_length_sum

    return metrics


def summarize_completed_games(
    completed_games: Sequence[tuple[int, Mapping[str, float | int]]],
) -> dict[str, float]:
    if not completed_games:
        return {}

    attack_sum = 0.0
    line_sum = 0.0
    episode_length_sum = 0.0
    holds_sum = 0.0
    traversal_total_sum = 0.0
    traversal_expansion_sum = 0.0
    traversal_terminal_sum = 0.0
    traversal_horizon_sum = 0.0
    max_attack = float("-inf")
    max_lines = float("-inf")

    for _, game_stats in completed_games:
        total_attack = float(game_stats["total_attack"])
        total_lines = float(game_stats["total_lines"])
        episode_length = float(game_stats["episode_length"])
        holds = float(game_stats["holds"])
        if episode_length <= 0.0:
            raise ValueError(
                "Invalid episode_length while aggregating completed games: "
                f"{episode_length}"
            )
        attack_sum += total_attack
        line_sum += total_lines
        episode_length_sum += episode_length
        holds_sum += holds
        traversal_total_sum += float(game_stats.get("traversal_total", 0.0))
        traversal_expansion_sum += float(game_stats.get("traversal_expansions", 0.0))
        traversal_terminal_sum += float(game_stats.get("traversal_terminal_ends", 0.0))
        traversal_horizon_sum += float(game_stats.get("traversal_horizon_ends", 0.0))
        max_attack = max(max_attack, total_attack)
        max_lines = max(max_lines, total_lines)

    completed_count = float(len(completed_games))
    first_game_number = float(completed_games[0][0])
    last_game_number = float(completed_games[-1][0])
    metrics = {
        "replay/completed_games_logged": completed_count,
        "replay/completed_games_first_number": first_game_number,
        "replay/completed_games_last_number": last_game_number,
        "replay/completed_games_avg_attack": attack_sum / completed_count,
        "replay/completed_games_avg_lines": line_sum / completed_count,
        "replay/completed_games_avg_moves": episode_length_sum / completed_count,
        "replay/completed_games_max_attack": max_attack,
        "replay/completed_games_max_lines": max_lines,
        "replay/completed_games_avg_attack_per_move": attack_sum / episode_length_sum,
        "replay/completed_games_avg_hold_rate": holds_sum / episode_length_sum,
    }
    if traversal_total_sum > 0.0:
        metrics["replay/completed_games_traversal_expansion_fraction"] = (
            traversal_expansion_sum / traversal_total_sum
        )
        metrics["replay/completed_games_traversal_terminal_fraction"] = (
            traversal_terminal_sum / traversal_total_sum
        )
        metrics["replay/completed_games_traversal_horizon_fraction"] = (
            traversal_horizon_sum / traversal_total_sum
        )
    return metrics
