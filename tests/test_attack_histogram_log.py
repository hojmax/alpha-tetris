from __future__ import annotations

import numpy as np
import pytest
import wandb

from tetris_bot.ml.attack_histogram_log import RecentAttackHistogram


def _games(
    attacks: list[int], first_number: int = 1
) -> list[tuple[int, dict[str, int]]]:
    return [
        (first_number + i, {"total_attack": attack, "episode_length": 50})
        for i, attack in enumerate(attacks)
    ]


def test_keeps_only_the_last_window_of_games() -> None:
    histogram = RecentAttackHistogram(window_games=3, interval_seconds=60.0, now_s=0.0)
    histogram.record(_games([10, 20, 30, 40, 50]))

    payload = histogram.wandb_payload(now_s=60.0)

    assert payload["game_dist/window_games"] == 3
    assert payload["game_dist/total_attack_mean"] == pytest.approx(40.0)
    hist = payload["game_dist/total_attack"]
    assert isinstance(hist, wandb.Histogram)
    assert sum(hist.histogram) == 3
    # Integer-wide bins centred on whole attack values.
    assert np.allclose(np.diff(hist.bins), 1.0)
    assert hist.bins[0] == pytest.approx(29.5)
    assert isinstance(payload["game_dist/total_attack_plot"], wandb.Image)


def test_due_only_after_interval_and_with_games() -> None:
    histogram = RecentAttackHistogram(window_games=10, interval_seconds=60.0, now_s=0.0)
    assert not histogram.is_due(now_s=120.0)  # no games yet

    histogram.record(_games([5]))
    assert not histogram.is_due(now_s=59.0)
    assert histogram.is_due(now_s=60.0)

    histogram.wandb_payload(now_s=70.0)
    assert not histogram.is_due(now_s=129.0)
    assert histogram.is_due(now_s=130.0)


def test_zero_interval_disables_logging() -> None:
    histogram = RecentAttackHistogram(window_games=10, interval_seconds=0.0, now_s=0.0)
    histogram.record(_games([5, 6]))
    assert not histogram.is_due(now_s=1e9)
