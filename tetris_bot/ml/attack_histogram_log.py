"""Rolling distribution of self-play total attack, logged to W&B as a histogram and a plot."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np
import wandb
from matplotlib import pyplot as plt

from tetris_bot.attack_histogram import AttackDistribution, draw_attack_histogram

_SELF_PLAY_COLOR = "#5b8fe0"


class RecentAttackHistogram:
    """Keeps the total attack of the last `window_games` completed self-play games."""

    def __init__(self, *, window_games: int, interval_seconds: float, now_s: float):
        if window_games <= 0:
            raise ValueError(f"window_games must be > 0 (got {window_games})")
        self.interval_seconds = interval_seconds
        self._games: deque[tuple[int, int]] = deque(maxlen=window_games)
        self._next_log_time_s = now_s + interval_seconds

    @property
    def enabled(self) -> bool:
        return self.interval_seconds > 0

    def record(
        self, completed_game_stats: Sequence[tuple[int, Mapping[str, float | int]]]
    ) -> None:
        for game_number, stats in completed_game_stats:
            self._games.append((game_number, int(stats["total_attack"])))

    def is_due(self, now_s: float) -> bool:
        return self.enabled and bool(self._games) and now_s >= self._next_log_time_s

    def wandb_payload(self, now_s: float) -> dict[str, object]:
        """Build the W&B log entry and schedule the next one."""
        self._next_log_time_s = now_s + self.interval_seconds
        game_numbers = [number for number, _ in self._games]
        attacks = np.array([attack for _, attack in self._games])
        # Integer-wide bins so the W&B histogram matches the plotted one.
        edges = np.arange(attacks.min(), attacks.max() + 2) - 0.5
        counts, _ = np.histogram(attacks, bins=edges)
        fig = draw_attack_histogram(
            [AttackDistribution("Self-play", _SELF_PLAY_COLOR, attacks.tolist())],
            title=(
                f"Total attack, last {len(attacks)} self-play games "
                f"(#{min(game_numbers)}-#{max(game_numbers)})"
            ),
        )
        try:
            image = wandb.Image(fig)
        finally:
            plt.close(fig)
        return {
            "game_dist/total_attack": wandb.Histogram(np_histogram=(counts, edges)),
            "game_dist/total_attack_plot": image,
            "game_dist/window_games": len(attacks),
            "game_dist/total_attack_mean": float(attacks.mean()),
            "game_dist/total_attack_p10": float(np.percentile(attacks, 10)),
            "game_dist/total_attack_p50": float(np.percentile(attacks, 50)),
            "game_dist/total_attack_p90": float(np.percentile(attacks, 90)),
        }
