"""Histogram of total episode attack, shared by the paper plot and W&B training logs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")

_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#333333",
    "axes.labelsize": 11,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
_MEAN_LINESTYLE = (0, (5, 3))


@dataclass(frozen=True)
class AttackDistribution:
    name: str
    color: str | None
    attacks: Sequence[int]


def draw_attack_histogram(
    distributions: Sequence[AttackDistribution],
    *,
    title: str,
    width_inches: float = 9.0,
    height_inches: float = 5.4,
    bin_width: int = 1,
) -> Figure:
    """Overlaid integer-binned histograms with mean lines and min/max labels.

    The caller owns the returned figure and must close it (`plt.close(fig)`).
    """
    if not distributions or any(len(d.attacks) == 0 for d in distributions):
        raise ValueError("every distribution needs at least one game")

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(width_inches, height_inches), constrained_layout=True
        )
        overall_min = min(min(d.attacks) for d in distributions)
        overall_max = max(max(d.attacks) for d in distributions)
        bins = np.arange(overall_min, overall_max + bin_width + 1, bin_width) - 0.5
        bin_centers = (bins[:-1] + bins[1:]) / 2
        alpha = 0.85 if len(distributions) == 1 else 0.65

        legend_handles: list[tuple[Patch | Line2D, str]] = []
        annotations: list[tuple[float, float, str, str]] = []
        for index, distribution in enumerate(distributions):
            color = distribution.color or f"C{index}"
            attacks = np.asarray(distribution.attacks)
            mean_attack = float(attacks.mean())
            min_attack = int(attacks.min())
            max_attack = int(attacks.max())

            counts, _, _ = ax.hist(
                attacks,
                bins=bins.tolist(),
                color=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=alpha,
            )
            ax.axvline(
                mean_attack,
                color=color,
                linestyle=_MEAN_LINESTYLE,
                linewidth=1.8,
                alpha=0.95,
            )
            for value, label in (
                (min_attack, f"Min {min_attack}"),
                (max_attack, f"Max {max_attack}"),
            ):
                bar_idx = int(np.argmin(np.abs(bin_centers - value)))
                annotations.append((value, float(counts[bar_idx]), label, color))

            legend_handles.append(
                (
                    Patch(
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=alpha,
                    ),
                    f"{distribution.name}  (n = {len(attacks)}, "
                    f"Min = {min_attack}, Max = {max_attack})",
                )
            )
            legend_handles.append(
                (
                    Line2D(
                        [0], [0], color=color, linestyle=_MEAN_LINESTYLE, linewidth=1.8
                    ),
                    f"Mean = {mean_attack:.2f}",
                )
            )

        # Pad the y-axis so min/max labels don't bump into the title.
        ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
        for x, y, label, color in annotations:
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

        ax.set_xlabel("Total episode attack")
        ax.set_ylabel("Number of games")
        ax.set_title(title, loc="center", fontsize=14, fontweight="semibold", pad=14)
        ax.grid(True, axis="y", alpha=0.35, linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(x=0.02)
        ax.tick_params(length=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(
            [handle for handle, _ in legend_handles],
            [text for _, text in legend_handles],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            edgecolor="#e0e0e0",
            fontsize=9,
            handlelength=1.6,
            labelspacing=0.6,
        )
    return fig
