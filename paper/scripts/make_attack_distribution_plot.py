from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog
from matplotlib import pyplot as plt
from simple_parsing import parse

from tetris_bot.attack_histogram import AttackDistribution, draw_attack_histogram
from tetris_bot.constants import PROJECT_ROOT

logger = structlog.get_logger()


@dataclass(frozen=True)
class Distribution:
    name: str
    color: str | None
    attacks: list[int]
    sort_order: int


@dataclass
class PlotArgs:
    results_root: Path = PROJECT_ROOT / "paper" / "results" / "attack_distribution"
    output_path: Path = PROJECT_ROOT / "paper" / "plots" / "attack_distribution.pdf"
    width_inches: float = 9.0
    height_inches: float = 5.4
    bin_width: int = 1
    dpi: int = 300


def load_distributions(results_root: Path) -> list[Distribution]:
    summary_paths = sorted(results_root.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(
            f"No attack distribution summaries found under {results_root}."
        )

    distributions: list[Distribution] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text())
        attacks_payload = payload["attacks"]
        if not isinstance(attacks_payload, list) or not attacks_payload:
            raise ValueError(f"summary file has no attacks: {summary_path}")
        plot_config = payload.get("plot", {})
        if not isinstance(plot_config, dict):
            raise ValueError(f"plot config must be an object: {summary_path}")
        distributions.append(
            Distribution(
                name=str(payload["label"]),
                color=(
                    str(plot_config["color"])
                    if plot_config.get("color") is not None
                    else None
                ),
                attacks=[int(a) for a in attacks_payload],
                sort_order=int(plot_config.get("sort_order", 0)),
            )
        )

    ordered = sorted(distributions, key=lambda d: (d.sort_order, d.name))
    logger.info(
        "Loaded attack distributions",
        results_root=str(results_root),
        distributions=[d.name for d in ordered],
    )
    return ordered


def write_plot(
    results_root: Path,
    output_path: Path,
    *,
    width_inches: float,
    height_inches: float,
    bin_width: int,
    dpi: int,
) -> None:
    distributions = load_distributions(results_root)
    total_n = sum(len(d.attacks) for d in distributions)
    title = (
        f"Histogram of Total Episode Attack ({total_n} games)"
        if len(distributions) == 1
        else "Histogram of Total Episode Attack"
    )
    fig = draw_attack_histogram(
        [
            AttackDistribution(name=d.name, color=d.color, attacks=d.attacks)
            for d in distributions
        ],
        title=title,
        width_inches=width_inches,
        height_inches=height_inches,
        bin_width=bin_width,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info(
        "Wrote attack distribution plot",
        path=str(output_path),
        results_root=str(results_root),
    )


def main() -> None:
    args = parse(PlotArgs)
    write_plot(
        args.results_root,
        args.output_path,
        width_inches=args.width_inches,
        height_inches=args.height_inches,
        bin_width=args.bin_width,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
