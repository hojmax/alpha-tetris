"""Plot avg attack vs c_puct, one curve per num_simulations, from sweep JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simple_parsing import parse


@dataclass
class Args:
    inputs: list[Path] = field(default_factory=list)
    out: Path = Path("investigations/cpuct_sweep/results/cpuct_sweep.png")


def main(args: Args) -> None:
    rows = [
        json.loads(line)
        for path in args.inputs
        if path.exists()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    by_sims: dict[int, list[dict]] = {}
    for row in rows:
        by_sims.setdefault(row["num_simulations"], []).append(row)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("viridis")
    sims_sorted = sorted(by_sims)
    for i, sims in enumerate(sims_sorted):
        pts = sorted(by_sims[sims], key=lambda r: r["c_puct"])
        color = cmap(i / max(1, len(sims_sorted) - 1) * 0.85)
        ax.errorbar(
            [r["c_puct"] for r in pts],
            [r["avg_attack"] for r in pts],
            yerr=[r["sem"] for r in pts],
            marker="o",
            capsize=3,
            color=color,
            label=f"{sims} sims",
        )
    n_games = "/".join(str(n) for n in sorted({r["num_games"] for r in rows}))
    ax.set_xscale("log")
    ax.set_xticks(sorted({r["c_puct"] for r in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("c_puct (log scale)")
    ax.set_ylabel("avg total attack (greedy, ±1 SEM)")
    ax.set_title(f"Attack vs c_puct, same {n_games} seeds per point")
    ax.grid(alpha=0.3)
    ax.legend(title="search depth")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)


if __name__ == "__main__":
    main(parse(Args))
