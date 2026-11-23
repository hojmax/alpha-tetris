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
    x_key: str = "c_puct"  # Row field on the x axis: c_puct or policy_temperature
    fixed_c_puct: float | None = None  # Keep only rows with this c_puct


def main(args: Args) -> None:
    rows = [
        json.loads(line)
        for path in args.inputs
        if path.exists()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        row.setdefault("policy_temperature", 1.0)
    if args.fixed_c_puct is not None:
        rows = [r for r in rows if r["c_puct"] == args.fixed_c_puct]
    x_label = {"c_puct": "c_puct", "policy_temperature": "policy softmax temperature"}[
        args.x_key
    ]
    groups: dict[tuple[int, bool], list[dict]] = {}
    for row in rows:
        key = (row["num_simulations"], row.get("parent_value_fpu", True))
        groups.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("viridis")
    sims_sorted = sorted({sims for sims, _ in groups})
    for key in sorted(groups, key=lambda k: (k[0], not k[1])):
        sims, fpu = key
        pts = sorted(groups[key], key=lambda r: r[args.x_key])
        i = sims_sorted.index(sims)
        color = cmap(i / max(1, len(sims_sorted) - 1) * 0.85)
        ax.errorbar(
            [r[args.x_key] for r in pts],
            [r["avg_attack"] for r in pts],
            yerr=[r["sem"] for r in pts],
            marker="o" if fpu else "s",
            linestyle="-" if fpu else "--",
            capsize=3,
            color=color,
            label=f"{sims} sims" + ("" if fpu else ", unvisited Q = 0"),
        )
    n_games = "/".join(str(n) for n in sorted({r["num_games"] for r in rows}))
    ax.set_xscale("log")
    ax.set_xticks(sorted({r[args.x_key] for r in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"{x_label} (log scale)")
    ax.set_ylabel("avg total attack (greedy, ±1 SEM)")
    fixed = "" if args.fixed_c_puct is None else f", c_puct={args.fixed_c_puct:g}"
    ax.set_title(f"Attack vs {x_label}{fixed}, same {n_games} seeds per point")
    ax.grid(alpha=0.3)
    ax.legend(title="search depth")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)


if __name__ == "__main__":
    main(parse(Args))
