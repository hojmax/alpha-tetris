"""Heatmap of seed-paired attack difference over c_puct x policy temperature at one depth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import parse


@dataclass
class Args:
    input: Path = Path("investigations/cpuct_sweep/results/policy_temp.jsonl")
    sims: int = 300
    base_c_puct: float = 1.5
    base_temperature: float = 1.0
    out: Path = Path("investigations/cpuct_sweep/results/cpuct_x_temp_300.png")


def main(args: Args) -> None:
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    attacks = {
        (r["c_puct"], r.get("policy_temperature", 1.0)): np.array(r["attacks"])
        for r in rows
        if r["num_simulations"] == args.sims and r.get("parent_value_fpu", True)
    }
    base = attacks[(args.base_c_puct, args.base_temperature)]
    c_pucts = sorted({c for c, _ in attacks})
    temps = sorted({t for _, t in attacks})
    diff = np.full((len(temps), len(c_pucts)), np.nan)
    sem = np.full_like(diff, np.nan)
    for (c, t), a in attacks.items():
        d = a - base
        diff[temps.index(t), c_pucts.index(c)] = d.mean()
        sem[temps.index(t), c_pucts.index(c)] = d.std(ddof=1) / np.sqrt(len(d))

    fig, ax = plt.subplots(figsize=(1.1 * len(c_pucts) + 2.5, 0.6 * len(temps) + 1.8))
    lim = np.nanmax(np.abs(diff))
    im = ax.imshow(
        diff, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto"
    )
    for i in range(len(temps)):
        for j in range(len(c_pucts)):
            if np.isnan(diff[i, j]):
                continue
            ax.text(
                j,
                i,
                f"{diff[i, j]:+.1f}\n±{sem[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(diff[i, j]) > 0.6 * lim else "black",
            )
    ax.set_xticks(range(len(c_pucts)), [f"{c:g}" for c in c_pucts])
    ax.set_yticks(range(len(temps)), [f"{t:g}" for t in temps])
    ax.set_xlabel("c_puct")
    ax.set_ylabel("policy softmax temperature")
    ax.set_title(
        f"Attack vs c_puct={args.base_c_puct:g}, T={args.base_temperature:g} "
        f"({base.mean():.1f}), {args.sims} sims, {len(base)} paired seeds"
    )
    fig.colorbar(im, ax=ax, label="Δ avg attack")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)


if __name__ == "__main__":
    main(parse(Args))
