"""Information-theoretic optimal search depth (Leela Zero issue #1416 style).

For a sample of self-play states, run one long fresh-root MCTS search with the trained
network and snapshot the root visit distribution after x simulations for a dense grid of
x. The visit distribution at the largest budget N is treated as the "ground truth"
policy pi_N, and

    L(x) = KL(pi_N || pi_x)                              (nats)
    information gain per visit  I(x) = (L(0) - L(x)) / x

where pi_0 is the network prior. The budget that maximizes I(x) gives the most policy
information per simulation spent and is the principled per-move search depth.

pi_x has zeros on unvisited actions, which would make the KL infinite, so it is smoothed
by treating the network prior as `alpha` pseudo-visits:

    pi_x(a) = (n_x(a) + alpha * P(a)) / (x + alpha)

alpha = 1 (prior worth one visit) is the headline setting; pi_0 is then exactly the prior.

Usage:
    python investigations/optimal_search_depth/search_depth_info_gain.py \
        --run_dir training_runs/drive_final
    python investigations/optimal_search_depth/search_depth_info_gain.py \
        --run_dir training_runs/drive_final --plot_only true
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import structlog
from simple_parsing import parse

from tetris_core.tetris_core import MCTSAgent, MCTSConfig, TetrisEnv
from tetris_bot.constants import PROJECT_ROOT
from tetris_bot.scripts.utils.run_search_config import (
    build_mcts_config,
    default_model_path,
)

logger = structlog.get_logger()

INVESTIGATION_DIR = PROJECT_ROOT / "investigations" / "optimal_search_depth"
PROB_FLOOR = 1e-12

# Reference categorical palette (light surface) from the dataviz skill.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


@dataclass
class ScriptArgs:
    run_dir: Path = PROJECT_ROOT / "training_runs" / "drive_final"
    model_path: Path | None = None  # Defaults to <run_dir>/checkpoints/incumbent.onnx
    output_dir: Path = INVESTIGATION_DIR / "results"

    num_games: int = 96  # Self-play games used to sample analysis states
    seed_start: int = 10_000
    state_stride: int = 3  # Analyze every Nth decision of each game
    trajectory_add_noise: bool = True  # Match self-play state distribution
    trajectory_num_simulations: int | None = None  # Defaults to run config (2000)

    ground_truth_simulations: list[int] = field(
        default_factory=lambda: [16_384, 32_768, 65_536]
    )
    smoothing_alphas: list[float] = field(default_factory=lambda: [0.25, 1.0, 4.0])
    headline_alpha: float = 1.0
    num_grid_points: int = 260
    reference_simulations: int = 2000  # Former training setting, highlighted on plots

    num_workers: int = max(2, (os.cpu_count() or 4) - 2)
    num_bootstrap: int = 1000
    plot_only: bool = False


@dataclass
class StateRecord:
    game_seed: int
    decision_index: int
    placement_count: int
    num_valid_actions: int
    prior_entropy: float
    ground_truth_entropy: float
    analysis_seconds: float


def simulation_grid(
    max_simulations: int, num_points: int, extra: list[int]
) -> np.ndarray:
    log_grid = np.geomspace(1, max_simulations, num_points).round().astype(np.int64)
    return np.unique(np.concatenate([[0], log_grid, extra])).astype(np.int64)


def entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def kl_curves(
    prior: np.ndarray,
    visits: np.ndarray,
    grid: np.ndarray,
    ground_truths: list[int],
    alphas: list[float],
) -> np.ndarray:
    """Return KL(pi_N || pi_x) with shape (len(ground_truths), len(alphas), len(grid))."""
    floored_prior = np.maximum(prior, PROB_FLOOR)
    grid_index = {int(x): i for i, x in enumerate(grid)}
    out = np.empty((len(ground_truths), len(alphas), len(grid)), dtype=np.float64)
    for g, n in enumerate(ground_truths):
        target = visits[grid_index[n]] / n
        support = target > 0
        q = target[support]
        log_q = np.log(q)
        counts = visits[:, support]
        p = floored_prior[support]
        for a, alpha in enumerate(alphas):
            pi_x = (counts + alpha * p) / (grid[:, None] + alpha)
            out[g, a] = (q * (log_q - np.log(pi_x))).sum(axis=1)
    return out


_WORKER: dict = {}


def _init_worker(args: ScriptArgs, grid: np.ndarray) -> None:
    model_path = str(args.model_path or default_model_path(args.run_dir))
    base = build_mcts_config(args.run_dir)

    trajectory_config = _copy_config(base)
    if args.trajectory_num_simulations is not None:
        trajectory_config.num_simulations = args.trajectory_num_simulations
    trajectory_agent = MCTSAgent(trajectory_config)

    analysis_config = _copy_config(base)
    analysis_config.visit_sampling_epsilon = 0.0
    analysis_agent = MCTSAgent(analysis_config)

    for agent in (trajectory_agent, analysis_agent):
        if not agent.load_model(model_path):
            raise RuntimeError(f"Failed to load model: {model_path}")
    _WORKER.update(
        args=args,
        grid=grid,
        trajectory_agent=trajectory_agent,
        analysis_agent=analysis_agent,
        max_placements=base.max_placements,
    )


def _copy_config(config: MCTSConfig) -> MCTSConfig:
    copy = MCTSConfig()
    for name in dir(config):
        if name.startswith("_") or callable(getattr(config, name)):
            continue
        setattr(copy, name, getattr(config, name))
    return copy


def _analyze_game(game_seed: int) -> tuple[list[StateRecord], np.ndarray]:
    args: ScriptArgs = _WORKER["args"]
    grid: np.ndarray = _WORKER["grid"]
    trajectory_agent: MCTSAgent = _WORKER["trajectory_agent"]
    analysis_agent: MCTSAgent = _WORKER["analysis_agent"]
    max_placements: int = _WORKER["max_placements"]
    ground_truths = sorted(args.ground_truth_simulations)

    env = TetrisEnv.with_seed(10, 20, game_seed)
    records: list[StateRecord] = []
    curves: list[np.ndarray] = []
    decision_index = 0
    while not env.game_over and env.placement_count < max_placements:
        if decision_index % args.state_stride == 0:
            start = time.perf_counter()
            snapshot = analysis_agent.search_root_visit_snapshots(env, grid.tolist())
            elapsed = time.perf_counter() - start
            if snapshot is None:
                break
            prior = np.asarray(snapshot[0], dtype=np.float64)
            visits = np.asarray(snapshot[1], dtype=np.float64)
            curves.append(
                kl_curves(prior, visits, grid, ground_truths, args.smoothing_alphas)
            )
            n_max = ground_truths[-1]
            records.append(
                StateRecord(
                    game_seed=game_seed,
                    decision_index=decision_index,
                    placement_count=int(env.placement_count),
                    num_valid_actions=int((prior > 0).sum()),
                    prior_entropy=entropy(prior),
                    ground_truth_entropy=entropy(
                        visits[int(np.searchsorted(grid, n_max))] / n_max
                    ),
                    analysis_seconds=elapsed,
                )
            )

        result = trajectory_agent.select_action(
            env, add_noise=args.trajectory_add_noise
        )
        if result is None:
            break
        if env.execute_action_index(result.action) is None:
            raise RuntimeError(f"Search chose invalid action {result.action}")
        decision_index += 1

    empty = np.empty(
        (0, len(ground_truths), len(args.smoothing_alphas), len(grid)), dtype=np.float64
    )
    return records, (np.stack(curves) if curves else empty)


def collect(args: ScriptArgs, grid: np.ndarray) -> tuple[list[StateRecord], np.ndarray]:
    seeds = list(range(args.seed_start, args.seed_start + args.num_games))
    records: list[StateRecord] = []
    curves: list[np.ndarray] = []
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        args.num_workers, initializer=_init_worker, initargs=(args, grid)
    ) as pool:
        for i, (game_records, game_curves) in enumerate(
            pool.imap_unordered(_analyze_game, seeds), start=1
        ):
            records.extend(game_records)
            curves.append(game_curves)
            logger.info(
                "game analyzed",
                games=f"{i}/{len(seeds)}",
                states=len(records),
                elapsed_s=round(time.perf_counter() - start, 1),
            )
    order = sorted(
        range(len(records)),
        key=lambda i: (records[i].game_seed, records[i].decision_index),
    )
    stacked = np.concatenate(curves)[order]
    return [records[i] for i in order], stacked


def info_gain(kl: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """(L(0) - L(x)) / x along the last axis; the x = 0 slot is 0 and always masked out."""
    gain = np.zeros_like(kl)
    gain[..., 1:] = (kl[..., :1] - kl[..., 1:]) / grid[1:]
    return gain


def valid_range(grid: np.ndarray, ground_truth: int) -> np.ndarray:
    """Exclude x >= N/2, where pi_x shares most of its visits with pi_N."""
    return (grid >= 1) & (grid <= ground_truth // 2)


def peak(curve: np.ndarray, grid: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    idx = np.where(mask)[0]
    best = idx[int(np.argmax(curve[idx]))]
    return int(grid[best]), float(curve[best])


def bootstrap_peaks(
    gain_per_state: np.ndarray,
    game_ids: np.ndarray,
    grid: np.ndarray,
    mask: np.ndarray,
    num_bootstrap: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample games with replacement; returns (peak x per sample, mean curves)."""
    rng = np.random.default_rng(seed)
    unique_games, inverse = np.unique(game_ids, return_inverse=True)
    per_game_sum = np.zeros((len(unique_games), gain_per_state.shape[1]))
    np.add.at(per_game_sum, inverse, gain_per_state)
    per_game_count = np.bincount(inverse, minlength=len(unique_games)).astype(float)
    peaks = np.empty(num_bootstrap)
    curves = np.empty((num_bootstrap, gain_per_state.shape[1]))
    for b in range(num_bootstrap):
        sample = rng.integers(0, len(unique_games), len(unique_games))
        curve = per_game_sum[sample].sum(0) / per_game_count[sample].sum()
        curves[b] = curve
        peaks[b] = peak(curve, grid, mask)[0]
    return peaks, curves


def summarize(
    args: ScriptArgs,
    records: list[StateRecord],
    kl: np.ndarray,
    grid: np.ndarray,
) -> dict:
    ground_truths = sorted(args.ground_truth_simulations)
    gain = info_gain(kl, grid)  # (S, G, A, X)
    mean_gain = np.mean(gain, axis=0)
    game_ids = np.array([r.game_seed for r in records])
    g_main = len(ground_truths) - 1
    a_main = args.smoothing_alphas.index(args.headline_alpha)
    main_mask = valid_range(grid, ground_truths[g_main])

    peaks_boot, _ = bootstrap_peaks(
        gain[:, g_main, a_main], game_ids, grid, main_mask, args.num_bootstrap
    )
    x_star, gain_star = peak(mean_gain[g_main, a_main], grid, main_mask)
    ref_idx = int(np.searchsorted(grid, args.reference_simulations))
    ref_gain = float(mean_gain[g_main, a_main, ref_idx])

    sensitivity = []
    for g, n in enumerate(ground_truths):
        for a, alpha in enumerate(args.smoothing_alphas):
            x, value = peak(mean_gain[g, a], grid, valid_range(grid, n))
            sensitivity.append(
                {
                    "ground_truth_simulations": n,
                    "alpha": alpha,
                    "peak_simulations": x,
                    "peak_nats_per_visit": value,
                }
            )

    # Within 90% of the peak: the flat region where the exact choice barely matters.
    near = main_mask & (mean_gain[g_main, a_main] >= 0.9 * gain_star)
    return {
        "headline": {
            "ground_truth_simulations": ground_truths[g_main],
            "alpha": args.headline_alpha,
            "optimal_simulations": x_star,
            "optimal_nats_per_visit": gain_star,
            "bootstrap_95ci_simulations": [
                float(np.percentile(peaks_boot, 2.5)),
                float(np.percentile(peaks_boot, 97.5)),
            ],
            "within_90pct_of_peak_simulations": [
                int(grid[near].min()),
                int(grid[near].max()),
            ],
            "reference_simulations": args.reference_simulations,
            "reference_nats_per_visit": ref_gain,
            "reference_relative_efficiency": ref_gain / gain_star,
            "kl_prior_vs_ground_truth_mean": float(kl[:, g_main, a_main, 0].mean()),
            "kl_at_reference_mean": float(kl[:, g_main, a_main, ref_idx].mean()),
        },
        "sensitivity": sensitivity,
        "num_states": len(records),
        "num_games": int(len(np.unique(game_ids))),
        "mean_valid_actions": float(np.mean([r.num_valid_actions for r in records])),
        "mean_prior_entropy": float(np.mean([r.prior_entropy for r in records])),
        "mean_ground_truth_entropy": float(
            np.mean([r.ground_truth_entropy for r in records])
        ),
        "mean_analysis_seconds_per_state": float(
            np.mean([r.analysis_seconds for r in records])
        ),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(args).items()
        },
    }


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def _zoom_to_peak(ax: Axes, curves: list[np.ndarray]) -> None:
    """Small-x gains are strongly negative; frame the positive region around the peak."""
    top = max(float(np.max(c)) for c in curves)
    ax.set_ylim(-0.35 * top, 1.2 * top)


def _figure(ncols: int = 1, width: float = 7.5) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        _style(ax)
    return fig, axes[0]


def plot_all(
    args: ScriptArgs,
    records: list[StateRecord],
    kl: np.ndarray,
    grid: np.ndarray,
    summary: dict,
) -> None:
    out = args.output_dir
    ground_truths = sorted(args.ground_truth_simulations)
    g_main = len(ground_truths) - 1
    a_main = args.smoothing_alphas.index(args.headline_alpha)
    gain = info_gain(kl, grid)
    mean_gain = np.mean(gain, axis=0)
    main_mask = valid_range(grid, ground_truths[g_main])
    game_ids = np.array([r.game_seed for r in records])
    headline = summary["headline"]
    x_star = headline["optimal_simulations"]
    ref = args.reference_simulations

    _, boot_curves = bootstrap_peaks(
        gain[:, g_main, a_main], game_ids, grid, main_mask, args.num_bootstrap
    )
    lo, hi = np.percentile(boot_curves, [2.5, 97.5], axis=0)
    x = grid[main_mask]
    y = mean_gain[g_main, a_main, main_mask]

    # 1) Headline: information gain per visit, log-x and linear-x views.
    fig, (ax_log, ax_lin) = _figure(ncols=2, width=12)
    for ax, xmax in ((ax_log, None), (ax_lin, 8000)):
        sel = x <= (xmax or x.max())
        ax.fill_between(
            x[sel],
            lo[main_mask][sel],
            hi[main_mask][sel],
            color=SERIES[0],
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(x[sel], y[sel], color=SERIES[0], linewidth=2)
        ax.axvline(x_star, color=TEXT_PRIMARY, linewidth=1, linestyle="--")
        ax.axvline(ref, color=SERIES[1], linewidth=1, linestyle=":")
        ax.scatter(
            [x_star],
            [headline["optimal_nats_per_visit"]],
            color=SERIES[0],
            s=40,
            zorder=5,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.set_xlabel("Simulations per move (x)")
        ax.set_ylabel("Information gain per visit (nats)")
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    for ax in (ax_log, ax_lin):
        _zoom_to_peak(ax, [hi[main_mask]])
    ax_log.set_xscale("log")
    ax_log.set_xlim(left=50)
    ax_log.set_title("(L(0) − L(x)) / x, log scale")
    ax_lin.set_title("Same curve, linear scale (Leela Zero #1416 view)")
    ax_lin.set_xlim(0, 8000)
    ax_lin.annotate(
        f"peak {x_star} sims\n{headline['optimal_nats_per_visit']:.2e} nats/visit",
        (x_star, headline["optimal_nats_per_visit"]),
        xytext=(14, 8),
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=9,
    )
    ax_lin.annotate(
        f"current {ref} sims\n{headline['reference_nats_per_visit']:.2e} "
        f"({100 * headline['reference_relative_efficiency']:.0f}% of peak)",
        (ref, headline["reference_nats_per_visit"]),
        xytext=(10, -42),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    fig.suptitle(
        f"Policy information gained per MCTS visit (ground truth = {ground_truths[g_main]:,} sims, "
        f"{summary['num_states']} states, 95% bootstrap band over games)",
        color=TEXT_PRIMARY,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "info_gain_per_visit.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2) Raw KL curve L(x).
    fig, (ax,) = _figure()
    mean_kl = kl[:, g_main, a_main].mean(axis=0)
    sel = grid <= ground_truths[g_main] // 2
    ax.plot(np.maximum(grid[sel], 0.7), mean_kl[sel], color=SERIES[0], linewidth=2)
    ax.axhline(mean_kl[0], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax.annotate(
        "L(0): network prior",
        (1, mean_kl[0]),
        xytext=(4, 4),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.axvline(x_star, color=TEXT_PRIMARY, linewidth=1, linestyle="--")
    ax.axvline(ref, color=SERIES[1], linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Simulations per move (x)")
    ax.set_ylabel("Mean KL(π_N ‖ π_x) (nats)")
    ax.set_title(f"Distance from the {ground_truths[g_main]:,}-sim policy")
    fig.tight_layout()
    fig.savefig(out / "kl_vs_simulations.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 3) Sensitivity: ground-truth budget (alpha fixed) and smoothing alpha (N fixed).
    fig, (ax_gt, ax_alpha) = _figure(ncols=2, width=12)
    for g, n in enumerate(ground_truths):
        m = valid_range(grid, n)
        ax_gt.plot(
            grid[m],
            mean_gain[g, a_main, m],
            color=SERIES[g],
            linewidth=2,
            label=f"N = {n:,}",
        )
    for a, alpha in enumerate(args.smoothing_alphas):
        ax_alpha.plot(
            x,
            mean_gain[g_main, a, main_mask],
            color=SERIES[a],
            linewidth=2,
            label=f"α = {alpha:g}",
        )
    for ax, title in (
        (ax_gt, f"Ground-truth budget N (α = {args.headline_alpha:g})"),
        (ax_alpha, f"Prior pseudo-count α (N = {ground_truths[g_main]:,})"),
    ):
        ax.set_xscale("log")
        _zoom_to_peak(ax, [line.get_ydata() for line in ax.get_lines()])
        ax.axvline(ref, color=SERIES[1], linewidth=1, linestyle=":")
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Simulations per move (x)")
        ax.set_ylabel("Information gain per visit (nats)")
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "sensitivity.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 4) By game phase (placement count buckets).
    placements = np.array([r.placement_count for r in records])
    edges = [0, 17, 34, 50]
    fig, (ax,) = _figure()
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sel_states = (placements >= a) & (placements < b)
        if not sel_states.any():
            continue
        curve = np.mean(gain[sel_states, g_main, a_main], axis=0)[main_mask]
        px, _ = peak(np.mean(gain[sel_states, g_main, a_main], axis=0), grid, main_mask)
        ax.plot(
            x,
            curve,
            color=SERIES[i],
            linewidth=2,
            label=f"placements {a}–{b - 1} (peak {px}, n={sel_states.sum()})",
        )
    ax.set_xscale("log")
    _zoom_to_peak(ax, [line.get_ydata() for line in ax.get_lines()])
    ax.axvline(ref, color=SERIES[3], linewidth=1, linestyle=":")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlabel("Simulations per move (x)")
    ax.set_ylabel("Information gain per visit (nats)")
    ax.set_title("Information gain per visit by game phase")
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "info_gain_by_game_phase.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(args: ScriptArgs) -> None:
    if args.headline_alpha not in args.smoothing_alphas:
        raise ValueError("headline_alpha must be one of smoothing_alphas")
    if args.num_workers < 2:
        raise ValueError("num_workers must be > 1 (parallel generation is required)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "kl_curves.npz"
    records_path = args.output_dir / "states.jsonl"

    if args.plot_only:
        data = np.load(data_path)
        grid, kl = data["grid"], data["kl"]
        records = [
            StateRecord(**json.loads(line))
            for line in records_path.read_text().splitlines()
        ]
    else:
        ground_truths = sorted(args.ground_truth_simulations)
        grid = simulation_grid(
            ground_truths[-1],
            args.num_grid_points,
            [args.reference_simulations, *ground_truths],
        )
        records, kl = collect(args, grid)
        np.savez_compressed(
            data_path,
            grid=grid,
            kl=kl.astype(np.float32),
            ground_truths=np.array(ground_truths),
            alphas=np.array(args.smoothing_alphas),
        )
        records_path.write_text(
            "\n".join(json.dumps(asdict(r)) for r in records) + "\n"
        )

    summary = summarize(args, records, kl.astype(np.float64), grid)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_all(args, records, kl.astype(np.float64), grid, summary)
    logger.info("done", **summary["headline"], output_dir=str(args.output_dir))


if __name__ == "__main__":
    main(parse(ScriptArgs))
