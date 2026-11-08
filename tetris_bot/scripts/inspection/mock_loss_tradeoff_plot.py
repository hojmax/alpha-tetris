"""Generate a mock illustration of the policy/value loss tradeoff idea.

The plot is purely illustrative — numbers are made up to convey the concept
of fitting a sigmoid from per-head noise injection to a (loss, avg attack)
curve, then using the ratio of the two curves' derivatives at the current
training loss to set the value-loss weight dynamically.

Output: docs/loss_tradeoff_mock.png
"""

from __future__ import annotations


import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tetris_bot.constants import PROJECT_ROOT


def sigmoid_decreasing(x: np.ndarray, ymax: float, k: float, x0: float) -> np.ndarray:
    return ymax / (1.0 + np.exp(k * (x - x0)))


def sigmoid_derivative(x: float, ymax: float, k: float, x0: float) -> float:
    e = np.exp(k * (x - x0))
    return -ymax * k * e / (1.0 + e) ** 2


def tangent_line(
    x_center: float,
    y_center: float,
    slope: float,
    span: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([x_center - span, x_center + span])
    ys = y_center + slope * (xs - x_center)
    return xs, ys


def main() -> None:
    rng = np.random.default_rng(7)

    policy_params = dict(ymax=52.0, k=1.6, x0=2.4)
    value_params = dict(ymax=52.0, k=0.18, x0=22.0)

    policy_noise_levels = np.array([0.0, 0.25, 0.55, 0.95, 1.45, 2.10])
    policy_baseline_loss = 0.9
    policy_loss_points = policy_baseline_loss + 1.6 * policy_noise_levels
    policy_attack_points = sigmoid_decreasing(
        policy_loss_points, **policy_params
    ) + rng.normal(0.0, 1.2, size=policy_loss_points.size)

    value_noise_levels = np.array([0.0, 2.5, 5.0, 8.0, 12.0, 17.0])
    value_baseline_loss = 6.0
    value_loss_points = value_baseline_loss + 2.0 * value_noise_levels
    value_attack_points = sigmoid_decreasing(
        value_loss_points, **value_params
    ) + rng.normal(0.0, 1.2, size=value_loss_points.size)

    current_policy_loss = 1.1
    current_value_loss = 7.5

    current_policy_attack = sigmoid_decreasing(
        np.array([current_policy_loss]), **policy_params
    )[0]
    current_value_attack = sigmoid_decreasing(
        np.array([current_value_loss]), **value_params
    )[0]
    policy_slope = sigmoid_derivative(current_policy_loss, **policy_params)
    value_slope = sigmoid_derivative(current_value_loss, **value_params)

    weight = abs(value_slope) / abs(policy_slope)

    fig = plt.figure(figsize=(16.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.28)
    ax_policy = fig.add_subplot(gs[0, 0])
    ax_value = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])

    policy_color = "#1f77b4"
    value_color = "#d6622a"
    tangent_color = "#222222"

    xs_p = np.linspace(0.3, 5.0, 400)
    ys_p = sigmoid_decreasing(xs_p, **policy_params)
    ax_policy.plot(xs_p, ys_p, color=policy_color, lw=2.2, label="fitted sigmoid")
    ax_policy.scatter(
        policy_loss_points,
        policy_attack_points,
        color=policy_color,
        s=55,
        zorder=5,
        edgecolor="white",
        linewidth=1.0,
    )
    for x, y, n in zip(policy_loss_points, policy_attack_points, policy_noise_levels):
        label = "no noise" if n == 0.0 else f"σ={n:g}"
        ax_policy.annotate(
            label,
            xy=(x, y),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )

    tx, ty = tangent_line(
        current_policy_loss, current_policy_attack, policy_slope, 0.95
    )
    ax_policy.plot(
        tx, ty, color=tangent_color, lw=2.0, label=f"slope ≈ {policy_slope:.1f}"
    )
    ax_policy.scatter(
        [current_policy_loss],
        [current_policy_attack],
        color=tangent_color,
        s=70,
        zorder=6,
        label="current loss",
    )
    ax_policy.annotate(
        f"current  L_p = {current_policy_loss:.2f}",
        xy=(current_policy_loss, current_policy_attack),
        xytext=(110, 28),
        textcoords="offset points",
        fontsize=10,
        color=tangent_color,
        arrowprops=dict(arrowstyle="->", color=tangent_color, lw=0.8),
    )

    ax_policy.set_title("Policy head (vary policy noise, value fixed)", fontsize=12)
    ax_policy.set_xlabel("Policy loss (cross-entropy vs. MCTS policy)")
    ax_policy.set_ylabel("Average attack per game")
    ax_policy.set_xlim(0.3, 5.0)
    ax_policy.set_ylim(-2, 58)
    ax_policy.grid(alpha=0.25)
    ax_policy.legend(loc="lower left", framealpha=0.9, fontsize=9)

    xs_v = np.linspace(2.0, 45.0, 400)
    ys_v = sigmoid_decreasing(xs_v, **value_params)
    ax_value.plot(xs_v, ys_v, color=value_color, lw=2.2, label="fitted sigmoid")
    ax_value.scatter(
        value_loss_points,
        value_attack_points,
        color=value_color,
        s=55,
        zorder=5,
        edgecolor="white",
        linewidth=1.0,
    )
    for x, y, n in zip(value_loss_points, value_attack_points, value_noise_levels):
        label = "no noise" if n == 0.0 else f"σ={n:g}"
        ax_value.annotate(
            label,
            xy=(x, y),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )

    tx, ty = tangent_line(current_value_loss, current_value_attack, value_slope, 8.0)
    ax_value.plot(
        tx, ty, color=tangent_color, lw=2.0, label=f"slope ≈ {value_slope:.2f}"
    )
    ax_value.scatter(
        [current_value_loss],
        [current_value_attack],
        color=tangent_color,
        s=70,
        zorder=6,
        label="current loss",
    )
    ax_value.annotate(
        f"current  L_v = {current_value_loss:.2f}",
        xy=(current_value_loss, current_value_attack),
        xytext=(110, 18),
        textcoords="offset points",
        fontsize=10,
        color=tangent_color,
        arrowprops=dict(arrowstyle="->", color=tangent_color, lw=0.8),
    )

    ax_value.set_title("Value head (bias-perturb value, policy fixed)", fontsize=12)
    ax_value.set_xlabel("Value loss (MSE vs. remaining attack)")
    ax_value.set_ylabel("Average attack per game")
    ax_value.set_xlim(2.0, 45.0)
    ax_value.set_ylim(-2, 58)
    ax_value.grid(alpha=0.25)
    ax_value.legend(loc="lower left", framealpha=0.9, fontsize=9)

    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    ax_text.text(
        0.02,
        0.98,
        "Use marginal slopes to balance the heads",
        fontsize=15,
        weight="bold",
        va="top",
    )
    ax_text.text(
        0.02,
        0.88,
        "At every training step, read off the slope of\n"
        "each sigmoid at the current loss, then set the\n"
        "value-loss weight so a unit of gradient on\n"
        "each head buys the same expected attack.",
        fontsize=12.5,
        va="top",
        color="#333333",
    )

    box_y = 0.60
    ax_text.text(
        0.02,
        box_y,
        "Current state",
        fontsize=13.5,
        weight="bold",
        va="top",
    )
    ax_text.text(
        0.02,
        box_y - 0.08,
        f"  L_p = {current_policy_loss:.2f}     L_v = {current_value_loss:.2f}",
        fontsize=12.5,
        va="top",
        family="monospace",
    )
    ax_text.text(
        0.02,
        box_y - 0.17,
        f"  dA/dL_p ≈ {policy_slope:.2f}   (policy slope)\n"
        f"  dA/dL_v ≈ {value_slope:.2f}   (value slope)",
        fontsize=12.5,
        va="top",
        family="monospace",
        color="#333333",
    )

    ax_text.text(
        0.02,
        0.30,
        "Weight rule",
        fontsize=13.5,
        weight="bold",
        va="top",
    )
    ax_text.text(
        0.02,
        0.22,
        r"$w_v \;=\; \dfrac{|dA/dL_v|}{|dA/dL_p|}$"
        f"  ≈  {abs(value_slope):.2f} / {abs(policy_slope):.2f}  ≈  {weight:.3f}",
        fontsize=14,
        va="top",
    )
    ax_text.text(
        0.02,
        0.08,
        "total_loss  =  policy_loss\n"
        f"             +  {weight:.3f} · value_loss",
        fontsize=12.5,
        va="top",
        family="monospace",
        color="#111111",
    )

    fig.suptitle(
        "Mock idea plot — dynamic policy/value loss weighting from sigmoid fits",
        fontsize=13.5,
        y=1.005,
    )

    out_path = PROJECT_ROOT / "docs" / "loss_tradeoff_mock.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
