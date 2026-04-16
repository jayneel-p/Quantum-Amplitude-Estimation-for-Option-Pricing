#!/usr/bin/env python3
"""Generate a publication-style schematic of the quantum pricing A-operator."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_MPLCONFIGDIR = RESULTS_DIR / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle


INK = "#111827"
FILL = "#f4f7f7"
BLUE = "#2f5f98"


def block(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    sublabel: str | None = None,
) -> None:
    rect = Rectangle(
        xy,
        width,
        height,
        facecolor=FILL,
        edgecolor=INK,
        linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2 + (0.12 if sublabel else 0.0),
        label,
        ha="center",
        va="center",
        fontsize=11.5,
    )
    if sublabel:
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2 - 0.28,
            sublabel,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#374151",
        )


def wire(ax, y: float, x0: float = 0.75, x1: float = 8.8) -> None:
    ax.plot([x0, x1], [y, y], color=INK, linewidth=1.1)


def arrow(ax, start: tuple[float, float], end: tuple[float, float], lw: float = 1.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=INK,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 3.05)
    ax.axis("off")

    y_price = 2.08
    y_obj = 1.03

    wire(ax, y_price, x0=0.75, x1=7.95)
    wire(ax, y_obj, x0=0.75, x1=7.95)

    ax.text(0.54, y_price, r"$|0\rangle_n$", ha="right", va="center", fontsize=11.5)
    ax.text(0.54, y_obj, r"$|0\rangle$", ha="right", va="center", fontsize=11.5)
    ax.text(1.18, y_price + 0.28, r"price register", ha="right", va="center", fontsize=8.6)
    ax.text(1.18, y_obj - 0.28, r"objective qubit", ha="right", va="center", fontsize=8.6)

    block(
        ax,
        (1.35, y_price - 0.36),
        1.25,
        0.72,
        r"$P_X$",
        r"distribution",
    )

    block(
        ax,
        (3.45, y_obj - 0.52),
        1.9,
        y_price - y_obj + 1.04,
        r"$U_f$",
        r"payoff rotation",
    )

    block(ax, (6.35, y_obj - 0.27), 0.82, 0.54, r"$M$")
    meter = Arc((6.76, y_obj + 0.02), 0.38, 0.38, theta1=0, theta2=180, color=INK, linewidth=1.0)
    ax.add_patch(meter)
    ax.plot([6.76, 6.91], [y_obj + 0.02, y_obj + 0.17], color=INK, linewidth=1.0)

    arrow(ax, (7.18, y_obj), (7.72, y_obj), lw=0.9)
    ax.text(7.86, y_obj, r"$a$", ha="left", va="center", fontsize=12)

    ax.text(3.35, 2.83, r"$A=U_f(P_X\otimes I)$", ha="center", va="center", fontsize=12.8)
    ax.text(6.95, 0.31, r"$C_0=e^{-rT}g_{\max}a$", ha="center", va="center", fontsize=10.6)

    ax.text(
        2.0,
        0.31,
        r"$P_X|0\rangle_n=\sum_i\sqrt{p_i}|i\rangle_n$",
        ha="center",
        va="center",
        fontsize=9.2,
    )
    ax.text(
        4.4,
        0.31,
        r"$U_f|i\rangle|0\rangle=|i\rangle(\sqrt{1-f_i}|0\rangle+\sqrt{f_i}|1\rangle)$",
        ha="center",
        va="center",
        fontsize=8.6,
        color=BLUE,
    )

    ax.text(5.0, y_price + 0.34, r"$f_i=g(x_i)/g_{\max}$", ha="center", va="center", fontsize=9.2)

    fig.tight_layout(pad=0.4)
    path = RESULTS_DIR / "quantum_a_operator_schematic.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
