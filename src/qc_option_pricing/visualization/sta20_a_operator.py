"""Sta20-style schematic for the European-call A operator (PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


def save_european_a_operator_sta20_style(
    n_distribution_qubits: int,
    out_path: Path,
    *,
    dpi: int = 300,
) -> None:
    """Save schematic to out_path (2^n grid on n_distribution_qubits)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "Nimbus Roman"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "cm",
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 2.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.55)
    ax.axis("off")

    y_reg, y_pay = 1.78, 0.48
    x_left, x_right = 1.32, 9.35
    lw = 1.35

    ax.text(
        0.08,
        y_reg,
        rf"$|i\rangle_{{{n_distribution_qubits}}}$",
        fontsize=15,
        va="center",
        ha="left",
    )
    ax.text(0.08, y_pay, r"$|0\rangle$", fontsize=15, va="center", ha="left")

    px_x0, px_x1 = 1.52, 3.42
    px_h = 0.58
    px_y0 = y_reg - px_h / 2
    ax.plot([x_left, px_x0], [y_reg, y_reg], color="black", linewidth=lw, solid_capstyle="round")
    ax.add_patch(
        FancyBboxPatch(
            (px_x0, px_y0),
            px_x1 - px_x0,
            px_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.05,
            edgecolor="black",
            facecolor="#fafafa",
        )
    )
    ax.text(
        (px_x0 + px_x1) / 2,
        y_reg,
        r"$P_X^{S}$",
        fontsize=14,
        ha="center",
        va="center",
    )

    x_j = 5.05
    r_w, r_h = 1.82, 0.62
    rx0 = x_j
    ry0 = y_pay - r_h / 2

    ax.plot([px_x1, x_j], [y_reg, y_reg], color="black", linewidth=lw, solid_capstyle="round")
    ax.plot([x_left, x_j], [y_pay, y_pay], color="black", linewidth=lw, solid_capstyle="round")
    ax.plot([x_j, x_j], [y_reg, ry0 + r_h], color="black", linewidth=1.05, zorder=2)
    ax.add_patch(
        Circle((x_j, y_reg), radius=0.075, facecolor="black", edgecolor="black", zorder=4)
    )

    ax.add_patch(
        FancyBboxPatch(
            (rx0, ry0),
            r_w,
            r_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.05,
            edgecolor="black",
            facecolor="#fafafa",
            zorder=3,
        )
    )
    ax.text(
        rx0 + r_w / 2,
        y_pay,
        r"$R_y[\tilde f(i)]$",
        fontsize=13,
        ha="center",
        va="center",
    )

    ax.plot([rx0 + r_w, x_right], [y_reg, y_reg], color="black", linewidth=lw, solid_capstyle="round")
    ax.plot([rx0 + r_w, x_right], [y_pay, y_pay], color="black", linewidth=lw, solid_capstyle="round")

    ax.set_title(
        rf"European call: operator $\mathcal{{A}}$ ($n={n_distribution_qubits}$ "
        r"qubits for $\mathbb{P}(S_T)$ on the discretized grid)",
        fontsize=12,
        pad=14,
    )
    ax.text(
        5.0,
        0.06,
        r"Notation after Stamatopoulos et al.\ (2020), Fig.\ 2 (right); "
        r"$\tilde f$ is the normalized call payoff on the grid. "
        "Implementation: Qiskit Finance LogNormalDistribution + LinearAmplitudeFunction.",
        fontsize=8.5,
        ha="center",
        va="bottom",
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
