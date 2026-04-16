#!/usr/bin/env python3
"""Recreate the three-qubit log-normal input distribution from Stamatopoulos Fig. 3."""

from __future__ import annotations

import math
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
import numpy as np


def lognormal_pdf(x: np.ndarray, mu: float, sigma_t: float) -> np.ndarray:
    """Risk-neutral log-normal density evaluated on a grid."""
    return (
        1.0
        / (x * sigma_t * math.sqrt(2.0 * math.pi))
        * np.exp(-((np.log(x) - mu) ** 2) / (2.0 * sigma_t**2))
    )


def main() -> None:
    n_qubits = 3
    s0 = 2.0
    sigma = 0.10
    r = 0.04
    t = 300.0 / 365.0
    low, high = 1.5, 2.5

    mu = math.log(s0) + (r - 0.5 * sigma**2) * t
    sigma_t = sigma * math.sqrt(t)

    values = np.linspace(low, high, 2**n_qubits)
    probabilities = lognormal_pdf(values, mu, sigma_t)
    probabilities = probabilities / probabilities.sum()

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.bar(
        values,
        probabilities,
        width=0.105,
        color="#5d8f8b",
        edgecolor="black",
        linewidth=0.65,
    )

    for idx, (value, probability) in enumerate(zip(values, probabilities)):
        label = format(idx, f"0{n_qubits}b")
        ax.text(
            value,
            probability + 0.012,
            rf"$|{label}\rangle$",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel(r"Spot price at maturity $S_T$")
    ax.set_ylabel("Probability")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:.3f}" for value in values], fontsize=8)
    ax.set_ylim(0.0, 0.36)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stamatopoulos_fig3_reproduction.png", dpi=220)
    plt.close(fig)

    print("Saved results/stamatopoulos_fig3_reproduction.png")


if __name__ == "__main__":
    main()
