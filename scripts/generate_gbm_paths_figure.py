#!/usr/bin/env python3
"""Write results/gbm_sample_paths.png (GBM paths; Asian-style fixings exclude S_0)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MPLCONFIGDIR = _REPO_ROOT / "results" / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_option_pricing.classical.gbm import gbm_path

S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.2
T = 1.0
N_STEPS = 252
N_SHOW = 8
N_ENSEMBLE = 500

RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def _select_highlight_path(paths: np.ndarray) -> int:
    """Choose a visually useful path that finishes above strike when possible."""
    for idx, path in enumerate(paths):
        if path[-1] > K * 1.05:
            return idx
    return int(np.argmax(paths[:, -1]))


def main() -> None:
    rng = np.random.default_rng(21)  # seed chosen for visual variety + visible drift
    all_paths = gbm_path(S0, R, SIGMA, T, N_STEPS, N_ENSEMBLE, rng=rng)
    show_paths = all_paths[:N_SHOW]

    t_grid = np.linspace(0, T, N_STEPS + 1)
    ensemble_mean = all_paths.mean(axis=0)

    highlight_idx = _select_highlight_path(show_paths)
    highlighted_fixings = show_paths[highlight_idx, 1:]
    fixing_counts = np.arange(1, N_STEPS + 1)
    arith_running = np.cumsum(highlighted_fixings) / fixing_counts
    geo_running = np.exp(np.cumsum(np.log(highlighted_fixings)) / fixing_counts)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.8), constrained_layout=True)

    for i, path in enumerate(show_paths):
        is_highlighted = i == highlight_idx
        ax.plot(
            t_grid,
            path,
            color="#2563eb" if is_highlighted else "#64748b",
            alpha=0.82 if is_highlighted else 0.45,
            linewidth=1.8 if is_highlighted else 1.05,
            label="Highlighted path" if is_highlighted else ("Sample GBM paths" if i == 0 else None),
        )

    ax.plot(
        t_grid,
        ensemble_mean,
        color="#7c3aed",
        linewidth=2.0,
        linestyle="-.",
        alpha=0.85,
        label=f"Ensemble mean ({N_ENSEMBLE} paths)",
    )

    t_theory = np.linspace(0, T, 200)
    ax.plot(
        t_theory,
        S0 * np.exp(R * t_theory),
        color="#7c3aed",
        linewidth=1.2,
        linestyle=":",
        alpha=0.7,
        label=r"$S_0 e^{rt}$",
    )

    ax.plot(
        t_grid[1:],
        arith_running,
        color="#dc2626",
        linewidth=2.3,
        label=r"Running arithmetic avg. $\bar{A}(t)$",
    )
    ax.plot(
        t_grid[1:],
        geo_running,
        color="#16a34a",
        linewidth=2.3,
        linestyle="--",
        label=r"Running geometric avg. $\bar{G}(t)$",
    )
    ax.axhline(
        K,
        color="#111827",
        linewidth=1.0,
        linestyle=":",
        alpha=0.75,
        label=f"Strike $K={K:.0f}$",
    )

    ax.set_xlabel(r"Time $t$ (years)", fontsize=11)
    ax.set_ylabel(r"Stock price $S(t)$", fontsize=11)
    ax.set_title(
        "Risk-Neutral GBM Sample Paths\n"
        rf"$\mathrm{{d}}S = rS\,\mathrm{{d}}t + \sigma S\,\mathrm{{d}}W$; "
        rf"$S_0={S0:.0f}$, $r={R:.2f}$, $\sigma={SIGMA:.2f}$, $T={T:.0f}$",
        fontsize=12,
        pad=10,
    )
    ax.legend(fontsize=8.5, loc="upper left", frameon=True, ncols=2)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, T)
    ax.margins(x=0.0, y=0.08)

    out_path = RESULTS_DIR / "gbm_sample_paths.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved {out_path}")


if __name__ == "__main__":
    main()
