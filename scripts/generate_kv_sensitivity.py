#!/usr/bin/env python3
"""KV sensitivity: results/kv_sensitivity_table.txt and kv_sensitivity.png (--fast → results/fast/)."""

from __future__ import annotations

import argparse
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

from qc_option_pricing.classical import (
    arithmetic_asian_vanilla_cv_shared,
    asian_kv_payoff_correlation,
)

S0 = 100.0
R = 0.05
T = 1.0
N_STEPS = 252

SIGMAS_FULL = [0.15, 0.2, 0.25, 0.3, 0.4]
K_RATIOS_FULL = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]

BASE_RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR = BASE_RESULTS_DIR
RESULTS_DIR.mkdir(exist_ok=True)


def run(*, fast: bool) -> None:
    global RESULTS_DIR
    if fast:
        RESULTS_DIR = BASE_RESULTS_DIR / "fast"
        RESULTS_DIR.mkdir(exist_ok=True)
        sigmas = [0.2, 0.3, 0.4]
        k_ratios = [0.9, 1.0, 1.1]
        n_paths = 25_000
        n_trials_vr = 6
        n_rho_draws = 3
    else:
        sigmas = SIGMAS_FULL
        k_ratios = K_RATIOS_FULL
        n_paths = 50_000
        n_trials_vr = 14
        n_rho_draws = 5

    rows: list[tuple[float, float, float, float, float, float, float]] = []

    print("KV sensitivity sweep (shared-path vanilla vs CV)")
    print(f"  M={n_paths:,} paths, VR averaged over {n_trials_vr} trials")
    print(f"  ρ = median Pearson corr over {n_rho_draws} panels\n")

    for sigma in sigmas:
        for kr in k_ratios:
            k = kr * S0
            v_ses, cv_ses = [], []
            for tr in range(n_trials_vr):
                seed = (
                    10_007
                    + tr
                    + int(sigma * 10_000) % 500_000
                    + int(round(kr * 1_000)) * 97
                )
                rng = np.random.default_rng(seed)
                (_, v_se), (_, cv_se) = arithmetic_asian_vanilla_cv_shared(
                    S0, k, R, sigma, T, N_STEPS, n_paths, rng=rng
                )
                v_ses.append(v_se)
                cv_ses.append(cv_se)
            mean_v = float(np.mean(v_ses))
            mean_cv = float(np.mean(cv_ses))
            vr = (mean_v / mean_cv) ** 2 if mean_cv > 0 else float("nan")

            rhos = []
            for j in range(n_rho_draws):
                rng_r = np.random.default_rng(90_001 + j + int(1e4 * sigma) + int(1e3 * kr))
                rho = asian_kv_payoff_correlation(
                    S0, k, R, sigma, T, N_STEPS, n_paths, rng=rng_r
                )
                if not np.isnan(rho):
                    rhos.append(rho)
            rho_med = float(np.median(rhos)) if rhos else float("nan")

            rows.append((sigma, k, kr, mean_v, mean_cv, vr, rho_med))
            print(f"  σ={sigma:.2f}  K/S0={kr:.2f}  VR={vr:,.0f}x  ρ={rho_med:.4f}")

    table_path = RESULTS_DIR / "kv_sensitivity_table.txt"
    with open(table_path, "w") as f:
        f.write(
            "Kemna–Vorst sensitivity (arithmetic Asian call, geometric control).\n"
            f"S0={S0}, T={T}, r={R}, N_steps={N_STEPS}, M={n_paths:,}\n"
            f"VR = (mean vanilla stderr / mean KV stderr)^2 over {n_trials_vr} trials.\n"
            f"rho = median Pearson corr(discounted arith payoff, disc. geo payoff) "
            f"over {n_rho_draws} path panels.\n\n"
        )
        hdr = (
            f"{'sigma':>6}  {'K':>8}  {'K/S0':>6}  "
            f"{'mean_v_se':>10}  {'mean_cv_se':>11}  {'VR':>10}  {'rho':>8}\n"
        )
        f.write(hdr)
        f.write("-" * len(hdr.rstrip()) + "\n")
        for sigma, k, kr, mean_v, mean_cv, vr, rho_med in rows:
            f.write(
                f"{sigma:>6.2f}  {k:>8.1f}  {kr:>6.2f}  "
                f"{mean_v:>10.6f}  {mean_cv:>11.6f}  {vr:>10.1f}  {rho_med:>8.4f}\n"
            )
    print(f"\n→ Wrote {table_path}")

    by_sigma: dict[float, list[tuple[float, float, float]]] = {s: [] for s in sigmas}
    for sigma, k, kr, mean_v, mean_cv, vr, rho_med in rows:
        by_sigma[sigma].append((kr, vr, rho_med))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(sigmas)))

    for i, sigma in enumerate(sigmas):
        pts = sorted(by_sigma[sigma], key=lambda x: x[0])
        xs = [p[0] for p in pts]
        vrs = [p[1] for p in pts]
        rh = [p[2] for p in pts]
        label = f"σ = {sigma:.2f}"
        ax0.plot(xs, vrs, "o-", color=cmap[i], label=label, markersize=5)
        ax1.plot(xs, rh, "s-", color=cmap[i], label=label, markersize=5)

    ax0.set_xlabel("Moneyness  $K/S_0$", fontsize=11)
    ax0.set_ylabel("Variance reduction factor  $(\\bar{s}_v/\\bar{s}_{cv})^2$", fontsize=11)
    ax0.set_title("KV: VR vs strike (by volatility)", fontsize=12)
    ax0.set_yscale("log")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8, title="Volatility")

    ax1.set_xlabel("Moneyness  $K/S_0$", fontsize=11)
    ax1.set_ylabel(r"Payoff correlation  $\rho$", fontsize=11)
    ax1.set_title(
        "Arithmetic vs geometric discounted payoff\n(y-axis zoomed near 1)",
        fontsize=11,
    )
    # Tight ylim so curves are distinguishable — values typically sit in [0.99, 1.0]
    rho_vals = [p[2] for pts in by_sigma.values() for p in pts if not np.isnan(p[2])]
    rho_min = max(0.99, float(np.min(rho_vals)) - 0.001) if rho_vals else 0.99
    ax1.set_ylim(rho_min, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, title="Volatility")

    fig.suptitle(
        "Kemna–Vorst control variate: sensitivity to moneyness and volatility",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig_path = RESULTS_DIR / "kv_sensitivity.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Wrote {fig_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="Smaller grid and fewer trials")
    args = ap.parse_args()
    run(fast=args.fast)
