#!/usr/bin/env python3
"""Generate classical MC figures/tables under results/. Use --fast for smaller grids (writes results/fast/)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
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
    european_call,
    european_call_mc,
    arithmetic_asian_vanilla_cv_shared,
    geometric_asian_call_exact,
    geometric_asian_call_mc,
)

S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.2
T = 1.0
N_STEPS = 252
N_TRIALS_EUR = 50
N_TRIALS_ASIAN = 30

EUR_PATH_COUNTS = [
    500, 1_000, 2_000, 5_000, 10_000, 20_000,
    50_000, 100_000, 200_000, 500_000,
]
ASIAN_PATH_COUNTS = [
    1_000, 2_000, 5_000, 10_000, 20_000,
    50_000, 100_000, 200_000,
]
GEO_CROSSCHECK_RUNS = 20
GEO_CROSSCHECK_PATHS = 100_000

BASE_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR = BASE_RESULTS_DIR
RESULTS_DIR.mkdir(exist_ok=True)


def apply_fast_preset() -> None:
    """Shrink grids and trials for quick iteration (overwrites module-level settings)."""
    global RESULTS_DIR
    global N_TRIALS_EUR, N_TRIALS_ASIAN, EUR_PATH_COUNTS, ASIAN_PATH_COUNTS
    global GEO_CROSSCHECK_RUNS, GEO_CROSSCHECK_PATHS
    RESULTS_DIR = BASE_RESULTS_DIR / "fast"
    RESULTS_DIR.mkdir(exist_ok=True)
    N_TRIALS_EUR = 8
    N_TRIALS_ASIAN = 4
    EUR_PATH_COUNTS = [500, 2_000, 10_000, 50_000]
    ASIAN_PATH_COUNTS = [1_000, 5_000, 20_000, 50_000]
    GEO_CROSSCHECK_RUNS = 5
    GEO_CROSSCHECK_PATHS = 25_000


def crosscheck_geometric():
    print("=" * 60)
    print("GEOMETRIC ASIAN — CLOSED FORM vs MC CROSS-CHECK")
    print("=" * 60)

    geo_exact = geometric_asian_call_exact(S0, K, R, SIGMA, T, N_STEPS)
    print(f"  Closed-form price: {geo_exact:.6f}")

    mc_prices = []
    for seed in range(GEO_CROSSCHECK_RUNS):
        rng = np.random.default_rng(seed)
        price, se = geometric_asian_call_mc(
            S0, K, R, SIGMA, T, N_STEPS, GEO_CROSSCHECK_PATHS, rng=rng
        )
        mc_prices.append(price)
    mc_mean = np.mean(mc_prices)
    mc_std = np.std(mc_prices, ddof=1)
    bias = mc_mean - geo_exact
    print(f"  MC mean ({GEO_CROSSCHECK_RUNS} runs × {GEO_CROSSCHECK_PATHS:,} paths): "
          f"{mc_mean:.6f}")
    print(f"  MC std across runs:              {mc_std:.6f}")
    print(f"  Bias (MC - exact):               {bias:.6f}")
    n_r = GEO_CROSSCHECK_RUNS
    ok = abs(bias) < 3 * mc_std / math.sqrt(n_r) if n_r > 1 else True
    print(f"  Within 3-sigma? {'YES' if ok else 'NO — check formula!'}\n")

    with open(RESULTS_DIR / "geo_crosscheck.txt", "w") as f:
        f.write(f"Geometric Asian Call Cross-Check\n")
        f.write(f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, "
                f"T={T}, N={N_STEPS}\n\n")
        f.write(f"Closed-form price:  {geo_exact:.6f}\n")
        f.write(f"MC mean ({n_r}×{GEO_CROSSCHECK_PATHS}):  {mc_mean:.6f}\n")
        f.write(f"MC std across runs: {mc_std:.6f}\n")
        f.write(f"Bias:               {bias:.6f}\n")
        f.write(f"Within 3-sigma:     {'YES' if ok else 'NO'}\n")
    print(f"  → Saved geo_crosscheck.txt\n")
    return geo_exact


def run_european():
    print("=" * 60)
    print("EUROPEAN CALL — MC CONVERGENCE vs BLACK-SCHOLES")
    print("=" * 60)

    analytic = european_call(S0, K, R, SIGMA, T)
    print(f"Black-Scholes analytic price: {analytic:.6f}\n")

    path_counts = EUR_PATH_COUNTS

    mean_errors, mean_stderrs = [], []
    q25_errors, q75_errors = [], []

    for n in path_counts:
        trial_errors, trial_stderrs = [], []
        for trial in range(N_TRIALS_EUR):
            rng = np.random.default_rng(trial * 10_000 + n)
            res = european_call_mc(S0, K, R, SIGMA, T, n, rng=rng)
            trial_errors.append(abs(res.estimate - analytic))
            trial_stderrs.append(res.stderr)
        mean_errors.append(np.mean(trial_errors))
        mean_stderrs.append(np.mean(trial_stderrs))
        q25_errors.append(np.percentile(trial_errors, 25))
        q75_errors.append(np.percentile(trial_errors, 75))
        print(f"  M={n:>10,d}  |  mean|err|={mean_errors[-1]:.5f}  "
              f"  mean_stderr={mean_stderrs[-1]:.5f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.fill_between(path_counts, q25_errors, q75_errors,
                    color="#dc2626", alpha=0.12, label="25th–75th pctile |error|")
    ax.loglog(path_counts, mean_errors, "s-", color="#dc2626",
              label=f"Mean |MC − BS|  ({N_TRIALS_EUR} trials)", markersize=5)
    ax.loglog(path_counts, mean_stderrs, "o-", color="#2563eb",
              label="MC standard error (CLT)", markersize=5)

    ref_x = np.array(path_counts, dtype=float)
    ref_y = mean_stderrs[0] * np.sqrt(path_counts[0]) / np.sqrt(ref_x)
    ax.loglog(ref_x, ref_y, ":", color="gray", linewidth=1.5,
              label=r"$O(1/\sqrt{M})$ reference")

    ax.set_xlabel("Number of MC paths  $M$", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("European Call — MC Convergence to Black-Scholes", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "european_convergence.png", dpi=200)
    plt.close(fig)
    print(f"\n  → Saved european_convergence.png")

    with open(RESULTS_DIR / "european_table.txt", "w") as f:
        f.write(f"European Call Parameters: S0={S0}, K={K}, r={R}, "
                f"sigma={SIGMA}, T={T}\n")
        f.write(f"Black-Scholes analytic price: {analytic:.6f}\n")
        f.write(f"Each row: mean over {N_TRIALS_EUR} independent trials\n\n")
        f.write(f"{'Paths':>12s}  {'Mean Stderr':>12s}  "
                f"{'Mean |Error|':>12s}  {'Q25 |Err|':>10s}  {'Q75 |Err|':>10s}\n")
        f.write("-" * 64 + "\n")
        for i, n in enumerate(path_counts):
            f.write(f"{n:>12,d}  {mean_stderrs[i]:>12.5f}  "
                    f"{mean_errors[i]:>12.5f}  {q25_errors[i]:>10.5f}  "
                    f"{q75_errors[i]:>10.5f}\n")
    print(f"  → Saved european_table.txt\n")


def run_asian():
    print("=" * 60)
    print("ARITHMETIC ASIAN CALL — METHOD COMPARISON (shared paths)")
    print("=" * 60)

    geo_exact = geometric_asian_call_exact(S0, K, R, SIGMA, T, N_STEPS)
    print(f"Geometric Asian exact price:  {geo_exact:.6f}\n")

    path_counts = ASIAN_PATH_COUNTS
    method_names = ["Vanilla MC", "Control Variate (KV)"]
    results = {name: {"mean_est": [], "mean_se": [], "mean_time": []} for name in method_names}

    for n in path_counts:
        trial_v_est, trial_v_se = [], []
        trial_cv_est, trial_cv_se = [], []
        trial_times = []
        for trial in range(N_TRIALS_ASIAN):
            rng = np.random.default_rng(trial * 10_000 + n)
            t0 = time.perf_counter()
            (v_est, v_se), (cv_est, cv_se) = arithmetic_asian_vanilla_cv_shared(
                S0, K, R, SIGMA, T, N_STEPS, n, rng=rng
            )
            trial_times.append(time.perf_counter() - t0)
            trial_v_est.append(v_est)
            trial_v_se.append(v_se)
            trial_cv_est.append(cv_est)
            trial_cv_se.append(cv_se)

        results["Vanilla MC"]["mean_est"].append(np.mean(trial_v_est))
        results["Vanilla MC"]["mean_se"].append(np.mean(trial_v_se))
        results["Vanilla MC"]["mean_time"].append(np.mean(trial_times))

        results["Control Variate (KV)"]["mean_est"].append(np.mean(trial_cv_est))
        results["Control Variate (KV)"]["mean_se"].append(np.mean(trial_cv_se))
        results["Control Variate (KV)"]["mean_time"].append(np.mean(trial_times))

        print(f"\n  M={n:>10,d}  (one GBM panel / trial, vanilla + KV)")
        print(f"    Vanilla MC           |  price={results['Vanilla MC']['mean_est'][-1]:8.4f}  |  "
              f"stderr={results['Vanilla MC']['mean_se'][-1]:.6f}  |  "
              f"{results['Vanilla MC']['mean_time'][-1]:.3f}s")
        print(f"    Control Variate (KV) |  price={results['Control Variate (KV)']['mean_est'][-1]:8.4f}  |  "
              f"stderr={results['Control Variate (KV)']['mean_se'][-1]:.6f}  |  "
              f"{results['Control Variate (KV)']['mean_time'][-1]:.3f}s")

    colors = {"Vanilla MC": "#dc2626", "Control Variate (KV)": "#2563eb"}
    markers = {"Vanilla MC": "o", "Control Variate (KV)": "s"}

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name in method_names:
        ax.loglog(path_counts, results[name]["mean_se"], f"{markers[name]}-",
                  label=name, color=colors[name], markersize=6)

    ref_x = np.array(path_counts, dtype=float)
    ref_y = (results["Vanilla MC"]["mean_se"][0]
             * np.sqrt(path_counts[0]) / np.sqrt(ref_x))
    ax.loglog(ref_x, ref_y, ":", color="gray", alpha=0.5,
              label=r"$O(1/\sqrt{M})$")

    ax.set_xlabel("Number of MC paths  $M$", fontsize=12)
    ax.set_ylabel("Standard error of price estimate", fontsize=12)
    ax.set_title("Arithmetic Asian Call — Variance Reduction Comparison", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "asian_methods_comparison.png", dpi=200)
    plt.close(fig)
    print(f"\n  → Saved asian_methods_comparison.png")

    try:
        idx_100k = path_counts.index(100_000)
    except ValueError:
        idx_100k = len(path_counts) - 1
    vanilla_var = results["Vanilla MC"]["mean_se"][idx_100k] ** 2

    m_ref = path_counts[idx_100k]
    print(f"\n{'='*60}")
    print(f"VARIANCE REDUCTION FACTORS (at M = {m_ref:,}, avg over "
          f"{N_TRIALS_ASIAN} trials)")
    print("=" * 60)

    with open(RESULTS_DIR / "variance_reduction_summary.txt", "w") as f:
        f.write(f"Asian Call Parameters: S0={S0}, K={K}, r={R}, "
                f"sigma={SIGMA}, T={T}, N_steps={N_STEPS}\n")
        f.write(f"Geometric Asian exact price: {geo_exact:.6f}\n")
        f.write(f"Averages over {N_TRIALS_ASIAN} independent trials per point\n\n")
        f.write(f"Variance reduction factors at M = {m_ref:,}:\n")
        f.write(f"(factor = vanilla_variance / method_variance)\n\n")
        for name in method_names:
            method_var = results[name]["mean_se"][idx_100k] ** 2
            factor = vanilla_var / method_var if method_var > 0 else float("inf")
            line = (f"  {name:<25s}: stderr={results[name]['mean_se'][idx_100k]:.6f}"
                    f"  VR factor={factor:8.1f}x")
            print(line)
            f.write(line + "\n")

        cv_se = results["Control Variate (KV)"]["mean_se"][idx_100k]
        van_se = results["Vanilla MC"]["mean_se"][idx_100k]
        effective_M = (van_se / cv_se) ** 2 * m_ref
        f.write(f"\nEffective sample equivalence:\n")
        line = f"  CV at {m_ref:,} paths ≈ Vanilla MC at {effective_M:,.0f} paths"
        print(line)
        f.write(line + "\n")

        note = (
            "Both columns use the same simulated paths per trial (common random numbers).\n"
            "Wall-clock time is one GBM run plus both estimators (reported for each label)."
        )
        print(note)
        f.write(f"\n{note}\n")

    print(f"\n  → Saved variance_reduction_summary.txt")

    with open(RESULTS_DIR / "asian_table.txt", "w") as f:
        f.write(f"Asian Call Parameters: S0={S0}, K={K}, r={R}, "
                f"sigma={SIGMA}, T={T}, N_steps={N_STEPS}\n")
        f.write(f"Geometric Asian exact: {geo_exact:.6f}\n")
        f.write(f"Averages over {N_TRIALS_ASIAN} independent trials per point\n\n")
        for name in method_names:
            f.write(f"\n{name}:\n")
            f.write(f"{'Paths':>12s}  {'Price':>10s}  {'Stderr':>12s}  "
                    f"{'Time(s)':>8s}\n")
            f.write("-" * 48 + "\n")
            for i, n in enumerate(path_counts):
                f.write(f"{n:>12,d}  {results[name]['mean_est'][i]:>10.4f}  "
                        f"{results[name]['mean_se'][i]:>12.6f}  "
                        f"{results[name]['mean_time'][i]:>8.3f}\n")
    print(f"  → Saved asian_table.txt\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classical MC results and plots.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fewer trials and smaller path grids for quick iteration.",
    )
    args = parser.parse_args()
    if args.fast:
        apply_fast_preset()
        print("*** FAST MODE: smaller grids / fewer trials (not for final paper) ***\n")

    print(f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}")
    print(f"Asian monitoring: {N_STEPS} steps (daily)")
    print(f"Trials: European={N_TRIALS_EUR}, Asian={N_TRIALS_ASIAN}\n")
    crosscheck_geometric()
    run_european()
    run_asian()
    print(f"Done! All results in {RESULTS_DIR}/")
