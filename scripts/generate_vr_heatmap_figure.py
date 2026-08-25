#!/usr/bin/env python3
"""Prospectively regenerate the Kemna--Vorst variance-reduction heatmap.

Each volatility/trial pair draws one seeded path panel and evaluates all five
strikes on those common random numbers.  The simulation is chunked, but its
sample moments are the same quantities used by the repository's in-memory
vanilla/Kemna--Vorst estimators.  Trial-level estimates, standard errors,
fitted coefficients, correlations, and seeds are retained as JSON and CSV.

Default outputs are canonical manuscript artifacts under ``results/``.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_option_pricing.classical.asian_mc import geometric_asian_call_exact

S0, R, T, N_STEPS = 100.0, 0.05, 1.0, 252
STRIKES = [90, 95, 100, 105, 110]
SIGMAS = [0.10, 0.15, 0.20, 0.30, 0.40]
DEFAULT_PATHS = 500_000
DEFAULT_TRIALS = 8
DEFAULT_CHUNK_SIZE = 20_000
DEFAULT_MASTER_SEED = 20_260_716


def _trial_statistics(
    sigma: float,
    n_paths: int,
    seed: int,
    chunk_size: int,
) -> list[dict[str, float]]:
    """Return sufficient-statistic estimates for all strikes on one path panel."""
    rng = np.random.default_rng(seed)
    n_strikes = len(STRIKES)
    sum_x = np.zeros(n_strikes)
    sum_y = np.zeros(n_strikes)
    sum_xx = np.zeros(n_strikes)
    sum_yy = np.zeros(n_strikes)
    sum_xy = np.zeros(n_strikes)
    dt = T / N_STEPS
    drift = (R - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    time_indices = np.arange(1, N_STEPS + 1)
    discount = math.exp(-R * T)
    strikes = np.asarray(STRIKES, dtype=float)

    completed = 0
    while completed < n_paths:
        batch = min(chunk_size, n_paths - completed)
        log_prices = rng.standard_normal((batch, N_STEPS))
        np.cumsum(log_prices, axis=1, out=log_prices)
        log_prices *= diffusion
        log_prices += math.log(S0) + drift * time_indices
        geometric_average = np.exp(log_prices.mean(axis=1))
        np.exp(log_prices, out=log_prices)
        arithmetic_average = log_prices.mean(axis=1)

        x = discount * np.maximum(arithmetic_average[:, None] - strikes, 0.0)
        y = discount * np.maximum(geometric_average[:, None] - strikes, 0.0)
        sum_x += x.sum(axis=0)
        sum_y += y.sum(axis=0)
        sum_xx += np.square(x).sum(axis=0)
        sum_yy += np.square(y).sum(axis=0)
        sum_xy += (x * y).sum(axis=0)
        completed += batch

    mean_x = sum_x / n_paths
    mean_y = sum_y / n_paths
    var_x = (sum_xx - n_paths * np.square(mean_x)) / (n_paths - 1)
    var_y = (sum_yy - n_paths * np.square(mean_y)) / (n_paths - 1)
    covariance = (sum_xy - n_paths * mean_x * mean_y) / (n_paths - 1)
    beta = np.divide(covariance, var_y, out=np.ones_like(covariance), where=var_y > 0)
    residual_variance = np.maximum(var_x + beta**2 * var_y - 2.0 * beta * covariance, 0.0)
    correlation = covariance / np.sqrt(var_x * var_y)

    rows: list[dict[str, float]] = []
    for index, strike in enumerate(STRIKES):
        geo_exact = geometric_asian_call_exact(S0, strike, R, sigma, T, N_STEPS)
        rows.append({
            "strike": float(strike),
            "vanilla_estimate": float(mean_x[index]),
            "vanilla_se": float(math.sqrt(var_x[index] / n_paths)),
            "cv_estimate": float(mean_x[index] + beta[index] * (geo_exact - mean_y[index])),
            "cv_se": float(math.sqrt(residual_variance[index] / n_paths)),
            "beta": float(beta[index]),
            "payoff_correlation": float(correlation[index]),
            "geometric_exact": float(geo_exact),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_REPO / "results")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--fast", action="store_true", help="40,000 paths and 3 trials")
    args = parser.parse_args(argv)
    if args.fast:
        args.paths, args.trials = 40_000, 3
    if min(args.paths, args.trials, args.chunk_size) < 2:
        parser.error("paths, trials, and chunk-size must all be at least 2")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"VR heatmap prospective run: M={args.paths:,}, trials={args.trials}, "
          f"master_seed={args.master_seed}", flush=True)
    trial_rows: list[dict] = []
    for sigma_index, sigma in enumerate(SIGMAS):
        for trial_index in range(args.trials):
            seed = args.master_seed + sigma_index * args.trials + trial_index
            for row in _trial_statistics(sigma, args.paths, seed, args.chunk_size):
                trial_rows.append({
                    "sigma": sigma,
                    "trial_index": trial_index,
                    "path_panel_seed": seed,
                    "n_paths": args.paths,
                    **row,
                })
            print(f"  sigma={sigma:.2f} trial={trial_index + 1}/{args.trials} seed={seed}",
                  flush=True)

    cell_rows: list[dict] = []
    vr = np.zeros((len(SIGMAS), len(STRIKES)))
    for sigma_index, sigma in enumerate(SIGMAS):
        for strike_index, strike in enumerate(STRIKES):
            selected = [row for row in trial_rows
                        if row["sigma"] == sigma and row["strike"] == float(strike)]
            mean_vanilla_se = float(np.mean([row["vanilla_se"] for row in selected]))
            mean_cv_se = float(np.mean([row["cv_se"] for row in selected]))
            ratio = (mean_vanilla_se / mean_cv_se) ** 2
            correlations = [row["payoff_correlation"] for row in selected]
            vr[sigma_index, strike_index] = ratio
            cell_rows.append({
                "sigma": sigma,
                "strike": float(strike),
                "n_paths_per_trial": args.paths,
                "n_trials": args.trials,
                "mean_vanilla_se": mean_vanilla_se,
                "mean_cv_se": mean_cv_se,
                "variance_reduction": ratio,
                "minimum_payoff_correlation": float(min(correlations)),
                "mean_payoff_correlation": float(np.mean(correlations)),
            })
        print(f"  sigma={sigma:.2f}: " + "  ".join(
            f"K{k}={vr[sigma_index, j]:,.0f}x" for j, k in enumerate(STRIKES)
        ), flush=True)

    payload = {
        "schema": "parikh-rayan-vr-heatmap-v1",
        "config": {
            "s0": S0,
            "rate": R,
            "maturity": T,
            "monitoring_dates": N_STEPS,
            "strikes": STRIKES,
            "volatilities": SIGMAS,
            "paths_per_trial": args.paths,
            "trials": args.trials,
            "chunk_size": args.chunk_size,
            "master_seed": args.master_seed,
            "seed_rule": "master_seed + sigma_index * trials + trial_index",
            "common_random_numbers": "all strikes share each sigma/trial path panel",
            "numpy_version": np.__version__,
            "scipy_version": importlib.metadata.version("scipy"),
        },
        "cells": cell_rows,
        "trials": trial_rows,
    }
    (args.out / "vr_heatmap_data.json").write_text(json.dumps(payload, indent=1))
    _write_csv(args.out / "vr_heatmap_cells.csv", cell_rows)
    _write_csv(args.out / "vr_heatmap_trials.csv", trial_rows)

    minimum_correlation = min(row["minimum_payoff_correlation"] for row in cell_rows)
    print(f"VR range: {vr.min():,.0f}x -- {vr.max():,.0f}x; "
          f"minimum trial correlation={minimum_correlation:.6f}", flush=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    image = ax.imshow(
        vr,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=max(1, vr.min()), vmax=vr.max()),
    )
    ax.set_xticks(range(len(STRIKES)), [str(k) for k in STRIKES])
    ax.set_yticks(range(len(SIGMAS)), [f"{s:.2f}" for s in SIGMAS])
    ax.set_xlabel("strike $K$")
    ax.set_ylabel(r"volatility $\sigma$")
    ax.set_title("Kemna--Vorst variance-reduction factor  Var(vanilla)/Var(CV)")
    for i in range(len(SIGMAS)):
        for j in range(len(STRIKES)):
            ax.text(j, i, f"{vr[i, j]:,.0f}", ha="center", va="center",
                    color="white" if vr[i, j] < vr.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="VR factor (log scale)")
    fig.tight_layout()
    figure_path = args.out / "vr_heatmap.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> wrote {figure_path}, vr_heatmap_data.json, and CSV tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
