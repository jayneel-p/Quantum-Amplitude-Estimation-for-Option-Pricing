#!/usr/bin/env python3
"""Generate two diagnostics for the range-versus-variance Principle.

The first figure compares the variance and exact finite-grid oscillation of

    R_beta = H - beta C,

for the arithmetic Asian payoff H and geometric Asian control C on the paper's
12-date, four-point conditional-mean encoding grid.  Both moments and extrema
are exact for that finite grid; they are not presented as continuous-model
bounds.

The second figure combines the existing 25-cell variance and normalisation
experiments.  Both axes are reconstructed at beta=1.  The vertical axis uses
the manuscript's matched-exceedance A_N-B_1 proxy, not a theorem-faithful
executed amplitude-estimation query count.

Outputs:
  results/principle_beta_sweep.png
  results/principle_grid_scatter.png
  results/principle_beta_sweep.csv
  results/principle_grid_scatter.csv
  results/principle_diagnostics.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


S0 = 100.0
STRIKE = 100.0
RATE = 0.05
SIGMA = 0.20
MATURITY = 1.0
DEFAULT_GRID_DATES = 12
DEFAULT_GRID_CHUNK_SIZE = 1 << 20
DEFAULT_BETA_MIN = 0.85
DEFAULT_BETA_MAX = 1.15
DEFAULT_BETA_POINTS = 121

VR_INPUT = _REPO / "results" / "vr_heatmap_data.json"
RANGE_INPUT = _REPO / "results" / "v4" / "linf_heatmap.txt"
SECTION6_INPUT = _REPO / "paper" / "ParikhRayan_section6_replacement.tex"
EXACT12_INPUT = _REPO / "results" / "v4" / "asian_exact12.txt"
GENERATOR_INPUT = Path(__file__).resolve()
VR_GENERATOR_INPUT = _REPO / "scripts" / "generate_vr_heatmap_figure.py"
RANGE_GENERATOR_INPUT = _REPO / "scripts" / "v4_linf_heatmap.py"
EXACT12_GENERATOR_INPUT = _REPO / "scripts" / "v4_asian_exact12.py"

BLUE = "#2f6f9f"
ORANGE = "#c05a28"
GREY = "#555555"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    """Use a repository-relative label when possible, otherwise an absolute path."""

    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normal_bin_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the four conditional-mean normal shocks used by the encoding."""

    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    probabilities = np.diff(norm.cdf(edges))
    shocks = np.array(
        [
            (norm.pdf(lower) - norm.pdf(upper)) / probability
            for lower, upper, probability in zip(
                edges[:-1], edges[1:], probabilities
            )
        ]
    )
    return shocks, probabilities


SHOCKS, SHOCK_PROBABILITIES = _normal_bin_grid()
LOG_SHOCK_PROBABILITIES = np.log(SHOCK_PROBABILITIES)


def _exact_grid_payoffs(
    n_dates: int, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate probabilities and undiscounted payoffs on the encoded grid."""

    n_states = 4**n_dates
    target = np.empty(n_states, dtype=np.float64)
    control = np.empty(n_states, dtype=np.float64)
    probabilities = np.empty(n_states, dtype=np.float64)
    dt = MATURITY / n_dates
    drift = (RATE - 0.5 * SIGMA**2) * dt
    volatility = SIGMA * math.sqrt(dt)

    for lower in range(0, n_states, chunk_size):
        upper = min(lower + chunk_size, n_states)
        indices = np.arange(lower, upper, dtype=np.int64)
        count = upper - lower
        cumulative_logs = np.empty((n_dates, count), dtype=np.float64)
        cumulative_log_price = np.full(count, math.log(S0))
        log_probability = np.zeros(count)
        for date in range(n_dates):
            digit = (indices >> (2 * date)) & 3
            log_probability += LOG_SHOCK_PROBABILITIES[digit]
            cumulative_log_price += drift + volatility * SHOCKS[digit]
            cumulative_logs[date] = cumulative_log_price

        arithmetic_average = np.exp(cumulative_logs).mean(axis=0)
        geometric_average = np.exp(cumulative_logs.mean(axis=0))
        target[lower:upper] = np.maximum(arithmetic_average - STRIKE, 0.0)
        control[lower:upper] = np.maximum(geometric_average - STRIKE, 0.0)
        probabilities[lower:upper] = np.exp(log_probability)

    return target, control, probabilities


def _beta_sweep(
    target: np.ndarray,
    control: np.ndarray,
    probabilities: np.ndarray,
    beta_min: float,
    beta_max: float,
    beta_points: int,
) -> tuple[list[dict], dict, list[dict]]:
    """Evaluate exact probability-weighted variance and grid oscillation."""

    probability_sum = float(probabilities.sum())
    weights = probabilities / probability_sum
    target_mean = float(weights @ target)
    control_mean = float(weights @ control)
    target_centered = target - target_mean
    control_centered = control - control_mean
    var_target = float(weights @ (target_centered * target_centered))
    var_control = float(weights @ (control_centered * control_centered))
    cov_target_control = float(weights @ (target_centered * control_centered))
    if var_control <= 0:
        raise RuntimeError("control variance is nonpositive")
    beta_variance = cov_target_control / var_control

    def oscillation(beta: float) -> float:
        residual = target - beta * control
        return float(residual.max() - residual.min())

    range_result = minimize_scalar(
        oscillation,
        bounds=(beta_min, beta_max),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )
    if not range_result.success:
        raise RuntimeError(f"range optimisation failed: {range_result.message}")
    beta_range = float(range_result.x)

    positive_control = control > 0
    beta_nonnegative_limit = float(
        np.min(target[positive_control] / control[positive_control])
    )
    betas = np.linspace(beta_min, beta_max, beta_points)
    rows: list[dict] = []
    for beta in betas:
        residual = target - beta * control
        variance = (
            var_target
            + beta * beta * var_control
            - 2.0 * beta * cov_target_control
        )
        lower = float(residual.min())
        upper = float(residual.max())
        rows.append(
            {
                "beta": float(beta),
                "variance_dollars_squared": float(variance),
                "grid_min_dollars": lower,
                "grid_max_dollars": upper,
                "grid_oscillation_dollars": upper - lower,
                "signed_on_grid": bool(lower < -1e-12),
            }
        )

    def summary_at(beta: float) -> dict:
        residual = target - beta * control
        residual_mean = float(weights @ residual)
        return {
            "beta": beta,
            "variance_dollars_squared": float(
                weights @ ((residual - residual_mean) ** 2)
            ),
            "grid_min_dollars": float(residual.min()),
            "grid_max_dollars": float(residual.max()),
            "grid_oscillation_dollars": float(
                residual.max() - residual.min()
            ),
        }

    beta_one_residual = target - control
    range_summary = summary_at(beta_range)
    gates = [
        {
            "name": "finite-grid probabilities normalise",
            "passed": bool(abs(probability_sum - 1.0) < 1e-12),
            "observed": probability_sum,
            "tolerance": 1e-12,
        },
        {
            "name": "AM-GM residual is nonnegative at beta=1",
            "passed": bool(beta_one_residual.min() >= -1e-10),
            "observed": float(beta_one_residual.min()),
            "tolerance": -1e-10,
        },
        {
            "name": "variance optimum lies in the declared sweep",
            "passed": bool(beta_min < beta_variance < beta_max),
            "observed": beta_variance,
            "interval": [beta_min, beta_max],
        },
        {
            "name": "range optimum lies in the declared sweep",
            "passed": bool(beta_min < beta_range < beta_max),
            "observed": beta_range,
            "interval": [beta_min, beta_max],
        },
        {
            "name": "the two optima are materially separated",
            "passed": bool(abs(beta_range - beta_variance) > 0.01),
            "observed": abs(beta_range - beta_variance),
            "threshold": 0.01,
        },
        {
            "name": "range-optimal residual is signed on the finite grid",
            "passed": bool(range_summary["grid_min_dollars"] < 0),
            "observed": range_summary["grid_min_dollars"],
            "threshold": 0.0,
        },
        {
            "name": "bounded optimiser is no worse than the sampled range curve",
            "passed": bool(
                range_summary["grid_oscillation_dollars"]
                <= min(row["grid_oscillation_dollars"] for row in rows) + 1e-9
            ),
            "observed": range_summary["grid_oscillation_dollars"],
            "sampled_minimum": min(
                row["grid_oscillation_dollars"] for row in rows
            ),
            "tolerance": 1e-9,
        },
    ]
    summary = {
        "beta_variance_optimal": beta_variance,
        "beta_range_optimal": beta_range,
        "beta_nonnegative_limit_on_grid": beta_nonnegative_limit,
        "grid_probability_sum": probability_sum,
        "discounted_target_price": math.exp(-RATE * MATURITY) * target_mean,
        "grid_target_max_dollars": float(target.max()),
        "beta_one": summary_at(1.0),
        "variance_optimum": summary_at(beta_variance),
        "range_optimum": range_summary,
    }
    return rows, summary, gates


def _parse_range_matrix(path: Path) -> tuple[list[float], list[float], dict[tuple[float, float], float]]:
    """Parse the manuscript's 5x5 matched-exceedance proxy table."""

    lines = path.read_text().splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("sigma\\K"))
    strikes = [float(item) for item in lines[header_index].split()[1:]]
    matrix: dict[tuple[float, float], float] = {}
    sigmas: list[float] = []
    for line in lines[header_index + 1 : header_index + 6]:
        fields = line.split()
        sigma = float(fields[0])
        values = [float(item) for item in fields[1:]]
        if len(values) != len(strikes):
            raise ValueError(f"malformed range row: {line}")
        sigmas.append(sigma)
        matrix.update({(sigma, strike): value for strike, value in zip(strikes, values)})
    return sigmas, strikes, matrix


def _grid_scatter() -> tuple[list[dict], list[dict]]:
    """Reconstruct beta=1 standard-deviation gains and join range proxies."""

    payload = json.loads(VR_INPUT.read_text())
    sigmas, strikes, range_ratio = _parse_range_matrix(RANGE_INPUT)
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    consistency_errors: list[float] = []

    for row in payload["trials"]:
        n_paths = int(row["n_paths"])
        var_target = float(row["vanilla_se"]) ** 2 * n_paths
        beta_star = float(row["beta"])
        correlation = float(row["payoff_correlation"])
        if beta_star <= 0 or not 0 <= correlation <= 1:
            raise ValueError("stored trial moments are outside the expected domain")

        var_control = correlation**2 * var_target / beta_star**2
        covariance = beta_star * var_control
        var_beta_one = max(
            var_target + var_control - 2.0 * covariance,
            0.0,
        )
        beta_one_se = math.sqrt(var_beta_one / n_paths)

        reconstructed_optimal = max(
            var_target + beta_star**2 * var_control - 2.0 * beta_star * covariance,
            0.0,
        )
        stored_optimal = float(row["cv_se"]) ** 2 * n_paths
        denominator = max(abs(stored_optimal), 1e-30)
        consistency_errors.append(abs(reconstructed_optimal - stored_optimal) / denominator)
        grouped[(float(row["sigma"]), float(row["strike"]))].append(
            {
                "vanilla_se": float(row["vanilla_se"]),
                "beta_one_residual_se": beta_one_se,
            }
        )

    rows: list[dict] = []
    for sigma in sigmas:
        for strike in strikes:
            trials = grouped[(sigma, strike)]
            if not trials:
                raise ValueError(f"missing variance trials for sigma={sigma}, K={strike}")
            mean_raw_se = float(np.mean([item["vanilla_se"] for item in trials]))
            mean_residual_se = float(
                np.mean([item["beta_one_residual_se"] for item in trials])
            )
            std_ratio = mean_raw_se / mean_residual_se
            ratio = range_ratio[(sigma, strike)]
            rows.append(
                {
                    "sigma": sigma,
                    "strike": strike,
                    "beta": 1.0,
                    "standard_deviation_ratio_beta_one": std_ratio,
                    "variance_ratio_beta_one": std_ratio**2,
                    "matched_exceedance_range_ratio_proxy": ratio,
                    "std_ratio_over_range_ratio": std_ratio / ratio,
                    "n_trials": len(trials),
                    "paths_per_trial": int(payload["config"]["paths_per_trial"]),
                }
            )

    joined = {(row["sigma"], row["strike"]) for row in rows}
    expected = {(sigma, strike) for sigma in sigmas for strike in strikes}
    gates = [
        {
            "name": "all 25 contract cells joined",
            "passed": joined == expected and len(rows) == 25,
            "observed": len(rows),
            "expected": 25,
        },
        {
            "name": "stored beta-optimal SE reconstructs from sufficient statistics",
            "passed": bool(max(consistency_errors) < 1e-8),
            "observed_max_relative_error": max(consistency_errors),
            "tolerance": 1e-8,
        },
        {
            "name": "all ratios are finite and positive",
            "passed": bool(
                all(
                    math.isfinite(row["standard_deviation_ratio_beta_one"])
                    and row["standard_deviation_ratio_beta_one"] > 0
                    and math.isfinite(row["matched_exceedance_range_ratio_proxy"])
                    and row["matched_exceedance_range_ratio_proxy"] > 0
                    for row in rows
                )
            ),
        },
    ]
    return rows, gates


def _plot_beta_sweep(
    rows: list[dict],
    summary: dict,
    path: Path,
    grid_dates: int,
    grid_states: int,
) -> None:
    betas = np.asarray([row["beta"] for row in rows])
    variances = np.asarray([row["variance_dollars_squared"] for row in rows])
    ranges = np.asarray([row["grid_oscillation_dollars"] for row in rows])
    beta_variance = summary["beta_variance_optimal"]
    beta_range = summary["beta_range_optimal"]
    beta_signed = summary["beta_nonnegative_limit_on_grid"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.25), sharex=True)

    for axis in axes:
        axis.axvspan(beta_signed, betas.max(), color="0.90", alpha=0.65, zorder=0)
        axis.axvline(1.0, color=GREY, linestyle=":", linewidth=1.2, label=r"$\beta=1$")
        axis.axvline(
            beta_variance,
            color=BLUE,
            linestyle="--",
            linewidth=1.35,
            label=rf"$\beta^*={beta_variance:.3f}$",
        )
        axis.axvline(
            beta_range,
            color=ORANGE,
            linestyle="-.",
            linewidth=1.35,
            label=rf"$\beta_\infty={beta_range:.3f}$",
        )
        axis.set_xlim(betas.min(), betas.max())
        axis.set_xlabel(r"control coefficient $\beta$")
        axis.grid(axis="y", color="0.86", linewidth=0.6)

    axes[0].plot(betas, variances, color=BLUE, linewidth=2.0)
    axes[0].scatter(
        [beta_variance],
        [summary["variance_optimum"]["variance_dollars_squared"]],
        color=BLUE,
        marker="o",
        s=42,
        zorder=4,
    )
    axes[0].scatter(
        [beta_range],
        [summary["range_optimum"]["variance_dollars_squared"]],
        facecolor="none",
        edgecolor=ORANGE,
        marker="s",
        s=46,
        linewidth=1.4,
        zorder=4,
    )
    axes[0].set_ylabel(r"exact grid variance of $R_\beta$ (dollars$^2$)")
    axes[0].set_title("(a) Probability-weighted variance criterion")

    axes[1].plot(betas, ranges, color=ORANGE, linewidth=2.0)
    axes[1].scatter(
        [beta_range],
        [summary["range_optimum"]["grid_oscillation_dollars"]],
        color=ORANGE,
        marker="s",
        s=42,
        zorder=4,
    )
    axes[1].scatter(
        [beta_variance],
        [summary["variance_optimum"]["grid_oscillation_dollars"]],
        facecolor="none",
        edgecolor=BLUE,
        marker="o",
        s=46,
        linewidth=1.4,
        zorder=4,
    )
    axes[1].set_ylabel(r"exact grid oscillation of $R_\beta$ (dollars)")
    axes[1].set_title("(b) Finite-grid range criterion")
    axes[1].text(
        0.985,
        0.97,
        "shaded: residual is signed\non the encoded grid",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=GREY,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.text(
        0.5,
        0.005,
        f"Exact enumeration of the four-point, {grid_dates}-date grid "
        f"({grid_states:,} states); not a continuous-model bound or an "
        "executed query law.",
        ha="center",
        fontsize=8.5,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_grid_scatter(rows: list[dict], path: Path) -> None:
    sigmas = sorted({row["sigma"] for row in rows})
    strikes = sorted({row["strike"] for row in rows})
    markers = ["o", "s", "^", "D", "P"]
    cmap = matplotlib.colormaps["cividis"]
    sigma_color = {
        sigma: cmap(index / max(1, len(sigmas) - 1))
        for index, sigma in enumerate(sigmas)
    }
    strike_marker = dict(zip(strikes, markers))

    x = np.asarray([row["standard_deviation_ratio_beta_one"] for row in rows])
    y = np.asarray([row["matched_exceedance_range_ratio_proxy"] for row in rows])
    lower = 0.82 * min(float(x.min()), float(y.min()))
    upper = 1.20 * max(float(x.max()), float(y.max()))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.65, 5.8))
    ax.plot([lower, upper], [lower, upper], color="0.45", linestyle="--", linewidth=1.0)
    ax.text(upper / 1.12, upper / 1.18, "equal gain", rotation=45, ha="right", va="top", color=GREY, fontsize=8.5)

    for row in rows:
        is_base = row["sigma"] == 0.20 and row["strike"] == 100.0
        ax.scatter(
            row["standard_deviation_ratio_beta_one"],
            row["matched_exceedance_range_ratio_proxy"],
            s=78 if is_base else 54,
            marker=strike_marker[row["strike"]],
            facecolor=sigma_color[row["sigma"]],
            edgecolor="black" if is_base else "white",
            linewidth=1.25 if is_base else 0.65,
            zorder=4 if is_base else 3,
        )
        if is_base:
            ax.annotate(
                "base case",
                (
                    row["standard_deviation_ratio_beta_one"],
                    row["matched_exceedance_range_ratio_proxy"],
                ),
                xytext=(7, -12),
                textcoords="offset points",
                fontsize=8.5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(which="major", color="0.86", linewidth=0.6)
    ax.set_xlabel(r"standard-deviation reduction at $\beta=1$")
    ax.set_ylabel(r"matched-exceedance normalisation ratio at $\beta=1$ (proxy)")
    ax.set_title("Range and variance gains differ across the contract grid")

    strike_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=strike_marker[strike],
            color="none",
            markerfacecolor="0.50",
            markeredgecolor="white",
            markersize=7,
            label=f"K={strike:.0f}",
        )
        for strike in strikes
    ]
    first_legend = ax.legend(
        handles=strike_handles,
        title="strike",
        loc="upper left",
        frameon=False,
        fontsize=8.2,
        title_fontsize=8.5,
    )
    ax.add_artist(first_legend)

    normalizer = matplotlib.colors.Normalize(vmin=min(sigmas), vmax=max(sigmas))
    scalar = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"volatility $\sigma$")
    colorbar.set_ticks(sigmas)
    colorbar.set_ticklabels([f"{sigma:.2f}" for sigma in sigmas])

    fig.text(
        0.5,
        0.008,
        r"The range axis uses the manuscript's matched-exceedance $A_N-B_1$ proxy; neither axis is an executed query count.",
        ha="center",
        fontsize=8.2,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_REPO / "results")
    parser.add_argument("--grid-dates", type=int, default=DEFAULT_GRID_DATES)
    parser.add_argument(
        "--grid-chunk-size", type=int, default=DEFAULT_GRID_CHUNK_SIZE
    )
    parser.add_argument("--beta-min", type=float, default=DEFAULT_BETA_MIN)
    parser.add_argument("--beta-max", type=float, default=DEFAULT_BETA_MAX)
    parser.add_argument("--beta-points", type=int, default=DEFAULT_BETA_POINTS)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use the exact 10-date grid and 81 beta values for a smoke render",
    )
    args = parser.parse_args(argv)
    if args.fast:
        args.grid_dates = 10
        args.beta_points = 81
    if args.grid_dates < 2 or args.grid_chunk_size < 2 or args.beta_points < 3:
        parser.error("grid dates and chunk size must be >=2; beta points must be >=3")
    if not args.beta_min < 1.0 < args.beta_max:
        parser.error("the beta sweep must contain beta=1")
    args.out.mkdir(parents=True, exist_ok=True)

    target, control, probabilities = _exact_grid_payoffs(
        args.grid_dates, args.grid_chunk_size
    )
    beta_rows, beta_summary, beta_gates = _beta_sweep(
        target,
        control,
        probabilities,
        args.beta_min,
        args.beta_max,
        args.beta_points,
    )
    if args.grid_dates == 12:
        beta_gates.append(
            {
                "name": "12-date grid reproduces the published exact-grid ledger",
                "passed": bool(
                    abs(beta_summary["discounted_target_price"] - 5.819041)
                    < 5e-7
                    and abs(beta_summary["grid_target_max_dollars"] - 89.0766)
                    < 5e-5
                    and abs(
                        beta_summary["beta_one"]["grid_max_dollars"] - 8.9303
                    )
                    < 5e-5
                ),
                "observed": {
                    "discounted_target_price": beta_summary[
                        "discounted_target_price"
                    ],
                    "grid_target_max_dollars": beta_summary[
                        "grid_target_max_dollars"
                    ],
                    "beta_one_residual_max_dollars": beta_summary["beta_one"][
                        "grid_max_dollars"
                    ],
                },
                "published_rounded": [5.819041, 89.0766, 8.9303],
            }
        )
    scatter_rows, scatter_gates = _grid_scatter()
    gates = beta_gates + scatter_gates

    beta_png = args.out / "principle_beta_sweep.png"
    scatter_png = args.out / "principle_grid_scatter.png"
    beta_csv = args.out / "principle_beta_sweep.csv"
    scatter_csv = args.out / "principle_grid_scatter.csv"
    json_path = args.out / "principle_diagnostics.json"
    _plot_beta_sweep(
        beta_rows,
        beta_summary,
        beta_png,
        args.grid_dates,
        4**args.grid_dates,
    )
    _plot_grid_scatter(scatter_rows, scatter_png)
    _write_csv(beta_csv, beta_rows)
    _write_csv(scatter_csv, scatter_rows)

    base_cell = next(
        row
        for row in scatter_rows
        if row["sigma"] == 0.20 and row["strike"] == 100.0
    )
    record = {
        "schema_version": "principle-diagnostics-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "scripts/figure_principle_diagnostics.py",
        "git_revision": _git_revision(),
        "inputs": {
            _path_label(path): {"sha256": _sha256(path)}
            for path in (
                GENERATOR_INPUT,
                VR_INPUT,
                VR_GENERATOR_INPUT,
                RANGE_INPUT,
                RANGE_GENERATOR_INPUT,
                SECTION6_INPUT,
                EXACT12_INPUT,
                EXACT12_GENERATOR_INPUT,
            )
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "beta_sweep": {
            "evidence_type": "exact probability-weighted enumeration of a finite four-point conditional-mean grid",
            "contract": {
                "s0": S0,
                "strike": STRIKE,
                "rate": RATE,
                "sigma": SIGMA,
                "maturity": MATURITY,
                "monitoring_dates": args.grid_dates,
            },
            "grid_states": 4**args.grid_dates,
            "grid_chunk_size": args.grid_chunk_size,
            "shock_grid": {
                "conditional_mean_points": SHOCKS.tolist(),
                "probabilities": SHOCK_PROBABILITIES.tolist(),
            },
            "beta_interval": [args.beta_min, args.beta_max],
            "beta_points": args.beta_points,
            "summary": beta_summary,
            "rows": beta_rows,
            "limitations": [
                "The variance and oscillation are exact only for the declared finite encoding grid.",
                "The grid is not the daily 252-date model or a continuous-Black-Scholes range bound.",
                "It does not execute an amplitude-estimation schedule.",
            ],
        },
        "grid_scatter": {
            "evidence_type": "derived synthesis of two existing seeded experiments",
            "beta": 1.0,
            "aggregation": "ratio of the mean raw standard error to the mean reconstructed beta=1 residual standard error across eight trials per cell",
            "base_cell": base_cell,
            "rows": scatter_rows,
            "limitations": [
                "The normalisation ratio is the manuscript's matched-exceedance A_N-B_1 proxy.",
                "The variance sufficient statistics and range proxy use different seeded path experiments.",
                "The plotted ratios are modelled coefficients, not executed query counts.",
            ],
        },
        "gates": gates,
        "outputs": {
            _path_label(path): {"sha256": _sha256(path)}
            for path in (beta_png, scatter_png, beta_csv, scatter_csv)
        },
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n")

    for gate in gates:
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] {gate['name']}")
    print(
        f"beta*: {beta_summary['beta_variance_optimal']:.6f}; "
        f"beta_infinity: {beta_summary['beta_range_optimal']:.6f}; "
        f"base-cell std/range ratios: "
        f"{base_cell['standard_deviation_ratio_beta_one']:.3f}/"
        f"{base_cell['matched_exceedance_range_ratio_proxy']:.3f}"
    )
    print(f"wrote {beta_png}")
    print(f"wrote {scatter_png}")
    print(f"wrote {json_path}")
    return 0 if all(gate["passed"] for gate in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
