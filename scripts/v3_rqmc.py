#!/usr/bin/env python3
"""v3: randomized quasi-Monte Carlo (scrambled Sobol + Brownian bridge) with
and without the Kemna-Vorst control variate, for the daily arithmetic Asian
call.

The Brownian-bridge construction assigns the terminal and then conditional
Brownian values to successive Sobol coordinates.  SciPy's scrambled Sobol
generator uses a left linear matrix scramble followed by a digital random
shift.  We report the sample standard deviation across N_REP independent
randomizations.  This is the standard deviation of one randomized replicate;
the standard error of their mean is smaller by sqrt(N_REP).

Reports replicate SD vs M for plain MC, MC+KV, RQMC, RQMC+KV, and the fitted
finite-range slope alpha (SD ~ M^-alpha).

Outputs: results/v3/rqmc_table.txt, results/v3/rqmc_convergence.png,
results/v3/rqmc_replicates.json, and results/v3/rqmc_replicates.csv.
"""
from __future__ import annotations

import math
import argparse
import csv
import importlib.metadata
import json
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
from scipy.stats import norm, qmc

from qc_option_pricing.classical.asian_mc import geometric_asian_call_exact

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N = 252
DISC = math.exp(-R * T)
M_EXPONENTS = [10, 11, 12, 13, 14, 15, 16]      # M = 2^m Sobol points
N_REP = 16                                       # independent scrambles
MASTER_SEED = 20_260_716

OUT = _REPO / "results" / "v3"
OUT.mkdir(parents=True, exist_ok=True)


def brownian_bridge_order() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the Brownian-bridge insertion order and interpolation weights.

    Returns (index, left, right, is_first) arrays: coordinate j of the point set
    fixes W at time index[j], conditioned on already-known values at left[j]
    and right[j] (Moskowitz-Caflisch construction on a dyadic-ish schedule).
    """
    known = [None] * (N + 1)          # slot i <-> time t_i = i*T/N; W_0 = 0 known
    order, lefts, rights = [], [], []
    known[0] = True
    # first coordinate fixes the terminal point
    order.append(N); lefts.append(0); rights.append(-1); known[N] = True
    queue = [(0, N)]
    while queue:
        lo, hi = queue.pop(0)
        if hi - lo < 2:
            continue
        mid = (lo + hi) // 2
        order.append(mid); lefts.append(lo); rights.append(hi)
        known[mid] = True
        queue.append((lo, mid)); queue.append((mid, hi))
    return (np.array(order), np.array(lefts), np.array(rights),
            np.array([r == -1 for r in rights]))


_BB = brownian_bridge_order()


def paths_from_uniforms(u: np.ndarray) -> np.ndarray:
    """Map uniforms (M, N) -> fixing prices (M, N) via Brownian bridge."""
    m = u.shape[0]
    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    dt = T / N
    w = np.zeros((m, N + 1))
    order, lefts, rights, is_terminal = _BB
    for j in range(N):
        idx, lo, hi = order[j], lefts[j], rights[j]
        if is_terminal[j]:
            w[:, idx] = math.sqrt(idx * dt) * z[:, j]
        else:
            t_l, t_m, t_h = lo * dt, idx * dt, hi * dt
            mean = ((t_h - t_m) * w[:, lo] + (t_m - t_l) * w[:, hi]) / (t_h - t_l)
            var = (t_m - t_l) * (t_h - t_m) / (t_h - t_l)
            w[:, idx] = mean + math.sqrt(var) * z[:, j]
    t_grid = np.arange(1, N + 1) * dt
    log_s = math.log(S0) + (R - 0.5 * SIGMA**2) * t_grid + SIGMA * w[:, 1:]
    return np.exp(log_s)


def estimators(prices: np.ndarray, geo_exact: float) -> tuple[float, float, float]:
    """Return vanilla estimate, fitted KV estimate, and fitted coefficient."""
    arith = np.maximum(prices.mean(axis=1) - K, 0.0) * DISC
    geo = np.maximum(np.exp(np.log(prices).mean(axis=1)) - K, 0.0) * DISC
    cov = np.cov(arith, geo)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
    return (
        float(arith.mean()),
        float(arith.mean() + beta * (geo_exact - geo.mean())),
        float(beta),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replicates", type=int, default=N_REP)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    args = parser.parse_args(argv)
    if args.replicates < 2:
        parser.error("replicates must be at least 2")
    args.out.mkdir(parents=True, exist_ok=True)

    geo_exact = geometric_asian_call_exact(S0, K, R, SIGMA, T, N)

    results = {"MC": [], "MC+KV": [], "RQMC": [], "RQMC+KV": []}
    replicate_rows: list[dict] = []
    ms = [2**e for e in M_EXPONENTS]
    for exponent_index, e in enumerate(M_EXPONENTS):
        m = 2**e
        v_mc, cv_mc, v_q, cv_q = [], [], [], []
        for rep in range(args.replicates):
            seed_offset = 2 * (exponent_index * args.replicates + rep)
            mc_seed = args.master_seed + seed_offset
            sobol_seed = args.master_seed + seed_offset + 1
            rng = np.random.default_rng(mc_seed)
            u = rng.random((m, N))                       # plain MC uniforms
            est_v, est_cv, beta_mc = estimators(paths_from_uniforms(u), geo_exact)
            v_mc.append(est_v); cv_mc.append(est_cv)

            sob = qmc.Sobol(d=N, scramble=True, seed=sobol_seed)
            uq = sob.random_base2(e)
            est_q, est_q_cv, beta_rqmc = estimators(paths_from_uniforms(uq), geo_exact)
            v_q.append(est_q); cv_q.append(est_q_cv)
            replicate_rows.append({
                "exponent": e,
                "points": m,
                "replicate_index": rep,
                "mc_seed": mc_seed,
                "sobol_scramble_seed": sobol_seed,
                "mc_estimate": est_v,
                "mc_kv_estimate": est_cv,
                "mc_kv_beta": beta_mc,
                "rqmc_estimate": est_q,
                "rqmc_kv_estimate": est_q_cv,
                "rqmc_kv_beta": beta_rqmc,
            })
        for name, vals in [("MC", v_mc), ("MC+KV", cv_mc),
                           ("RQMC", v_q), ("RQMC+KV", cv_q)]:
            sd = float(np.std(vals, ddof=1))
            results[name].append((m, float(np.mean(vals)), sd))
        print(f"M=2^{e}: " + "  ".join(
            f"{k}={results[k][-1][1]:.5f}±{results[k][-1][2]:.6f}"
            for k in results), flush=True)

    # fitted rates
    rates = {}
    for name, rows in results.items():
        lx = np.log([r_[0] for r_ in rows])
        ly = np.log([r_[2] for r_ in rows])
        rates[name] = -float(np.polyfit(lx, ly, 1)[0])

    lines = [
        "RQMC (scrambled Sobol + Brownian bridge) vs MC, daily Asian call",
        f"S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}, N={N}; "
        f"{args.replicates} independent sample sets/randomizations per point",
        f"master_seed={args.master_seed}; MC and Sobol seeds are recorded in rqmc_replicates.json",
        "SD = sample standard deviation across single-replicate estimates.",
        f"RQMC randomization = SciPy Sobol LMS+digital shift; points generated with random_base2.",
        "",
        f"{'M':>9} " + "".join(f"{k + ' est':>14}{'SD':>11}" for k in results),
    ]
    for i, m in enumerate(ms):
        row = f"{m:>9,}"
        for k in results:
            row += f"{results[k][i][1]:>14.5f}{results[k][i][2]:>11.6f}"
        lines.append(row)
    lines.append("")
    lines.append("fitted replicate SD ~ M^-alpha over the sweep: " + ", ".join(
        f"{k}: alpha={rates[k]:.2f}" for k in results))
    se_mc = results["MC"][-1][2]
    se_best = results["RQMC+KV"][-1][2]
    lines.append(f"at M=2^{M_EXPONENTS[-1]}: SD(MC)/SD(RQMC+KV) = {se_mc / se_best:,.0f}x "
                 f"(variance ratio {(se_mc / se_best)**2:,.0f}x)")
    (args.out / "rqmc_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))

    aggregate_rows = [
        {"method": method, "points": points, "mean": mean, "replicate_sd": sd}
        for method, rows in results.items()
        for points, mean, sd in rows
    ]
    payload = {
        "schema": "parikh-rayan-rqmc-replicates-v1",
        "config": {
            "s0": S0,
            "strike": K,
            "rate": R,
            "volatility": SIGMA,
            "maturity": T,
            "monitoring_dates": N,
            "exponents": M_EXPONENTS,
            "replicates": args.replicates,
            "master_seed": args.master_seed,
            "seed_rule": (
                "offset=2*(exponent_index*replicates+replicate_index); "
                "mc_seed=master_seed+offset; sobol_seed=mc_seed+1"
            ),
            "geometric_control_exact": geo_exact,
            "numpy_version": np.__version__,
            "scipy_version": importlib.metadata.version("scipy"),
        },
        "fitted_slopes": rates,
        "aggregates": aggregate_rows,
        "replicates": replicate_rows,
    }
    (args.out / "rqmc_replicates.json").write_text(json.dumps(payload, indent=1))
    with (args.out / "rqmc_replicates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replicate_rows[0]))
        writer.writeheader()
        writer.writerows(replicate_rows)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    styles = {"MC": ("#dc2626", "o-"), "MC+KV": ("#2563eb", "s-"),
              "RQMC": ("#f59e0b", "^-"), "RQMC+KV": ("#16a34a", "d-")}
    for name, rows in results.items():
        color, style = styles[name]
        ax.loglog([r_[0] for r_ in rows], [r_[2] for r_ in rows], style,
                  color=color, ms=5,
                  label=f"{name} ($\\alpha={rates[name]:.2f}$)")
    mref = np.array(ms, dtype=float)
    ax.loglog(mref, results["MC"][0][2] * (mref[0] / mref)**0.5, ":",
              color="grey", lw=1.2, label=r"$M^{-1/2}$")
    ax.loglog(mref, results["RQMC+KV"][0][2] * (mref[0] / mref), "--",
              color="grey", lw=1.2, label=r"$M^{-1}$")
    ax.set_xlabel("points per replicate $M$")
    ax.set_ylabel("standard deviation across replicate estimates")
    ax.set_title("RQMC + Brownian bridge + Kemna--Vorst, daily Asian call")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(args.out / "rqmc_convergence.png", dpi=200, bbox_inches="tight")
    print(f"-> wrote {args.out}/rqmc_convergence.png and seeded replicate tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
