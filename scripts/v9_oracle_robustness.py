"""Robustness of the recommended reference-oracle recipe across parameters.

Recipe: Gauss-Hermite 8 points, probabilities rematched on the rounded grid,
shock scale 32, price scale 256, no cap.  Encoded prices come from exhaustive
enumeration, which the executed circuits reproduce (results/v9).  References
are scrambled Sobol with replicate standard errors.

Parts:
  grid    K in {90,95,100,105,110} x sigma in {0.10,0.20,0.30,0.40}, 4 dates
  dates   N in {2,3,4,5,6} at S0=K=100, sigma=0.20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

from qc_option_pricing.quantum.asian_oracle import (
    AsianGridSpec,
    enumerate_encoded_asian,
    gauss_hermite_normal_grid,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v9" / "oracle_robustness.json"

S0, RATE, MATURITY = 100.0, 0.05, 1.0
SS, PS = 32, 256


def rematched_grid(q, ss, clip_tolerance=1e-5):
    """Rounded GH points with probabilities re-solved for the normal moments.

    At sixteen points the Vandermonde solution puts about -6e-7 on one outer
    node; probabilities above -clip_tolerance are clipped to zero and the
    vector renormalised, which perturbs the matched moments at the 1e-4
    relative level.  Anything more negative raises.
    """
    pts, _ = gauss_hermite_normal_grid(q)
    rp = tuple(round(z * ss) / ss for z in pts)
    x = np.asarray(rp)
    k = len(x)
    target = [float(np.prod(np.arange(m - 1, 0, -2))) if m % 2 == 0 else 0.0
              for m in range(k)]
    p = np.linalg.solve(np.vander(x, k, increasing=True).T, np.array(target))
    if np.any(p < -clip_tolerance):
        raise ValueError(f"rematch infeasible for q={q}, ss={ss}")
    p = np.clip(p, 0.0, None)
    return rp, tuple(float(v) for v in (p / p.sum()))


def reference_qmc(n, strike, sigma, log2_paths=21, reps=4, seed=7):
    dt = MATURITY / n
    drift = (RATE - 0.5 * sigma * sigma) * dt
    dif = sigma * math.sqrt(dt)
    disc = math.exp(-RATE * MATURITY)
    estimates = []
    for r in range(reps):
        sob = qmc.Sobol(d=n, scramble=True, seed=seed + r)
        U = sob.random(1 << log2_paths)
        Z = norm.ppf(np.clip(U, 1e-15, 1 - 1e-15))
        logS = math.log(S0) + np.cumsum(drift + dif * Z, axis=1)
        Avg = np.exp(logS).mean(axis=1)
        estimates.append(disc * float(np.maximum(Avg - strike, 0.0).mean()))
    return float(np.mean(estimates)), float(np.std(estimates, ddof=1) / math.sqrt(reps))


def encoded_price(n, strike, sigma, pts, prob):
    spec = AsianGridSpec(
        n_dates=n, shock_points=pts, shock_probabilities=prob,
        s0=S0, strike=strike, rate=RATE, volatility=sigma, maturity=MATURITY,
        shock_scale=SS, price_scale=PS,
    )
    ref = enumerate_encoded_asian(spec, max_paths=1_100_000)
    return math.exp(-RATE * MATURITY) * ref.clipped_raw_payoff_undiscounted


def merge(part, payload):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[part] = payload
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"merged part '{part}' into {OUT.relative_to(ROOT)}")


def part_grid(q=3):
    pts, prob = rematched_grid(q, SS)
    label = f"GH{1 << q} rematched, ss={SS}, ps={PS}"
    rows = []
    worst = 0.0
    for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
        for sigma in (0.10, 0.20, 0.30, 0.40):
            ref, se = reference_qmc(4, strike, sigma)
            enc = encoded_price(4, strike, sigma, pts, prob)
            err = enc - ref
            worst = max(worst, abs(err))
            rows.append({"strike": strike, "sigma": sigma, "reference": ref,
                         "reference_se": se, "encoded": enc, "error": err})
            print(f"  K={strike:5.0f} sigma={sigma:.2f}  ref={ref:9.6f}+/-{se:.1e}"
                  f"  enc={enc:9.6f}  err={err:+.6f}")
    print(f"  worst |error| over {len(rows)} cells: {worst:.6f}")
    merge("grid" if q == 3 else f"grid{1 << q}",
          {"recipe": label, "n_dates": 4, "cells": rows,
           "worst_abs_error": worst})


def part_dates(q=3, date_counts=(2, 3, 4, 5, 6)):
    pts, prob = rematched_grid(q, SS)
    points = 1 << q
    rows = []
    for n in date_counts:
        ref, se = reference_qmc(n, 100.0, 0.20)
        enc = encoded_price(n, 100.0, 0.20, pts, prob)
        err = enc - ref
        rows.append({"n_dates": n, "paths": points ** n, "reference": ref,
                     "reference_se": se, "encoded": enc, "error": err})
        print(f"  N={n}  paths={points**n:>9,}  ref={ref:9.6f}+/-{se:.1e}"
              f"  enc={enc:9.6f}  err={err:+.6f}")
    merge("dates" if q == 3 else f"dates{points}",
          {"recipe": f"GH{points} rematched, ss={SS}, ps={PS}",
           "strike": 100.0, "sigma": 0.20, "rows": rows})


def part_quadrature_ladder():
    """Pure quadrature error of unrounded GH grids at two and three dates.

    Exact real arithmetic, no shock or price rounding.  Shows the kink-driven
    oscillation of the Gauss-Hermite error with grid size, which is why the
    grid must be certified per instance by enumeration.
    """
    rows = []
    for n in (2, 3):
        ref, se = reference_qmc(n, 100.0, 0.20)
        row = {"n_dates": n, "reference": ref, "reference_se": se}
        line = f"  N={n}: "
        for q in (3, 4, 5, 6):
            pts, prob = gauss_hermite_normal_grid(q)
            pts_a, prob_a = np.asarray(pts), np.asarray(prob)
            dt = MATURITY / n
            drift = (RATE - 0.5 * 0.20 * 0.20) * dt
            dif = 0.20 * math.sqrt(dt)
            g = np.meshgrid(*([pts_a] * n), indexing="ij")
            Z = np.stack([x.ravel() for x in g], axis=1)
            pg = np.meshgrid(*([prob_a] * n), indexing="ij")
            P = np.ones(Z.shape[0])
            for x in pg:
                P *= x.ravel()
            logS = math.log(S0) + np.cumsum(drift + dif * Z, axis=1)
            pay = np.maximum(np.exp(logS).mean(axis=1) - 100.0, 0.0)
            err = math.exp(-RATE * MATURITY) * float(P @ pay) - ref
            row[f"gh{1 << q}_error"] = err
            line += f"GH{1 << q}={err:+.6f}  "
        rows.append(row)
        print(line)
    # record the marginal infeasibility of the sixteen-point rematch
    pts, _ = gauss_hermite_normal_grid(4)
    rp = tuple(round(z * SS) / SS for z in pts)
    x = np.asarray(rp)
    target = np.array([float(np.prod(np.arange(m - 1, 0, -2))) if m % 2 == 0
                       else 0.0 for m in range(16)])
    raw = np.linalg.solve(np.vander(x, 16, increasing=True).T, target)
    clipped = np.clip(raw, 0.0, None)
    clipped = clipped / clipped.sum()
    moment_errors = {m: float(clipped @ x**m) - float(target[m])
                     for m in range(6)}
    print(f"  GH16 rematch: min raw probability {raw.min():.2e}; "
          f"post-clip moment errors {moment_errors}")
    merge("quadrature_ladder",
          {"rows": rows,
           "gh16_rematch_min_raw_probability": float(raw.min()),
           "gh16_rematch_post_clip_moment_errors": moment_errors})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=["grid", "grid16", "dates", "dates16",
                                 "quadrature_ladder"])
    args = parser.parse_args()
    if args.part == "grid":
        part_grid(3)
    elif args.part == "grid16":
        part_grid(4)
    elif args.part == "dates":
        part_dates(3)
    elif args.part == "dates16":
        # 16**6 paths is out of enumeration range; five dates is the limit
        part_dates(4, date_counts=(2, 3, 4, 5))
    else:
        part_quadrature_ladder()


if __name__ == "__main__":
    main()
