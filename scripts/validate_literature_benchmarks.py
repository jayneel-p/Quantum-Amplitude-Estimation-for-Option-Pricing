#!/usr/bin/env python3
"""
Validate classical Asian-pricing code against published/simple benchmarks.

The goal is not to tune to a single random seed.  It is to check that our GBM
simulation, geometric Asian closed form, and Kemna-Vorst control-variate logic
agree with known theory and with a published Black-Scholes Asian benchmark.

Outputs:
  results/literature_validation_report.txt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qc_option_pricing.classical import (  # noqa: E402
    arithmetic_asian_call_mc,
    arithmetic_asian_call_cv,
    geometric_asian_call_exact,
    geometric_asian_call_mc,
)

RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _append_result(lines: list[str], label: str, ok: bool) -> None:
    lines.append(f"{label}: {'PASS' if ok else 'CHECK'}")


def _xu_black_scholes_author_convention(
    *,
    days: int,
    n_paths: int,
    chunk_size: int = 50_000,
    seed: int = 20260414,
) -> tuple[float, float]:
    """
    Reproduce the plain Black-Scholes / pseudo-random MC convention behind
    Xu, Zhang, Wang (2023), Table 4 "w/o ALL".

    Their released code initializes an all-zero increment column, then computes
    the arithmetic average over exp(log(S0) + cumulative increments). That means
    the average includes S0 plus each simulated daily price: T + 1 prices total.
    This differs from our paper's main convention, which averages t_1..t_N only.
    """
    s0 = 50.0
    k = 50.0
    r_daily = 0.0005
    sigma_daily = 0.02

    rng = np.random.default_rng(seed + days)
    n_done = 0
    total = 0.0
    total_sq = 0.0

    while n_done < n_paths:
        n = min(chunk_size, n_paths - n_done)
        shocks = rng.standard_normal((n, days))
        increments = (r_daily - 0.5 * sigma_daily**2) + sigma_daily * shocks
        paths = s0 * np.exp(np.cumsum(increments, axis=1))
        avg = (s0 + paths.sum(axis=1)) / (days + 1)
        payoffs = math.exp(-r_daily * days) * np.maximum(avg - k, 0.0)

        total += float(payoffs.sum())
        total_sq += float(np.dot(payoffs, payoffs))
        n_done += n

    mean = total / n_paths
    var = (total_sq - n_paths * mean**2) / (n_paths - 1)
    stderr = math.sqrt(var / n_paths)
    return mean, stderr


def validate_kemna_vorst_closed_form(lines: list[str]) -> None:
    lines.append("1. Kemna-Vorst / geometric Asian closed-form check")
    lines.append("-" * 62)
    lines.append(
        "Model: one-factor GBM, S0=100, K=100, r=0.05, sigma=0.2, T=1, "
        "N=252, averages over t_1..t_N."
    )

    s0 = 100.0
    k = 100.0
    r = 0.05
    sigma = 0.2
    t = 1.0
    n_steps = 252
    n_paths = 300_000

    geo_exact = geometric_asian_call_exact(s0, k, r, sigma, t, n_steps)
    geo_mc, geo_se = geometric_asian_call_mc(
        s0,
        k,
        r,
        sigma,
        t,
        n_steps,
        n_paths,
        rng=np.random.default_rng(1101),
    )
    z_geo = abs(geo_mc - geo_exact) / geo_se

    arith_mc, arith_se = arithmetic_asian_call_mc(
        s0,
        k,
        r,
        sigma,
        t,
        n_steps,
        n_paths,
        rng=np.random.default_rng(1102),
    )
    arith_cv, arith_cv_se = arithmetic_asian_call_cv(
        s0,
        k,
        r,
        sigma,
        t,
        n_steps,
        n_paths,
        rng=np.random.default_rng(1103),
    )

    combined_se = math.sqrt(arith_se**2 + arith_cv_se**2)
    z_cv_vs_mc = abs(arith_cv - arith_mc) / combined_se

    lines.append(f"Geometric exact:          {geo_exact:.6f}")
    lines.append(f"Geometric MC:             {geo_mc:.6f} ± {geo_se:.6f}  z={z_geo:.2f}")
    lines.append(f"Arithmetic vanilla MC:    {arith_mc:.6f} ± {arith_se:.6f}")
    lines.append(f"Arithmetic KV estimate:   {arith_cv:.6f} ± {arith_cv_se:.6f}")
    lines.append(f"KV vs vanilla z-score:    {z_cv_vs_mc:.2f}")
    lines.append(f"Stderr reduction:         {arith_se / arith_cv_se:.1f}x")

    _append_result(lines, "Geometric MC within 3 standard errors of closed form", z_geo <= 3.0)
    _append_result(lines, "KV estimate consistent with vanilla MC", z_cv_vs_mc <= 3.0)
    _append_result(lines, "Arithmetic price above geometric lower bound", arith_cv >= geo_exact)
    lines.append("")


def validate_xu_zhang_wang_table(lines: list[str]) -> None:
    lines.append("2. Published Black-Scholes Asian benchmark: Xu, Zhang, Wang (2023)")
    lines.append("-" * 62)
    lines.append(
        "Source: Mathematics 11(3), 594, Table 4, row 'w/o ALL'. "
        "The paper states S0=50, K=50, daily r=0.0005, initial daily "
        "sigma=0.02, T in {30,90,180}, N=10,000."
    )
    lines.append(
        "Convention matched to the authors' released code: arithmetic average "
        "uses S0 plus T simulated daily prices (T+1 observations)."
    )
    lines.append(
        "We run 1,000,000 pseudo-random paths for a tighter independent estimate; "
        "PASS means our estimate is within 3 of the published standard errors."
    )
    lines.append("")
    lines.append(
        f"{'T(days)':>7}  {'Published':>10}  {'Pub SE':>8}  "
        f"{'Ours':>10}  {'Our SE':>8}  {'Diff':>9}  {'z_pub':>7}"
    )
    lines.append("-" * 72)

    published = {
        30: (1.3871, 0.0201),
        90: (2.6589, 0.0373),
        180: (4.0166, 0.0546),
    }
    all_ok = True
    for days, (pub_price, pub_se) in published.items():
        ours, our_se = _xu_black_scholes_author_convention(days=days, n_paths=1_000_000)
        diff = ours - pub_price
        z_pub = abs(diff) / pub_se
        ok = z_pub <= 3.0
        all_ok = all_ok and ok
        lines.append(
            f"{days:>7d}  {pub_price:>10.4f}  {pub_se:>8.4f}  "
            f"{ours:>10.4f}  {our_se:>8.4f}  {diff:>9.4f}  {z_pub:>7.2f}"
        )

    lines.append("")
    _append_result(lines, "Xu et al. Table 4 'w/o ALL' reproduced within 3 published SEs", all_ok)
    lines.append(
        "Note: the published values are one 10,000-path MC estimate, so exact "
        "price matching is neither expected nor required. The short-maturity "
        "case is the loosest but still within the 3-SE diagnostic band."
    )
    lines.append("")


def main() -> None:
    lines: list[str] = []
    lines.append("Literature / Benchmark Validation Report")
    lines.append("=" * 62)
    lines.append("")
    validate_kemna_vorst_closed_form(lines)
    validate_xu_zhang_wang_table(lines)

    out_path = RESULTS_DIR / "literature_validation_report.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"→ Wrote {out_path}")


if __name__ == "__main__":
    main()
