#!/usr/bin/env python3
"""Cross-check KV/geo and Xu et al. (2023) Table 4; writes results/literature_validation_report.txt."""

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


def _xu_black_scholes_author_convention(
    *,
    days: int,
    n_paths: int,
    chunk_size: int = 50_000,
    seed: int = 20260414,
) -> tuple[float, float]:
    """Xu et al. Table4 w/o ALL: avg over S0 + T daily sim prices (not t1..tN-only)."""
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

    g_ok = z_geo <= 3.0
    kv_ok = z_cv_vs_mc <= 3.0
    lb_ok = arith_cv >= geo_exact
    lines.append(
        "KV/geo (S0=K=100,r=0.05,s=0.2,T=1,N=252,t1..tN): "
        f"geo_exact={geo_exact:.6f} geo_MC={geo_mc:.6f}±{geo_se:.6f} z={z_geo:.2f} "
        f"arith_MC={arith_mc:.6f}±{arith_se:.6f} arith_KV={arith_cv:.6f}±{arith_cv_se:.6f} "
        f"KV/vanilla z={z_cv_vs_mc:.2f} stderr×{arith_se / arith_cv_se:.1f} "
        f"geo_MC~exact {'PASS' if g_ok else 'FAIL'} KV~vanilla {'PASS' if kv_ok else 'FAIL'} "
        f"arith≥geo {'PASS' if lb_ok else 'FAIL'}"
    )


def validate_xu_zhang_wang_table(lines: list[str]) -> None:
    published = {
        30: (1.3871, 0.0201),
        90: (2.6589, 0.0373),
        180: (4.0166, 0.0546),
    }
    all_ok = True
    parts: list[str] = []
    for days, (pub_price, pub_se) in published.items():
        ours, our_se = _xu_black_scholes_author_convention(days=days, n_paths=1_000_000)
        z_pub = abs(ours - pub_price) / pub_se
        ok = z_pub <= 3.0
        all_ok = all_ok and ok
        parts.append(
            f"T{days} pub{pub_price:.4f}±{pub_se:.4f} ours{ours:.4f}±{our_se:.4f} z_pub{z_pub:.2f}"
        )
    lines.append(
        "Xu Zhang Wang 2023 Math11 Table4 w/o ALL; author code conv (S0+T dailies). 1e6 paths: "
        + " | ".join(parts)
        + f" | within 3 pub SE: {'PASS' if all_ok else 'FAIL'}"
    )


def main() -> None:
    lines: list[str] = []
    validate_kemna_vorst_closed_form(lines)
    validate_xu_zhang_wang_table(lines)

    out_path = RESULTS_DIR / "literature_validation_report.txt"
    out_path.write_text("\n".join(lines) + "\n\n")
    print("\n".join(lines))
    print(f"→ Wrote {out_path}")


if __name__ == "__main__":
    main()
