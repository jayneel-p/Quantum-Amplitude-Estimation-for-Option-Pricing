#!/usr/bin/env python3
"""v4: validate the Asian Monte Carlo pipeline against Linetsky (2004).

Linetsky, Operations Research 52(6):856-867, Table 3, gives continuously
averaged arithmetic Asian call prices exact to ten digits for seven standard
test cases (K = 2.0, q = 0 throughout).  Our simulator prices the discretely
monitored contract, whose difference from the continuous one is O(1/N).  For
each case we price with the Kemna-Vorst control variate at N = 125, 250, 500
monitoring dates and Richardson-extrapolate: C_inf ~ 2 C(500) - C(250).

Gates (fail-closed), per case:
  R1  the discrete-continuous gap shrinks with N (|C(500)-L| < |C(125)-L|);
  R2  |Richardson - Linetsky| < max(4 * SE_Richardson, 3e-4).

Output: results/v4/linetsky_benchmark.txt
"""
from __future__ import annotations

import math
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from scipy.stats import norm

K_STRIKE = 2.0
N_LIST = (125, 250, 500)
M_PATHS = 500_000
CHUNK = 100_000
SEED = 20260705

# case: (r, sigma, T, S0, Linetsky EE price)
CASES = [
    (0.02,   0.10, 1.0, 2.0, 0.0559860415),
    (0.18,   0.30, 1.0, 2.0, 0.2183875466),
    (0.0125, 0.25, 2.0, 2.0, 0.1722687410),
    (0.05,   0.50, 1.0, 1.9, 0.1931737903),
    (0.05,   0.50, 1.0, 2.0, 0.2464156905),
    (0.05,   0.50, 1.0, 2.1, 0.3062203648),
    (0.05,   0.50, 2.0, 2.0, 0.3500952190),
]

OUT = _REPO / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def geo_asian_closed(s0, k, r, sig, t, n) -> float:
    """Discrete geometric Asian call, fixings at iT/N, i=1..N."""
    sig_g = sig * math.sqrt((n + 1) * (2 * n + 1) / (6.0 * n * n))
    m_g = (r - sig * sig / 2) * (n + 1) / (2.0 * n)
    v = sig_g * sig_g * t
    d1 = (math.log(s0 / k) + m_g * t + v) / math.sqrt(v)
    d2 = d1 - math.sqrt(v)
    return math.exp(-r * t) * (s0 * math.exp(m_g * t + v / 2) * norm.cdf(d1)
                               - k * norm.cdf(d2))


def price_kv_cv(s0, r, sig, t, n, rng) -> tuple[float, float]:
    """KV-CV Monte Carlo price of the discrete arithmetic Asian call and SE."""
    dt = t / n
    drift = (r - sig * sig / 2) * dt
    vol = sig * math.sqrt(dt)
    disc = math.exp(-r * t)
    theta = geo_asian_closed(s0, K_STRIKE, r, sig, t, n)
    xs, ys = [], []
    for _ in range(M_PATHS // CHUNK):
        z = rng.standard_normal((CHUNK, n))
        logs = math.log(s0) + np.cumsum(drift + vol * z, axis=1)
        a = np.exp(logs).mean(axis=1)
        g = np.exp(logs.mean(axis=1))
        xs.append(disc * np.maximum(a - K_STRIKE, 0.0))
        ys.append(disc * np.maximum(g - K_STRIKE, 0.0))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    beta = float(np.cov(x, y)[0, 1] / np.var(y, ddof=1))
    resid = x + beta * (theta - y)
    return float(resid.mean()), float(resid.std(ddof=1)) / math.sqrt(len(resid))


def main() -> int:
    rng = np.random.default_rng(SEED)
    lines = [
        "KV-CV Monte Carlo vs Linetsky (2004) 10-digit continuous benchmarks",
        f"K=2.0, {M_PATHS} paths per (case, N), N in {N_LIST}, "
        "Richardson C_inf ~ 2 C(500) - C(250)",
        f"seed {SEED}",
        "",
        " case   r      sigma  T   S0    Linetsky        Richardson      "
        "+-SE       diff       z",
    ]
    for i, (r, sig, t, s0, lin) in enumerate(CASES, 1):
        cs, ses = {}, {}
        for n in N_LIST:
            cs[n], ses[n] = price_kv_cv(s0, r, sig, t, n, rng)
        rich = 2 * cs[500] - cs[250]
        se_rich = math.sqrt(4 * ses[500] ** 2 + ses[250] ** 2)
        diff = rich - lin
        z = diff / se_rich
        gap125, gap500 = abs(cs[125] - lin), abs(cs[500] - lin)
        check(f"R1 case {i} gap shrinks", gap500 < gap125,
              f"|C(125)-L|={gap125:.2e} -> |C(500)-L|={gap500:.2e}")
        check(f"R2 case {i} Richardson vs Linetsky",
              abs(diff) < max(4 * se_rich, 3e-4),
              f"diff={diff:+.2e} (SE {se_rich:.1e}, z={z:+.2f})")
        lines.append(f"   {i}   {r:6.4f} {sig:.2f}  {t:.0f}  {s0:.1f}  "
                     f"{lin:.10f}  {rich:.10f}  {se_rich:.1e}  "
                     f"{diff:+.2e}  {z:+5.2f}")
    lines += ["", "gates: " + ("ALL PASS" if not FAILURES else f"FAILED {FAILURES}")]
    (OUT / "linetsky_benchmark.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
