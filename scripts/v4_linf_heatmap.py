#!/usr/bin/env python3
"""v4: maximum-payoff (query-ratio) heatmap across moneyness and volatility.

The variance-reduction heatmap (Fig. vr_heatmap) shows the classical gain of the
Kemna-Vorst control variate. The quantum query count scales with the maximum
payoff, not the variance, so the quantum-side gain is a different quantity. This
script measures f_max(A)/f_max(R) at k=1 on the same (K, sigma) grid as the
variance-reduction sweep, using the same clipping rule (a fraction Phi(-3) of
simulated paths lies above each cap).

f_max(A) and f_max(R) are the empirical quantiles of A and of A-G at that
fraction, matching v4_qcv_extensions.py. The residual A-G bounds R pathwise.

Gates:
  base case (K=100, sigma=0.20) reproduces the daily ratio 16.0;
  A >= G on every path in every cell (Lemma 1);
  the ratio decreases with volatility at fixed strike.

Outputs: results/v4/linf_ratio_heatmap.png, results/v4/linf_heatmap.txt.
Exits nonzero if any gate fails.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

S0, RATE, T = 100.0, 0.05, 1.0
N = 252
TAIL = float(norm.cdf(-3.0))          # 1.3499e-3, the paper's clipping fraction
N_PATHS = 1_000_000
CHUNK = 250_000
KS = [90, 95, 100, 105, 110]
SIGMAS = [0.10, 0.15, 0.20, 0.30, 0.40]
SEED = 20260705

OUT = _REPO / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

GATES: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    GATES.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def cell(sigma: float, K: float, rng: np.random.Generator) -> tuple[float, float]:
    """Return (query ratio f_max(A)/f_max(R), min pathwise A-G) for one cell."""
    dt = T / N
    drift = (RATE - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    a_all, gap_all = [], []
    min_gap = math.inf
    done = 0
    while done < N_PATHS:
        m = min(CHUNK, N_PATHS - done)
        z = rng.standard_normal((m, N))
        logs = math.log(S0) + np.cumsum(drift + vol * z, axis=1)
        a = np.exp(logs).mean(axis=1)
        g = np.exp(logs.mean(axis=1))
        gap = a - g
        min_gap = min(min_gap, float(gap.min()))
        a_all.append(a.astype(np.float32))
        gap_all.append(gap.astype(np.float32))
        done += m
    a = np.concatenate(a_all)
    gap = np.concatenate(gap_all)
    fmax_a = float(np.quantile(a, 1 - TAIL)) - K
    fmax_r = float(np.quantile(gap, 1 - TAIL))
    return fmax_a / fmax_r, min_gap


def main() -> int:
    rng = np.random.default_rng(SEED)
    ratios = np.zeros((len(SIGMAS), len(KS)))
    min_gap_overall = math.inf
    for i, sigma in enumerate(SIGMAS):
        for j, K in enumerate(KS):
            r, mn = cell(sigma, float(K), rng)
            ratios[i, j] = r
            min_gap_overall = min(min_gap_overall, mn)
            print(f"  sigma={sigma:.2f} K={K:3d}: ratio={r:5.1f}  min(A-G)={mn:.2e}")

    base = ratios[SIGMAS.index(0.20), KS.index(100)]
    gate("base case reproduces daily ratio", abs(base - 16.0) < 1.0,
         f"ratio(K=100, sigma=0.20) = {base:.1f} (daily f_max ratio 16.0)")
    gate("Lemma 1 A>=G on every path/cell", min_gap_overall >= -1e-9,
         f"min(A-G) over all cells = {min_gap_overall:.2e}")
    lo_vol = ratios[SIGMAS.index(0.10), KS.index(100)]
    hi_vol = ratios[SIGMAS.index(0.40), KS.index(100)]
    gate("ratio decreases with volatility", lo_vol > hi_vol,
         f"ratio(sigma=0.10)={lo_vol:.1f} > ratio(sigma=0.40)={hi_vol:.1f}")
    gate("all ratios finite and > 1", bool(np.all(np.isfinite(ratios)) and np.all(ratios > 1)),
         f"range [{ratios.min():.1f}, {ratios.max():.1f}]")

    lines = [
        "v4 maximum-payoff query ratio f_max(A)/f_max(R) at k=1",
        f"daily Asian call, N={N}, {N_PATHS} paths/cell, clipping fraction {TAIL:.4e}",
        f"seed {SEED}",
        "",
        "rows = sigma, cols = K",
        "sigma\\K  " + "".join(f"{K:8d}" for K in KS),
    ]
    for i, sigma in enumerate(SIGMAS):
        lines.append(f"{sigma:7.2f}  " + "".join(f"{ratios[i, j]:8.1f}" for j in range(len(KS))))
    lines += [
        "",
        f"range: {ratios.min():.1f} (K={KS[int(np.argmin(ratios) % len(KS))]}, "
        f"sigma={SIGMAS[int(np.argmin(ratios) // len(KS))]}) to "
        f"{ratios.max():.1f} (K={KS[int(np.argmax(ratios) % len(KS))]}, "
        f"sigma={SIGMAS[int(np.argmax(ratios) // len(KS))]})",
        f"base case (K=100, sigma=0.20): {base:.1f}",
        "",
        "gates:",
    ]
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in GATES]
    (OUT / "linf_heatmap.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    im = ax.imshow(ratios, cmap="viridis", aspect="auto", origin="upper")
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])
    ax.set_yticks(range(len(SIGMAS)), [f"{s:.2f}" for s in SIGMAS])
    ax.set_xlabel("strike $K$")
    ax.set_ylabel(r"volatility $\sigma$")
    ax.set_title(r"Maximum-payoff ratio $f_{\max}(A)/f_{\max}(R)$ at $k=1$ "
                 f"($N={N}$)")
    thresh = 0.5 * (ratios.min() + ratios.max())
    for i in range(len(SIGMAS)):
        for j in range(len(KS)):
            ax.text(j, i, f"{ratios[i, j]:.1f}", ha="center", va="center",
                    color="white" if ratios[i, j] < thresh else "black", fontsize=10)
    fig.colorbar(im, label="oracle-query reduction at fixed price error")
    fig.tight_layout()
    fig.savefig(OUT / "linf_ratio_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"-> wrote {OUT}/linf_ratio_heatmap.png, {OUT}/linf_heatmap.txt")

    failed = [n for n, ok, _ in GATES if not ok]
    if failed:
        print(f"FAILED GATES: {failed}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
