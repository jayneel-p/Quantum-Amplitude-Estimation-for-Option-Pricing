"""Range-vs-variance intuition figure (Black--Scholes daily Asian, k=1 control).

Shows, for one contract, that the beta=1 geometric residual R = (A_N-K)^+ - (G_N-K)^+
collapses the sampled VARIANCE far more than it collapses the encoded RANGE (cutoff):
the classical control gain and the quantum control gain are set by different quantities.

Fail-closed: asserts A_N >= G_N pathwise (AM--GM), and that the recovered cutoff ratio
matches the value reported in Section 6 (~16x) within tolerance.

Outputs:
  results/range_vs_variance.png
  results/range_vs_variance.json   (numbers + provenance)
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_option_pricing.classical.asian_mc import (
    _arith_avg,
    _geo_avg,
    geometric_asian_call_exact,
)
from qc_option_pricing.classical.gbm import gbm_path

# ----- contract (Section 6 daily base case) -----
S0 = K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0
N = 252
EXCEEDANCE = 1.35e-3           # matched clip fraction used for f_max in Section 6
TOTAL_PATHS = 2_000_000
CHUNK = 250_000
SEED = 20260720

ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "results" / "range_vs_variance.png"
OUT = ROOT / "results" / "range_vs_variance.json"


def simulate() -> dict:
    rng = np.random.default_rng(SEED)
    raw = np.empty(TOTAL_PATHS, dtype=np.float64)     # (A_N - K)^+
    ctrl = np.empty(TOTAL_PATHS, dtype=np.float64)    # (G_N - K)^+
    done = 0
    while done < TOTAL_PATHS:
        m = min(CHUNK, TOTAL_PATHS - done)
        paths = gbm_path(S0, R, SIGMA, T, N, m, rng=rng)
        a = _arith_avg(paths)
        g = _geo_avg(paths)
        raw[done:done + m] = np.maximum(a - K, 0.0)
        ctrl[done:done + m] = np.maximum(g - K, 0.0)
        done += m
    resid = raw - ctrl                                # beta=1 residual, >=0 by AM--GM

    # AM--GM / nonnegativity gate
    min_resid = float(resid.min())
    assert min_resid >= -1e-9, f"residual negativity {min_resid} violates AM--GM"

    q = 1.0 - EXCEEDANCE
    fmax_raw = float(np.quantile(raw, q))
    fmax_resid = float(np.quantile(resid, q))
    cutoff_ratio = fmax_raw / fmax_resid

    var_raw = float(raw.var(ddof=1))
    var_resid = float(resid.var(ddof=1))
    var_ratio = var_raw / var_resid

    # classical Kemna--Vorst (beta-optimal) estimator variance, for reference
    geo_exact = geometric_asian_call_exact(S0, K, R, SIGMA, T, N)
    cov = np.cov(raw, ctrl)
    beta_star = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
    kv_est = raw + beta_star * (geo_exact - ctrl)
    var_ratio_kv = var_raw / float(kv_est.var(ddof=1))

    corr = float(np.corrcoef(raw, ctrl)[0, 1])

    # cutoff ratio must reproduce the Section 6 value (~16)
    assert 13.0 <= cutoff_ratio <= 19.0, f"cutoff ratio {cutoff_ratio} off Section 6 (~16)"

    return dict(
        raw=raw, resid=resid,
        fmax_raw=fmax_raw, fmax_resid=fmax_resid, cutoff_ratio=cutoff_ratio,
        var_raw=var_raw, var_resid=var_resid, var_ratio=var_ratio,
        var_ratio_kv=var_ratio_kv, beta_star=float(beta_star),
        std_raw=math.sqrt(var_raw), std_resid=math.sqrt(var_resid),
        corr=corr,
    )


def make_figure(d: dict) -> None:
    plt.rcParams.update({"font.size": 12})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    raw, resid = d["raw"], d["resid"]
    c_raw, c_resid = "#c0392b", "#2e8b57"   # match house red / green

    # Panel A: raw payoff and beta=1 residual distributions ------------------
    hi = d["fmax_raw"] * 1.15
    bins = np.linspace(0.0, hi, 160)
    axA.hist(raw, bins=bins, color=c_raw, alpha=0.55,
             label=f"raw payoff $(A_N-K)^+$ (cutoff {d['fmax_raw']:.1f})")
    axA.hist(resid, bins=bins, color=c_resid, alpha=0.8,
             label=f"$\\beta{{=}}1$ residual $R$ (cutoff {d['fmax_resid']:.2f})")
    axA.axvline(d["fmax_raw"], color=c_raw, ls="--", lw=1.4)
    axA.axvline(d["fmax_resid"], color=c_resid, ls="--", lw=1.4)
    axA.set_yscale("log")
    axA.set_xlabel("undiscounted payoff (\\$)")
    axA.set_ylabel("path count")
    axA.set_title("(a) Raw payoff and $\\beta{=}1$ residual, daily BS Asian ($N{=}252$)")
    axA.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    axA.grid(True, which="major", alpha=0.25)

    # Panel B: variance vs range reduction, same beta=1 residual --------------
    vals = [d["var_ratio"], d["cutoff_ratio"]]   # beta=1: variance and range ratios
    axB.bar([0], [vals[0]], width=0.6, color="#3b6ea5", zorder=3,
            label=f"variance ratio  {vals[0]:,.0f}$\\times$ (classical MC gain)")
    axB.bar([1], [vals[1]], width=0.6, color="#7d3ac1", zorder=3,
            label=f"range ratio  {vals[1]:.1f}$\\times$ (bounded-amplitude QAE gain)")
    axB.set_yscale("log")
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["variance", "range / cutoff"])
    axB.set_ylabel("reduction factor (raw $/$ residual)")
    axB.set_title("(b) $\\beta{=}1$ residual: variance vs range reduction")
    axB.grid(True, which="major", axis="y", alpha=0.25, zorder=0)
    axB.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    axB.set_ylim(1, vals[0] * 4)

    fig.tight_layout()
    fig.savefig(PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    d = simulate()
    make_figure(d)
    record = {
        "figure": "range_vs_variance",
        "contract": dict(S0=S0, K=K, r=R, sigma=SIGMA, T=T, N=N,
                         exceedance_fraction=EXCEEDANCE),
        "paths": TOTAL_PATHS, "seed": SEED,
        "fmax_raw": d["fmax_raw"], "fmax_resid": d["fmax_resid"],
        "cutoff_ratio": d["cutoff_ratio"],
        "var_raw": d["var_raw"], "var_resid": d["var_resid"],
        "variance_ratio_beta1_residual": d["var_ratio"],
        "variance_ratio_kv_beta_optimal": d["var_ratio_kv"],
        "beta_star": d["beta_star"],
        "std_raw": d["std_raw"], "std_resid": d["std_resid"],
        "payoff_correlation": d["corr"],
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "platform": platform.platform()},
        "git_rev": git_rev(),
    }
    OUT.write_text(json.dumps(record, indent=2))
    print(f"cutoff ratio      : {d['cutoff_ratio']:.2f}x  (raw {d['fmax_raw']:.3f} / resid {d['fmax_resid']:.4f})")
    print(f"variance ratio b=1: {d['var_ratio']:.1f}x")
    print(f"variance ratio KV : {d['var_ratio_kv']:.1f}x  (beta*={d['beta_star']:.4f})")
    print(f"payoff corr       : {d['corr']:.5f}")
    print(f"wrote {PNG}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
