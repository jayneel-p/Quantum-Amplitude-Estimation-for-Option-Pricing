"""Consistent (grid-free, CI budget) recomputation of the Wang-Kan CV resource
and cap analysis, and regeneration of the two figures.

Both the raw payoff and the geometric residual are capped at the SAME discounted
clipping-bias budget ($0.001, upper-90% CI), grid-free, so the reported total-T
reduction measures the control variate rather than a change of normalisation.
The published-normalisation (Z) figures are reported separately as a disclosure.

Outputs:
  results/v12/wang_kan_consistent.json     numbers + provenance
  results/wang_kan_cap_tradeoff.png        regenerated (precise-CI cutoffs)
  results/wang_kan_matched_resources.png   regenerated (consistent ratios)
"""
from __future__ import annotations
import json, math, platform, subprocess, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_option_pricing.classical.heston_weak_euler import (
    HestonWeakEulerSpec, simulate_weak_euler_payoffs)
from qc_option_pricing.quantum.wang_kan_resources import MatchedComparison, n_oracle_queries

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/private/tmp/claude-501/-Users-jayneelparikh-Desktop-QC-Final-Project/"
             "2543dd5c-5e85-4290-b27a-1354a4621fb2/scratchpad")
OUT_JSON = ROOT / "results" / "v12" / "wang_kan_consistent.json"
CALL = HestonWeakEulerSpec(s0=100., v0=0.1, rate=0.03, rho=-0.1, kappa=2., theta=0.12,
                           xi=0.3, maturity=1., n_steps=256, strike=90., option_type="call")
PUT = HestonWeakEulerSpec(s0=100., v0=0.05, rate=0.05, rho=-0.1, kappa=2., theta=0.04,
                          xi=0.2, maturity=1., n_steps=256, strike=110., option_type="put")
SPECS = {"call": (CALL, 200.0, "Call instance"), "put": (PUT, 100.0, "Put instance")}
SEEDS = tuple(range(2026071801, 2026071809))
BUDGET = 0.001
res_json = json.load(open(ROOT / "results/v10/wang_kan_cv_resources.json"))["instances"]
KEYMAP = {"call": "call_instance_1", "put": "put_instance_2"}


def arrays(name, spec):
    c = CACHE / f"wk_arrays_{KEYMAP[name]}.npz"
    if c.exists():
        z = np.load(c); return z["x"], z["d"]
    xs, ds = [], []
    for s in SEEDS:
        o = simulate_weak_euler_payoffs(spec, paths=131072, seed=s, keep_samples=True)
        xs.append(o.pop("_raw_payoff_samples")); ds.append(o.pop("_residual_samples"))
    x, d = np.concatenate(xs), np.concatenate(ds)
    np.savez(c, x=x, d=d); return x, d


def precise_ci_cap(s, disc, budget):
    n = s.size; lo, hi = 0.0, float(s.max())
    for _ in range(70):
        m = 0.5 * (lo + hi); e = np.maximum(s - m, 0.0)
        loss = disc * (float(e.mean()) + 1.645 * float(e.std(ddof=1)) / math.sqrt(n))
        lo, hi = (m, hi) if loss > budget else (lo, m)
    return hi


def mean_clip_loss(s, disc, cap):
    return disc * float(np.maximum(s - cap, 0.0).mean())


def total_t_ratio(rr, rc, disc, tqr, tqc, regime):
    b = 1e-3 * disc * rr if regime == "A" else 0.003
    return MatchedComparison(b, 0.1, disc, rr, rc, tqr, tqc).row()


data = {}
for name, (spec, Z, _) in SPECS.items():
    disc = spec.discount
    x, d = arrays(name, spec)
    rj = res_json[KEYMAP[name]]["matched_comparison"]["regime_A_wang_kan_epsilon"]
    tqr = rj["cv_rotation"]["t_q_raw"]; tq_rot = rj["cv_rotation"]["t_q_cv"]; tq_thr = rj["cv_threshold"]["t_q_cv"]
    raw = precise_ci_cap(x, disc, BUDGET); resid = precise_ci_cap(d, disc, BUDGET)
    thr = 2.0 ** math.ceil(math.log2(resid))
    data[name] = dict(
        spec=name, Z=Z, disc=disc, raw_cap=raw, resid_cap=resid, thr_cap=thr,
        variance_ratio=float(x.var(ddof=1) / d.var(ddof=1)),
        correlation=float(np.corrcoef(np.maximum(x, 0), np.maximum(x - d, 0))[0, 1]),
        clip_loss_resid=mean_clip_loss(d, disc, resid),
        clip_loss_raw=mean_clip_loss(x, disc, raw),
        range_ratio_rotation=raw / resid, range_ratio_threshold=raw / thr,
        range_ratio_rotation_vsZ=Z / resid, range_ratio_threshold_vsZ=Z / thr,
        max_raw=float(x.max()),
        resource={f"{enc}_{reg}": total_t_ratio(raw, (resid if enc == "rot" else thr),
                                                disc, tqr, (tq_rot if enc == "rot" else tq_thr), reg)["total_t_ratio"]
                  for enc in ("rot", "thr") for reg in ("A", "B")},
        resource_vsZ={f"{enc}_A": total_t_ratio(Z, (resid if enc == "rot" else thr),
                                                disc, tqr, (tq_rot if enc == "rot" else tq_thr), "A")["total_t_ratio"]
                      for enc in ("rot", "thr")},
        _x=x, _d=d, _tq=(tqr, tq_rot, tq_thr))

# ---------------- figure 1: cap tradeoff (precise-CI cutoffs) ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (name, (spec, Z, title)) in zip(axes, SPECS.items()):
    dd = data[name]; disc = dd["disc"]; x, d = dd["_x"], dd["_d"]
    grid_raw = np.linspace(np.quantile(x, 0.99), x.max(), 40)
    grid_res = np.linspace(np.quantile(d, 0.99), d.max(), 40)
    bias_raw = [mean_clip_loss(x, disc, c) for c in grid_raw]
    bias_res = [mean_clip_loss(d, disc, c) for c in grid_res]
    ax.plot(grid_raw, bias_raw, "-o", color="#c0392b", ms=3, label="raw payoff")
    ax.plot(grid_res, bias_res, "-s", color="#2e8b57", ms=3, label="residual")
    ax.axhline(BUDGET, ls="--", color="0.4", lw=1.3, label="$\\$0.001$ clipping budget")
    ax.axvline(Z, ls=":", color="0.5", lw=1.2, label=f"published $Z={Z:.0f}$")
    ax.plot([dd["raw_cap"]], [mean_clip_loss(x, disc, dd["raw_cap"])], "*", color="#c0392b", ms=15, mec="k")
    ax.plot([dd["resid_cap"]], [mean_clip_loss(d, disc, dd["resid_cap"])], "*", color="#2e8b57", ms=15, mec="k")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cutoff (\\$)"); ax.set_title(title)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower left")
    if name == "call":
        ax.set_ylabel("discounted clipping bias (\\$)")
fig.tight_layout(); fig.savefig(ROOT / "results/wang_kan_cap_tradeoff.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# The matched-resources figure (wang_kan_matched_resources.png) is generated by
# scripts/figure_resource_vs_budget.py (ratio-vs-budget), which supersedes the
# earlier parallel-line total-T version.

# ---------------- provenance JSON ----------------
def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes()); return h.hexdigest()
for name in data:
    for k in ("_x", "_d", "_tq"):
        data[name].pop(k)
payload = {
    "schema": "wang-kan-consistent-v1",
    "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "convention": "raw and residual capped at the same $0.001 discounted clipping-bias budget (upper-90% CI), grid-free",
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform()},
    "source_hashes": {"scripts/recompute_wang_kan_consistent.py": sha(__file__),
                      "src/qc_option_pricing/quantum/wang_kan_resources.py":
                          sha(ROOT / "src/qc_option_pricing/quantum/wang_kan_resources.py")},
    "instances": data,
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(payload, indent=2))
for name, dd in data.items():
    print(f"{name}: raw_cap={dd['raw_cap']:.3f} resid_cap={dd['resid_cap']:.4f} thr_cap={dd['thr_cap']:.0f} "
          f"clip_loss_resid={dd['clip_loss_resid']:.2e} range_rot={dd['range_ratio_rotation']:.2f} "
          f"rot_A={dd['resource']['rot_A']:.2f} thr_A={dd['resource']['thr_A']:.2f} "
          f"vsZ_rot={dd['resource_vsZ']['rot_A']:.2f}")
print("wrote", OUT_JSON.relative_to(ROOT))
