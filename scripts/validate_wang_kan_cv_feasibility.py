"""Classical feasibility of the Wang-Kan weak-Euler control-variate hybrid.

Parts (each merges into results/v10/wang_kan_cv_feasibility.json):
  --smoke        20,000-path sanity run per instance
  --production   8 x 131,072 paths per instance with sample retention:
                 residual statistics, correlation, matched-clipping cap grids,
                 variance comparison, convergence table
  --verify       independent-seed 32 x 262,144-path raw-arm verification of
                 the same weak-Euler price (cent-agreement test)
  --refine       N in {64,128,256,512} residual-mean refinement study

Provenance is recorded during the run: a manifest is written before
computation and the part payload is merged atomically after success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qc_option_pricing.classical.heston_weak_euler import (
    HestonWeakEulerSpec,
    simulate_weak_euler_payoffs,
    variance_positivity_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v10" / "wang_kan_cv_feasibility.json"
MANIFEST = ROOT / "results" / "v10" / "wang_kan_cv_feasibility.manifest.json"

CALL_1 = HestonWeakEulerSpec(
    s0=100.0, v0=0.1, rate=0.03, rho=-0.1, kappa=2.0, theta=0.12, xi=0.3,
    maturity=1.0, n_steps=256, strike=90.0, option_type="call",
)
PUT_2 = HestonWeakEulerSpec(
    s0=100.0, v0=0.05, rate=0.05, rho=-0.1, kappa=2.0, theta=0.04, xi=0.2,
    maturity=1.0, n_steps=256, strike=110.0, option_type="put",
)
INSTANCES = {"call_instance_1": CALL_1, "put_instance_2": PUT_2}
Z_NORMALIZATION = {"call_instance_1": 200.0, "put_instance_2": 100.0}

CV_SEEDS = tuple(range(2026071801, 2026071809))
VERIFY_SEEDS = tuple(range(2026071811, 2026071843))
REFINE_SEEDS = (2026071851, 2026071852)
CLIPPING_BUDGET_DISCOUNTED = 0.001  # dollars, per handoff section 10 Regime B


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(command: str) -> dict:
    import numpy, scipy, qiskit  # noqa: F401  (versions only)
    import qiskit_aer

    return {
        "command": command,
        "cwd": os.getcwd(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
        },
        "source_hashes": {
            "src/qc_option_pricing/classical/heston_weak_euler.py": sha256(
                ROOT / "src" / "qc_option_pricing" / "classical" / "heston_weak_euler.py"),
            "scripts/validate_wang_kan_cv_feasibility.py": sha256(Path(__file__)),
            "tests/test_heston_weak_euler.py": sha256(
                ROOT / "tests" / "test_heston_weak_euler.py"),
        },
    }


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def begin_manifest(part: str, command: str) -> dict:
    manifest = {"part": part, "created_at_start": now(), **provenance(command)}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def merge_atomic(part: str, payload: dict, manifest: dict) -> None:
    payload = {**manifest, "created_at_end": now(), **payload}
    data = json.loads(OUT.read_text()) if OUT.exists() else {
        "schema_version": "wang-kan-cv-feasibility-v1"}
    data[part] = payload
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(OUT)
    print(f"merged part '{part}' into {OUT.relative_to(ROOT)}")


def cap_grid_study(samples: np.ndarray, discount: float, budget: float,
                   grid_quantiles: tuple[float, ...]) -> dict:
    """Smallest cap whose discounted clipping-bias upper 95% CI is in budget."""
    n = samples.size
    rows = []
    chosen = None
    for q in grid_quantiles:
        cap = float(np.quantile(samples, q))
        if cap <= 0.0:
            continue
        excess = np.maximum(samples - cap, 0.0)
        bias = discount * float(excess.mean())
        se = discount * float(excess.std(ddof=1) / math.sqrt(n))
        upper = bias + 1.645 * se
        row = {"quantile": q, "cap": cap, "clipping_bias": bias,
               "clipping_bias_se": se, "upper_90_ci": upper,
               "exceedance_probability": float((samples > cap).mean())}
        rows.append(row)
        if upper <= budget and chosen is None:
            chosen = row
    if chosen is None and rows:
        chosen = rows[-1]
    return {"budget_discounted": budget, "grid": rows, "selected": chosen}


def part_production(smoke: bool = False) -> None:
    label = "smoke" if smoke else "production"
    manifest = begin_manifest(label, " ".join(sys.argv))
    payload: dict = {"instances": {}}
    reps = 1 if smoke else len(CV_SEEDS)
    paths_per_rep = 20_000 if smoke else 131_072
    for name, spec in INSTANCES.items():
        disc = spec.discount
        rep_rows = []
        x_parts, d_parts = [], []
        for r in range(reps):
            out = simulate_weak_euler_payoffs(
                spec, paths=paths_per_rep, seed=CV_SEEDS[r], keep_samples=True)
            x_parts.append(out.pop("_raw_payoff_samples"))
            d_parts.append(out.pop("_residual_samples"))
            rep_rows.append({k: out[k] for k in (
                "seed", "arithmetic_price_discounted", "arithmetic_price_se",
                "geometric_price_discounted", "residual_mean_discounted",
                "residual_se", "payoff_variance", "residual_variance",
                "payoff_geometric_correlation", "minimum_average_gap",
                "minimum_residual", "minimum_variance",
                "negative_variance_count")})
        x = np.concatenate(x_parts)
        d = np.concatenate(d_parts)
        n = x.size
        z_bound = Z_NORMALIZATION[name]
        raw_caps = cap_grid_study(x, disc, CLIPPING_BUDGET_DISCOUNTED,
                                  (0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0))
        res_caps = cap_grid_study(d, disc, CLIPPING_BUDGET_DISCOUNTED,
                                  (0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0))
        cap_res = res_caps["selected"]["cap"]
        d_clipped = np.minimum(d, cap_res)
        convergence = []
        for k in range(12, int(math.log2(n)) + 1):
            m = 1 << k
            convergence.append({
                "paths": m,
                "raw_running_mean": disc * float(x[:m].mean()),
                "raw_se": disc * float(x[:m].std(ddof=1) / math.sqrt(m)),
                "residual_running_mean": disc * float(d[:m].mean()),
                "residual_se": disc * float(d[:m].std(ddof=1) / math.sqrt(m)),
            })
        payload["instances"][name] = {
            "spec": rep_rows[0] and {**out["spec"]},
            "z_normalization": z_bound,
            "paths_total": n,
            "seeds": CV_SEEDS[:reps],
            "replicates": rep_rows,
            "variance_certificate": variance_positivity_certificate(spec),
            "pooled": {
                "arithmetic_price_discounted": disc * float(x.mean()),
                "arithmetic_price_se": disc * float(x.std(ddof=1) / math.sqrt(n)),
                "residual_mean_discounted": disc * float(d.mean()),
                "residual_se": disc * float(d.std(ddof=1) / math.sqrt(n)),
                "payoff_sd_dollars": disc * float(x.std(ddof=1)),
                "residual_sd_dollars": disc * float(d.std(ddof=1)),
                "clipped_residual_sd_dollars": disc * float(d_clipped.std(ddof=1)),
                "variance_ratio_raw_over_residual": float(x.var(ddof=1) / d.var(ddof=1)),
                "variance_ratio_raw_over_clipped_residual": float(
                    x.var(ddof=1) / d_clipped.var(ddof=1)),
                "payoff_exceeds_z_probability": float((x > z_bound).mean()),
                "max_raw_payoff": float(x.max()),
                "max_residual": float(d.max()),
            },
            "raw_cap_study": raw_caps,
            "residual_cap_study": res_caps,
            "range_ratio_raw_z_over_selected_residual_cap": (
                z_bound / res_caps["selected"]["cap"]),
            "convergence": convergence,
        }
        print(f"[{label}] {name}: raw sd=${payload['instances'][name]['pooled']['payoff_sd_dollars']:.3f} "
              f"residual sd=${payload['instances'][name]['pooled']['residual_sd_dollars']:.3f} "
              f"variance ratio={payload['instances'][name]['pooled']['variance_ratio_raw_over_residual']:.1f} "
              f"cap ${res_caps['selected']['cap']:.3f} vs Z=${z_bound:.0f}")
    merge_atomic(label, payload, manifest)


def part_verify() -> None:
    manifest = begin_manifest("verify", " ".join(sys.argv))
    payload: dict = {"instances": {}}
    for name, spec in INSTANCES.items():
        disc = spec.discount
        estimates = []
        for seed in VERIFY_SEEDS:
            out = simulate_weak_euler_payoffs(
                spec, paths=262_144, seed=seed, keep_samples=False)
            estimates.append(out["arithmetic_price_discounted"])
        est = float(np.mean(estimates))
        se = float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
        payload["instances"][name] = {
            "seeds": VERIFY_SEEDS,
            "paths_total": 262_144 * len(VERIFY_SEEDS),
            "replicate_estimates": estimates,
            "arithmetic_price_discounted": est,
            "arithmetic_price_se_replicate": se,
        }
        print(f"[verify] {name}: price={est:.6f} +/- {se:.6f} "
              f"({262_144 * len(VERIFY_SEEDS):,} paths)")
    merge_atomic("verify", payload, manifest)


def part_refine() -> None:
    manifest = begin_manifest("refine", " ".join(sys.argv))
    payload: dict = {"instances": {}}
    for name, base in INSTANCES.items():
        rows = []
        for n_steps in (64, 128, 256, 512):
            spec = HestonWeakEulerSpec(
                s0=base.s0, v0=base.v0, rate=base.rate, rho=base.rho,
                kappa=base.kappa, theta=base.theta, xi=base.xi,
                maturity=base.maturity, n_steps=n_steps, strike=base.strike,
                option_type=base.option_type)
            per_seed = [simulate_weak_euler_payoffs(
                spec, paths=131_072, seed=s, keep_samples=False)
                for s in REFINE_SEEDS]
            rows.append({
                "n_steps": n_steps,
                "arithmetic_price_discounted": float(np.mean(
                    [o["arithmetic_price_discounted"] for o in per_seed])),
                "residual_mean_discounted": float(np.mean(
                    [o["residual_mean_discounted"] for o in per_seed])),
                "residual_se": float(np.mean(
                    [o["residual_se"] for o in per_seed]) / math.sqrt(len(per_seed))),
            })
            print(f"[refine] {name} N={n_steps}: "
                  f"price={rows[-1]['arithmetic_price_discounted']:.6f} "
                  f"residual={rows[-1]['residual_mean_discounted']:.6f}")
        payload["instances"][name] = {"seeds": REFINE_SEEDS, "paths_per_cell":
                                      131_072 * len(REFINE_SEEDS), "rows": rows}
    merge_atomic("refine", payload, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--production", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        part_production(smoke=True)
    elif args.production:
        part_production(smoke=False)
    elif args.verify:
        part_verify()
    else:
        part_refine()


if __name__ == "__main__":
    main()
