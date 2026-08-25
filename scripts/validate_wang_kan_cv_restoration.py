"""Geometric-control restoration for the Wang-Kan control-variate hybrid.

Computes the exact continuous-time Heston price of the discretely monitored
geometric Asian option at the N = 256 Wang-Kan fixing dates via the affine
backward recursion (src/qc_option_pricing/classical/heston_geometric_asian.py),
cross-checks it against fine-substep continuous Heston Monte Carlo at the
production instances, and restores the hybrid arithmetic price

    V_hybrid = mu_G_exact + exp(-rT) E_weak[D]   (call; minus for put).

The recursion is an independent derivation of the same object as the
inaccessible Kim, Kim, Kim, Wee (2016) recursive method; its validation
chain (closed form vs ODE, Black-Scholes limit, parity, Monte Carlo) is in
tests/test_heston_geometric_asian.py.  The measured geometric-leg bias
mu_G_weak - mu_G_exact is exactly the model-bias component the hybrid
removes from the Wang-Kan raw estimand.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qc_option_pricing.classical.heston_geometric_asian import (
    HestonParams,
    geometric_asian_price,
)

ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY = ROOT / "results" / "v10" / "wang_kan_cv_feasibility.json"
OUT = ROOT / "results" / "v10" / "wang_kan_cv_restoration.json"

N_FIXINGS = 256
MC_PATHS = 500_000
MC_SUBSTEPS = 8
MC_SEED = 2026071861

INSTANCES = {
    "call_instance_1": {
        "params": HestonParams(s0=100.0, v0=0.1, rate=0.03, rho=-0.1,
                               kappa=2.0, theta=0.12, xi=0.3),
        "strike": 90.0, "option_type": "call",
    },
    "put_instance_2": {
        "params": HestonParams(s0=100.0, v0=0.05, rate=0.05, rho=-0.1,
                               kappa=2.0, theta=0.04, xi=0.2),
        "strike": 110.0, "option_type": "put",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mc_geometric_cross_check(params: HestonParams, strike: float,
                             option_type: str, seed: int) -> dict:
    """Full-truncation Euler with MC_SUBSTEPS substeps per fixing interval."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / (N_FIXINGS * MC_SUBSTEPS)
    sqrt_dt = math.sqrt(dt)
    rho_bar = math.sqrt(1.0 - params.rho**2)
    log_s = np.zeros(MC_PATHS)
    v = np.full(MC_PATHS, params.v0)
    sum_log = np.zeros(MC_PATHS)
    for _ in range(N_FIXINGS):
        for _ in range(MC_SUBSTEPS):
            z1 = rng.standard_normal(MC_PATHS)
            z2 = params.rho * z1 + rho_bar * rng.standard_normal(MC_PATHS)
            v_pos = np.maximum(v, 0.0)
            sq = np.sqrt(v_pos)
            log_s += (params.rate - 0.5 * v_pos) * dt + sq * sqrt_dt * z2
            v += (params.kappa * (params.theta - v_pos) * dt
                  + params.xi * sq * sqrt_dt * z1)
        sum_log += log_s
    g = params.s0 * np.exp(sum_log / N_FIXINGS)
    disc = math.exp(-params.rate)
    if option_type == "call":
        payoff = np.maximum(g - strike, 0.0)
    else:
        payoff = np.maximum(strike - g, 0.0)
    return {
        "paths": MC_PATHS, "substeps_per_interval": MC_SUBSTEPS, "seed": seed,
        "price": disc * float(payoff.mean()),
        "se": disc * float(payoff.std(ddof=1) / math.sqrt(MC_PATHS)),
        "negative_variance_fraction": float((v < 0).mean()),
    }


def main() -> None:
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    feas = json.loads(FEASIBILITY.read_text())
    result: dict = {
        "schema_version": "wang-kan-cv-restoration-v1",
        "created_at_start": started,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "numpy": np.__version__},
        "source_hashes": {
            "src/qc_option_pricing/classical/heston_geometric_asian.py": sha256(
                ROOT / "src" / "qc_option_pricing" / "classical"
                / "heston_geometric_asian.py"),
            "scripts/validate_wang_kan_cv_restoration.py": sha256(Path(__file__)),
            "tests/test_heston_geometric_asian.py": sha256(
                ROOT / "tests" / "test_heston_geometric_asian.py"),
            "results/v10/wang_kan_cv_feasibility.json": sha256(FEASIBILITY),
        },
        "method_note": (
            "affine backward-recursion derivation; independent of the "
            "inaccessible Kim et al. (2016) recursive method but computing "
            "the same discretely monitored geometric Asian price; validation "
            "chain in tests/test_heston_geometric_asian.py"
        ),
        "instances": {},
    }
    for name, inst in INSTANCES.items():
        params, strike = inst["params"], inst["strike"]
        option_type = inst["option_type"]
        disc = math.exp(-params.rate)
        formula = geometric_asian_price(params, strike, 1.0, N_FIXINGS,
                                        option_type)
        mc = mc_geometric_cross_check(params, strike, option_type, MC_SEED)
        mc_gap = formula["price"] - mc["price"]
        prod = feas["production"]["instances"][name]["pooled"]
        replicates = feas["production"]["instances"][name]["replicates"]
        mu_g_weak = float(np.mean(
            [r["geometric_price_discounted"] for r in replicates]))
        residual = prod["residual_mean_discounted"]
        residual_se = prod["residual_se"]
        sign = 1.0 if option_type == "call" else -1.0
        hybrid = formula["price"] + sign * residual
        raw_weak = prod["arithmetic_price_discounted"]
        geometric_leg_bias = mu_g_weak - formula["price"]
        result["instances"][name] = {
            "n_fixings": N_FIXINGS,
            "option_type": option_type,
            "strike": strike,
            "mu_g_exact_discrete_heston": formula["price"],
            "quad_error_estimate": formula["quad_error_estimate"],
            "expected_geometric_average_undiscounted": (
                formula["expected_geometric_average"]),
            "mc_cross_check": mc,
            "formula_minus_mc": mc_gap,
            "formula_minus_mc_in_se_units": mc_gap / mc["se"],
            "mu_g_weak_euler_1m_paths": mu_g_weak,
            "geometric_leg_weak_euler_bias": geometric_leg_bias,
            "residual_mean_discounted": residual,
            "residual_se": residual_se,
            "hybrid_price": hybrid,
            "hybrid_price_se": residual_se,
            "weak_euler_raw_price_8m_paths": raw_weak,
            "error_budget": {
                "amplitude_estimation": "set by regime; see resources JSON",
                "residual_statistical_se": residual_se,
                "clipping_bias_certificate": feas["production"]["instances"][
                    name]["residual_cap_study"]["selected"]["clipping_bias"],
                "restoration_mismatch_note": (
                    "E_weak[D] - E_exact[D] unmeasured directly; the residual "
                    "mean moved by less than 1.5e-3 across N in {64,...,512} "
                    "(refine part), and the geometric-leg bias eliminated by "
                    "the hybrid is measured above"),
            },
        }
        print(f"{name}:")
        print(f"  mu_G exact = {formula['price']:.6f} "
              f"(quad err {formula['quad_error_estimate']:.1e})")
        print(f"  MC cross-check {mc['price']:.6f} +/- {mc['se']:.6f} "
              f"(gap {mc_gap:+.6f} = {mc_gap/mc['se']:+.2f} SE)")
        print(f"  mu_G weak-Euler = {mu_g_weak:.6f} -> geometric-leg bias "
              f"{geometric_leg_bias:+.6f} (eliminated by hybrid)")
        print(f"  hybrid = {hybrid:.6f} +/- {residual_se:.6f} "
              f"vs weak-Euler raw {raw_weak:.6f}")
    result["created_at_end"] = datetime.now(
        timezone.utc).astimezone().isoformat(timespec="seconds")
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    tmp.replace(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
