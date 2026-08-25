"""MPS-simulate baseline vs improved reference-oracle configs (instance A).

Confirms the accuracy gains measured in scripts/v9_oracle_accuracy_sweeps.py
survive in the actual quantum circuit: the decoded circuit price equals the
exhaustive enumeration to float precision, and the improved configuration's
decoded price is far closer to the continuous truth.  The oracle source is
not modified; both configurations differ only in constructor parameters.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from qc_option_pricing.quantum.asian_oracle import (
    AsianGridSpec,
    build_asian_oracle,
    enumerate_encoded_asian,
    gauss_hermite_normal_grid,
    objective_probability_from_mps,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v9" / "oracle_circuit_check.json"

A = dict(n_dates=2, s0=2.0, strike=2.0, rate=0.05, volatility=0.30, maturity=1.0)


def exact_two_date_price(s0, strike, rate, sigma, maturity):
    """Closed-form inner expectation + adaptive 1D quadrature (near exact)."""
    dt = maturity / 2
    d = (rate - 0.5 * sigma * sigma) * dt
    s = sigma * math.sqrt(dt)

    def inner(z1):
        S1 = s0 * math.exp(d + s * z1)

        def f(z2):
            return 0.5 * S1 * (1.0 + math.exp(d + s * z2)) - strike

        if f(-40.0) >= 0.0:
            zstar = -40.0
        elif f(40.0) <= 0.0:
            return 0.0
        else:
            zstar = brentq(f, -40.0, 40.0, xtol=1e-14)
        a = 0.5 * S1 - strike
        b = 0.5 * S1 * math.exp(d)
        return a * norm.sf(zstar) + b * math.exp(0.5 * s * s) * norm.sf(zstar - s)

    val, _ = quad(lambda z1: inner(z1) * norm.pdf(z1), -12.0, 12.0,
                  limit=400, epsabs=1e-13, epsrel=1e-13)
    return math.exp(-rate * maturity) * val


truth = exact_two_date_price(A["s0"], A["strike"], A["rate"],
                             A["volatility"], A["maturity"])
print(f"continuous truth: {truth:.10f}")

# Moment-rematched probabilities for the rounded GH4 grid at shock_scale=4
gh4_pts, gh4_prob = gauss_hermite_normal_grid(2)
ss = 4
rp = tuple(round(z * ss) / ss for z in gh4_pts)
x = np.asarray(rp)
target = np.array([1.0, 0.0, 1.0, 0.0])  # E[Z^m], m=0..3
V = np.vander(x, 4, increasing=True).T
mm = np.linalg.solve(V, target)
assert np.all(mm > 0)
mm = tuple(float(v) for v in mm / mm.sum())
print(f"rounded GH4 points at ss={ss}: {rp}")
print(f"rematched probabilities: {tuple(round(p, 6) for p in mm)}")

configs = [
    ("baseline: binary +/-1, ss=1, ps=1",
     AsianGridSpec(shock_points=(-1.0, 1.0), shock_probabilities=(0.5, 0.5),
                   shock_scale=1, price_scale=1, **A)),
    ("improved: GH4+rematch, ss=4, ps=8",
     AsianGridSpec(shock_points=gh4_pts, shock_probabilities=mm,
                   shock_scale=ss, price_scale=8, **A)),
    ("improved+: GH4+rematch, ss=4, ps=32",
     AsianGridSpec(shock_points=gh4_pts, shock_probabilities=mm,
                   shock_scale=ss, price_scale=32, **A)),
]

results = {"truth": truth, "configs": []}
for name, spec in configs:
    ref = enumerate_encoded_asian(spec)
    for kind, method in (("raw", "rectangular_qrom"),
                         ("qcv", "factorized_arithmetic")):
        t0 = time.time()
        oracle = build_asian_oracle(spec, kind, residual_method=method)
        n_qubits = oracle.circuit.num_qubits
        p_circ, leakage = objective_probability_from_mps(oracle)
        elapsed = time.time() - t0
        p_ref = (ref.raw_objective_probability if kind == "raw"
                 else ref.qcv_objective_probability)
        decoded = oracle.post_process(p_circ)
        row = {"config": name, "kind": kind, "qubits": n_qubits,
               "p_circuit": p_circ, "p_reference": p_ref,
               "p_abs_err": abs(p_circ - p_ref), "work_leakage": leakage,
               "decoded_price": decoded, "price_error_vs_truth": decoded - truth,
               "sim_seconds": elapsed}
        results["configs"].append(row)
        print(f"\n{name} [{kind}] ({n_qubits} qubits, {elapsed:.1f}s)")
        print(f"  circuit p={p_circ:.12f}  reference p={p_ref:.12f}  "
              f"|diff|={abs(p_circ-p_ref):.2e}  leakage={leakage:.2e}")
        print(f"  decoded price={decoded:.6f}  error vs truth={decoded-truth:+.6f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=1))
print(f"\nwrote {OUT.relative_to(ROOT)}")
