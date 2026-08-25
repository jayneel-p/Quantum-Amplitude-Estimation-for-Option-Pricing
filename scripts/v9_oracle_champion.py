"""Executed-circuit validation of the recommended reference-oracle config.

Parts (run separately; each merges its results into the same JSON):
  champion_ps32  GH8+rematch, ss=8, ps=32 raw oracle, MPS (43 qubits)
  champion_ps64  GH8+rematch, ss=8, ps=64 raw oracle, MPS (45 qubits)
  capped         GH4+rematch, ss=4, ps=32 residual oracle with the
                 bias-budgeted cap actually binding, MPS (63 qubits)
  roundtrip      A then A^-1 leakage for the improved raw circuits
  statevector    dense Aer statevector cross-check of the MPS method

The oracle source is not modified; every configuration is constructor
arguments only.
"""

from __future__ import annotations

import argparse
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
    residual_cap_from_bias_budget,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v9" / "oracle_champion.json"

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


def rematched_grid(q, ss, clip_tolerance=1e-5):
    """GH points rounded to the integer grid with moment-rematched probs.

    Marginally negative solutions (about -6e-7 on one outer node at sixteen
    points) are clipped to zero and renormalised; anything more negative
    raises.
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


def merge(part, payload):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[part] = payload
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"merged part '{part}' into {OUT.relative_to(ROOT)}")


def run_mps_raw(spec, truth):
    ref = enumerate_encoded_asian(spec)
    t0 = time.time()
    oracle = build_asian_oracle(spec, "raw")
    p_circ, leakage = objective_probability_from_mps(oracle)
    elapsed = time.time() - t0
    decoded = oracle.post_process(p_circ)
    row = {"qubits": oracle.circuit.num_qubits,
           "p_circuit": p_circ,
           "p_reference": ref.raw_objective_probability,
           "p_abs_err": abs(p_circ - ref.raw_objective_probability),
           "work_leakage": leakage,
           "decoded_price": decoded,
           "price_error_vs_truth": decoded - truth,
           "sim_seconds": elapsed}
    print(json.dumps(row, indent=1))
    return row


def part_champion(ps, q=3):
    truth = exact_two_date_price(A["s0"], A["strike"], A["rate"],
                                 A["volatility"], A["maturity"])
    rp, prob = rematched_grid(q, 8)
    spec = AsianGridSpec(shock_points=rp, shock_probabilities=prob,
                         shock_scale=8, price_scale=ps, **A)
    row = run_mps_raw(spec, truth)
    row.update({"grid": f"GH{1 << q} rematched", "shock_scale": 8,
                "price_scale": ps, "points": rp, "probabilities": prob,
                "truth": truth})
    merge(f"champion_ps{ps}" if q == 3 else f"champion16_ps{ps}", row)


def part_capped():
    truth = exact_two_date_price(A["s0"], A["strike"], A["rate"],
                                 A["volatility"], A["maturity"])
    rp, prob = rematched_grid(2, 4)
    base = AsianGridSpec(shock_points=rp, shock_probabilities=prob,
                         shock_scale=4, price_scale=32, **A)
    budget = 1e-4
    cap = residual_cap_from_bias_budget(base, budget)
    spec = AsianGridSpec(shock_points=rp, shock_probabilities=prob,
                         shock_scale=4, price_scale=32,
                         residual_payoff_cap=cap, **A)
    ref = enumerate_encoded_asian(spec)
    disc = math.exp(-A["rate"] * A["maturity"])
    clip_bias = disc * (ref.residual_payoff_undiscounted
                        - ref.clipped_residual_payoff_undiscounted)
    t0 = time.time()
    oracle = build_asian_oracle(spec, "qcv",
                                residual_method="factorized_arithmetic")
    p_circ, leakage = objective_probability_from_mps(oracle)
    elapsed = time.time() - t0
    decoded = oracle.post_process(p_circ)
    row = {"qubits": oracle.circuit.num_qubits,
           "cap_dollars": oracle.cap_dollars,
           "requested_cap": cap,
           "budget": budget,
           "exact_clipping_bias": clip_bias,
           "p_circuit": p_circ,
           "p_reference": ref.qcv_objective_probability,
           "p_abs_err": abs(p_circ - ref.qcv_objective_probability),
           "work_leakage": leakage,
           "decoded_price": decoded,
           "decoded_price_plus_bias": decoded + clip_bias,
           "price_error_vs_truth": decoded - truth,
           "truth": truth,
           "sim_seconds": elapsed}
    print(json.dumps(row, indent=1))
    merge("capped", row)


def part_roundtrip():
    """A then A^-1 leakage via a per-qubit union bound.

    The module's `inverse_roundtrip_leakage_from_mps` saves one dense
    probability dictionary over every qubit, which is only tractable for the
    small baseline circuits.  This check composes the same A A^-1 circuit
    but bounds the nonzero probability by sum_q P(qubit q = 1), one
    single-qubit dictionary per qubit, exactly as
    `objective_probability_from_mps` does for work leakage.  The bound is
    zero exactly when the roundtrip is clean.
    """
    from qiskit_aer import AerSimulator

    rp, prob = rematched_grid(2, 4)
    for ps in (8, 32):
        spec = AsianGridSpec(shock_points=rp, shock_probabilities=prob,
                             shock_scale=4, price_scale=ps, **A)
        oracle = build_asian_oracle(spec, "raw")
        t0 = time.time()
        circuit = oracle.circuit.compose(oracle.circuit.inverse()).decompose(reps=8)
        for qubit in range(circuit.num_qubits):
            circuit.save_probabilities_dict([qubit], label=f"q_{qubit}")
        result = AerSimulator(method="matrix_product_state").run(
            circuit, shots=None).result()
        if not result.success:
            raise RuntimeError(f"MPS simulation failed: {result.status}")
        data = result.data(0)
        union_bound = sum(float(data[f"q_{qubit}"].get(1, 0.0))
                          for qubit in range(circuit.num_qubits))
        row = {"kind": "raw", "price_scale": ps,
               "qubits": oracle.circuit.num_qubits,
               "roundtrip_union_bound": max(0.0, union_bound),
               "sim_seconds": time.time() - t0}
        print(json.dumps(row))
        merge(f"roundtrip_raw_ps{ps}", row)


def part_statevector():
    """Dense Aer statevector vs MPS on the same decomposed circuit."""
    from qiskit_aer import AerSimulator

    truth = exact_two_date_price(A["s0"], A["strike"], A["rate"],
                                 A["volatility"], A["maturity"])
    rp, prob = rematched_grid(2, 4)
    for ps in (1, 2):
        spec = AsianGridSpec(shock_points=rp, shock_probabilities=prob,
                             shock_scale=4, price_scale=ps, **A)
        ref = enumerate_encoded_asian(spec)
        oracle = build_asian_oracle(spec, "raw")
        p_mps, _ = objective_probability_from_mps(oracle)
        circuit = oracle.circuit.decompose(reps=8)
        circuit.save_probabilities_dict([oracle.objective_qubit], label="objective")
        t0 = time.time()
        result = AerSimulator(method="statevector").run(circuit, shots=None).result()
        if not result.success:
            raise RuntimeError(f"statevector simulation failed: {result.status}")
        p_sv = float(result.data(0)["objective"].get(1, 0.0))
        row = {"price_scale": ps, "qubits": oracle.circuit.num_qubits,
               "p_statevector": p_sv, "p_mps": p_mps,
               "p_reference": ref.raw_objective_probability,
               "sv_vs_mps": abs(p_sv - p_mps),
               "sv_vs_reference": abs(p_sv - ref.raw_objective_probability),
               "decoded_price_sv": oracle.post_process(p_sv),
               "truth": truth,
               "sim_seconds": time.time() - t0}
        print(json.dumps(row, indent=1))
        merge(f"statevector_ps{ps}", row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=["champion_ps32", "champion_ps64",
                                 "champion16_ps32", "capped", "roundtrip",
                                 "statevector"])
    args = parser.parse_args()
    if args.part == "champion_ps32":
        part_champion(32)
    elif args.part == "champion_ps64":
        part_champion(64)
    elif args.part == "champion16_ps32":
        part_champion(32, q=4)
    elif args.part == "capped":
        part_capped()
    elif args.part == "roundtrip":
        part_roundtrip()
    else:
        part_statevector()


if __name__ == "__main__":
    main()
