#!/usr/bin/env python3
"""Verify the k=2 compositional ledger against a transpiled 252-date circuit.

The shipped k=1 oracle's resource formula is checked against a materialised
circuit at the production configuration.  This closes the same gap for the
two-block increment so both arms rest on the same class of evidence.

Writes results/v24/k2_transpile_check.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from qc_option_pricing.quantum.asian_oracle import AsianGridSpec  # noqa: E402
from qc_option_pricing.quantum.telescoping_asian_ladder import (  # noqa: E402
    build_k2_ladder_model,
    build_k2_ladder_oracle,
    estimate_k2_ladder_resources,
    primitive_counts_from_k2_ladder_circuit,
)

N_DATES = 252
PRICE_SCALE = 16_384
FRACTION_BITS = 30
K2_CAP_DOLLARS = 0.6808057512555804
OUTPUT = _REPO / "results" / "v24" / "k2_transpile_check.json"


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    spec = AsianGridSpec(
        n_dates=N_DATES,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        shock_scale=1,
        price_scale=PRICE_SCALE,
        geometric_leg="collapsed",
    )
    model = build_k2_ladder_model(
        spec,
        "blocked_to_target",
        multiplier_fraction_bits=FRACTION_BITS,
        increment_cap_dollars=K2_CAP_DOLLARS,
    )
    estimate = estimate_k2_ladder_resources(model)
    print(f"compositional: {estimate.a_counts.as_dict()}", flush=True)
    print(f"building the {N_DATES}-date circuit ...", flush=True)
    start = time.time()
    oracle = build_k2_ladder_oracle(model)
    built = time.time()
    print(f"built in {built - start:.1f}s, {oracle.circuit.num_qubits} qubits; "
          f"transpiling ...", flush=True)
    counted = primitive_counts_from_k2_ladder_circuit(oracle)
    done = time.time()
    print(f"transpiled and counted in {done - built:.1f}s", flush=True)

    matches = {
        "h": counted.h == estimate.a_counts.h,
        "x": counted.x == estimate.a_counts.x,
        "cx": counted.cx == estimate.a_counts.cx,
        "ccx": counted.ccx == estimate.a_counts.ccx,
        "qubits": oracle.circuit.num_qubits == estimate.a_qubits,
    }
    record = {
        "increment": "blocked_to_target",
        "n_dates": N_DATES,
        "price_scale": PRICE_SCALE,
        "multiplier_fraction_bits": FRACTION_BITS,
        "cap_dollars": K2_CAP_DOLLARS,
        "threshold_bits": model.threshold_bits,
        "basis_gates": ["h", "x", "cx", "ccx"],
        "optimization_level": 0,
        "toffoli_to_t_convention": "Toffoli = 7 T, 6 CX, 2 H (exact decomposition)",
        "compositional_formula": estimate.a_counts.as_dict(),
        "counted_circuit": counted.as_dict(),
        "compositional_a_qubits": estimate.a_qubits,
        "counted_a_qubits": oracle.circuit.num_qubits,
        "matches": matches,
        "all_match": all(matches.values()),
        "build_seconds": built - start,
        "transpile_and_count_seconds": done - built,
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
    for name, ok in matches.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"wrote {OUTPUT}")
    return 0 if record["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
