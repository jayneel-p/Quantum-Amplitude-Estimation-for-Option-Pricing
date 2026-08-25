"""Validate the REAL non-QROM arithmetic residual oracle at increasing N via MPS.

This is the genuine Clifford+T arithmetic oracle (arithmetic_asian_oracle.py,
qrom_rows=0), not the QROM table-based reference (asian_oracle.py) that the
earlier champion runs used.  Extends the N=2 executable validation to larger N,
recording objective-probability error against exhaustive enumeration, work-
register cleanliness, and roundtrip leakage.  Writes incrementally so partial
results survive if a large N is killed.
"""
from __future__ import annotations
import json, time, traceback
from pathlib import Path

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    arithmetic_objective_probability_from_mps,
    arithmetic_roundtrip_leakage_from_mps,
    build_arithmetic_asian_model,
    build_arithmetic_asian_oracle,
    enumerate_arithmetic_asian,
    primitive_counts_from_circuit,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v9" / "arithmetic_oracle_scaleup.json"
rows = []

for N in range(2, 9):
    try:
        spec = AsianGridSpec(
            n_dates=N, shock_points=(-1.0, 1.0), shock_probabilities=(0.5, 0.5),
            s0=1.0, strike=0.0, rate=0.05, volatility=0.1, maturity=1.0,
            shock_scale=1, price_scale=1,
        )
        model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=1)
        ref = enumerate_arithmetic_asian(model)
        oracle = build_arithmetic_asian_oracle(model)
        try:
            qubits = int(oracle.circuit.num_qubits)
        except Exception:
            qubits = int(getattr(oracle, "num_qubits", -1))
        counts = primitive_counts_from_circuit(oracle)
        t0 = time.time()
        prob, work_ham = arithmetic_objective_probability_from_mps(oracle)
        leak = arithmetic_roundtrip_leakage_from_mps(oracle)
        dt = time.time() - t0
        err = abs(prob - ref.objective_probability)
        row = dict(
            n_dates=N, qubits=qubits, shocks="binary +/-1", qrom_rows=0,
            arbitrary_rotations=0, toffoli=getattr(counts, "toffoli", None),
            t=getattr(counts, "t", None),
            objective_probability=prob, reference_objective_probability=ref.objective_probability,
            absolute_probability_error=err, work_hamming_weight=work_ham,
            roundtrip_leakage=leak, seconds=dt,
        )
        rows.append(row)
        print(f"N={N}: {qubits}q  err={err:.2e}  work={work_ham:.2e}  leak={leak:.2e}  "
              f"T={getattr(counts, 't', None)}  {dt:.1f}s", flush=True)
        OUT.write_text(json.dumps(
            {"schema": "arithmetic-oracle-scaleup-v1",
             "note": "non-QROM Clifford+T arithmetic residual oracle, MPS-validated",
             "rows": rows}, indent=2))
    except Exception:
        print(f"N={N}: FAILED\n{traceback.format_exc()}", flush=True)
        break
print("done", flush=True)
