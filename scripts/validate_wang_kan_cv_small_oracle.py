"""Executable semantic-reference oracle for the weak-Euler Heston residual.

Builds a truth-table A-operator for N in {1, 2} steps of the Wang-Kan weak
Euler Heston model: Hadamards load the 4**N equiprobable shock pairs, and
one multi-controlled Ry per shock basis state encodes min(D, cap)/cap on an
objective qubit, where D is the control-variate residual computed by the
classical scalar reference.  Statevector simulation checks the objective
probability against exhaustive enumeration, unit probability mass, and the
A A^-1 roundtrip.

This is a SEMANTIC REFERENCE per the handoff, Section 13: it validates the
payoff semantics and reversibility of the encoding, not the fixed-point
arithmetic of the resource-counted construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector

from qc_option_pricing.classical.heston_weak_euler import (
    HestonWeakEulerSpec,
    enumerate_weak_euler,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v10" / "wang_kan_cv_small_oracle.json"

CALL = dict(s0=100.0, v0=0.1, rate=0.03, rho=-0.1, kappa=2.0, theta=0.12,
            xi=0.3, maturity=1.0, strike=90.0, option_type="call")
PUT = dict(s0=100.0, v0=0.05, rate=0.05, rho=-0.1, kappa=2.0, theta=0.04,
           xi=0.2, maturity=1.0, strike=110.0, option_type="put")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_semantic_oracle(spec: HestonWeakEulerSpec, cap: float):
    """Truth-table A-operator over the 4**N shock basis states."""
    reference = enumerate_weak_euler(spec)
    n_shock = 2 * spec.n_steps
    shock = QuantumRegister(n_shock, "shock")
    objective = QuantumRegister(1, "objective")
    circuit = QuantumCircuit(shock, objective, name="wk_cv_semantic_A")
    circuit.h(shock)
    expected = 0.0
    for index, row in enumerate(reference["paths"]):
        residual = min(row["residual"], cap)
        expected += residual / cap / reference["path_count"]
        if residual <= 0.0:
            continue
        theta = 2.0 * math.asin(math.sqrt(residual / cap))
        # basis index: bit 2j is a_j, bit 2j+1 is b_j, sign +1 encodes 1
        flipped = []
        for bit in range(n_shock):
            step, which = divmod(bit, 2)
            sign = row["signs"][step][which]
            if sign == -1:
                circuit.x(shock[bit])
                flipped.append(shock[bit])
        circuit.append(RYGate(theta).control(n_shock), [*shock, objective[0]])
        for qubit in reversed(flipped):
            circuit.x(qubit)
    return circuit, reference, expected


def check_instance(name: str, params: dict, n_steps: int, cap_quantile: float):
    spec = HestonWeakEulerSpec(n_steps=n_steps, **params)
    reference = enumerate_weak_euler(spec)
    residuals = [row["residual"] for row in reference["paths"]]
    cap = float(np.quantile(residuals, cap_quantile))
    if cap <= 0.0:
        cap = max(residuals)
    circuit, _, expected = build_semantic_oracle(spec, cap)
    state = Statevector.from_instruction(circuit)
    mass = float(np.sum(np.abs(state.data) ** 2))
    objective_bit = circuit.num_qubits - 1
    p_one = float(sum(abs(amp) ** 2 for i, amp in enumerate(state.data)
                      if (i >> objective_bit) & 1))
    # exact expected probability including the binding cap
    clipped = [min(r, cap) / cap for r in residuals]
    p_exact = float(np.mean(clipped))
    roundtrip = circuit.compose(circuit.inverse())
    zero_amp = Statevector.from_instruction(roundtrip).data[0]
    roundtrip_leak = 1.0 - float(abs(zero_amp) ** 2)
    binding = any(r > cap for r in residuals)
    row = {
        "instance": name,
        "n_steps": n_steps,
        "paths": reference["path_count"],
        "qubits": circuit.num_qubits,
        "cap": cap,
        "cap_binding_on_some_path": binding,
        "p_circuit": p_one,
        "p_exact": p_exact,
        "p_abs_err": abs(p_one - p_exact),
        "probability_mass_minus_one": mass - 1.0,
        "roundtrip_nonzero_probability": max(roundtrip_leak, 0.0),
        "minimum_residual": reference["minimum_residual"],
        "residual_expectation": reference["residual_expectation"],
    }
    ok = (row["p_abs_err"] <= 1e-10 and abs(row["probability_mass_minus_one"]) <= 1e-12
          and row["roundtrip_nonzero_probability"] <= 1e-12
          and row["minimum_residual"] >= 0.0)
    row["pass"] = bool(ok)
    print(f"  {name} N={n_steps}: p={p_one:.12f} exact={p_exact:.12f} "
          f"|err|={row['p_abs_err']:.2e} roundtrip={row['roundtrip_nonzero_probability']:.2e} "
          f"cap binding={binding} pass={ok}")
    return row


def degeneracy_check(name: str, params: dict) -> dict:
    """At N=1 the arithmetic and geometric means coincide, so the residual
    is identically zero on every path (AM-GM equality case)."""
    spec = HestonWeakEulerSpec(n_steps=1, **params)
    reference = enumerate_weak_euler(spec)
    residuals = [row["residual"] for row in reference["paths"]]
    ok = all(abs(r) <= 1e-14 for r in residuals)
    print(f"  {name} N=1 degeneracy: max |residual| = "
          f"{max(abs(r) for r in residuals):.2e} pass={ok}")
    return {"instance": name, "n_steps": 1, "check": "residual identically zero",
            "max_abs_residual": max(abs(r) for r in residuals), "pass": bool(ok)}


def main() -> None:
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    rows = []
    print("semantic-reference oracle checks:")
    rows.append(degeneracy_check("call_instance_1", CALL))
    rows.append(degeneracy_check("put_instance_2", PUT))
    rows.append(check_instance("call_instance_1", CALL, 2, 0.75))
    rows.append(check_instance("call_instance_1", CALL, 3, 0.75))
    rows.append(check_instance("put_instance_2", PUT, 2, 0.75))
    rows.append(check_instance("put_instance_2", PUT, 3, 0.75))
    result = {
        "schema_version": "wang-kan-cv-small-oracle-v1",
        "created_at_start": started,
        "created_at_end": datetime.now(
            timezone.utc).astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "source_hashes": {
            "scripts/validate_wang_kan_cv_small_oracle.py": sha256(Path(__file__)),
            "src/qc_option_pricing/classical/heston_weak_euler.py": sha256(
                ROOT / "src" / "qc_option_pricing" / "classical" / "heston_weak_euler.py"),
        },
        "claim_label": "semantic reference (handoff section 13); validates "
                       "payoff semantics, cap clipping, and reversibility, "
                       "not the fixed-point arithmetic construction",
        "checks": rows,
        "all_pass": all(r["pass"] for r in rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    tmp.replace(OUT)
    print(f"all_pass={result['all_pass']}; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
