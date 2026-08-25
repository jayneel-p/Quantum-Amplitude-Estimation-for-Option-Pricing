"""Three-way price validation: raw circuit, residual circuit plus control, enumeration.

The manuscript's control-free baseline was previously a ledger recount that
subtracted control modules from the residual estimate; no raw circuit existed.
``AsianGridSpec.geometric_leg='none'`` now builds one, so the same contract can
be priced three ways that must agree exactly:

  (a) raw oracle          geometric_leg='none', payoff_cap=None.  The threshold
                          comparator encodes the arithmetic payoff itself, so
                          decoding is the bare normalisation with no control
                          added back.
  (b) residual oracle     geometric_leg='collapsed', uncapped.  Decoding gives
                          the residual expectation; the exact dynamic-programming
                          geometric control is added back classically.
  (c) enumeration         every shock path of the same rounded integer model,
                          walked here with plain integers.

Decode method.  Every gate of ``A`` is H, X, CX or Toffoli and the H gates act
only on the shock and threshold registers, so fixing those registers makes the
rest of ``A`` a permutation of computational basis states.  The objective
probability is therefore exactly the fraction of prepared basis states whose
objective qubit ends at 1.  That is evaluated here on the *transpiled* circuit
with a bit-parallel simulator: one uint64 bit-plane per qubit, all
``2**(n*q + m)`` prepared basis states advanced together.  This is an exact
statevector decode of a permutation circuit, not a truncated approximation, and
it also reports whether any qubit is left dirty on any branch.  It is
cross-checked against qiskit-aer's MPS decode with ``--mps-crosscheck``, which
runs the same instances at S0=1 where MPS is tractable.

Artifact.  Every value printed below is also written to
results/v20/three_way_price_validation.json with full provenance, so the
manuscript's three-way price table traces to a file rather than to a console
transcript.  The record keeps the prices at full double precision and keeps the
two agreement gaps as measured, including the one-ulp N=3 raw-minus-enumeration
gap, rather than rounding them to zero.

Usage:  python scripts/v20_three_way_price_validation.py [--mps-crosscheck]
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import qiskit
from qiskit import transpile

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    arithmetic_objective_probability_from_mps,
    build_arithmetic_asian_model,
    build_arithmetic_asian_oracle,
    enumerate_arithmetic_asian,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v20" / "three_way_price_validation.json"

RATE = 0.05
VOLATILITY = 0.20
MATURITY = 1.0
PRICE_SCALE = 64
FRACTION_BITS = 14
ALL_ONES = np.uint64(0xFFFF_FFFF_FFFF_FFFF)

# Modules whose behaviour the recorded prices depend on.
HASHED_SOURCES = (
    "scripts/v20_three_way_price_validation.py",
    "src/qc_option_pricing/quantum/arithmetic_asian_oracle.py",
    "src/qc_option_pricing/quantum/asian_oracle.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_info() -> dict[str, object]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    rev = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "rev": rev.stdout.strip() if rev.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def ulps(gap: float, reference: float) -> float:
    """``gap`` measured in units in the last place of ``reference``."""

    return gap / math.ulp(reference) if gap else 0.0


def _spec(n_dates: int, s0: float, geometric_leg: str) -> AsianGridSpec:
    return AsianGridSpec(
        n_dates=n_dates,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=s0,
        strike=s0,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        shock_scale=1,
        price_scale=PRICE_SCALE,
        geometric_leg=geometric_leg,
    )


def _primitive_program(oracle) -> tuple[list[tuple[int, tuple[int, ...]]], int]:
    circuit = transpile(
        oracle.circuit, basis_gates=["h", "x", "cx", "ccx"], optimization_level=0
    )
    prepared = set(oracle.shock_qubits) | set(oracle.threshold_qubits)
    program: list[tuple[int, tuple[int, ...]]] = []
    hadamards = 0
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if name == "h":
            if qubits[0] not in prepared:
                raise AssertionError("H acts outside the prepared registers")
            hadamards += 1
            continue
        if name not in ("x", "cx", "ccx"):
            raise AssertionError(f"unexpected primitive {name}")
        program.append((len(qubits), qubits))
    if hadamards != len(prepared):
        raise AssertionError("state preparation is not one H per prepared qubit")
    return program, circuit.num_qubits


def _bit_plane(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8), bitorder="little").view(np.uint64)


def _popcount(row: np.ndarray) -> int:
    return int(np.unpackbits(row.view(np.uint8), bitorder="little").sum())


def exact_objective_probability(oracle) -> tuple[int, int, int, float]:
    """Return ``(fired, prepared_states, dirty_states, seconds)`` for the built A."""

    program, num_qubits = _primitive_program(oracle)
    prepared = [*oracle.shock_qubits, *oracle.threshold_qubits]
    batch = 1 << len(prepared)
    if batch % 64:
        raise ValueError("this decoder needs at least 64 prepared basis states")
    index = np.arange(batch, dtype=np.int64)
    planes = [_bit_plane((index >> position) & 1) for position in range(len(prepared))]
    state = np.zeros((num_qubits, batch // 64), dtype=np.uint64)
    for qubit, plane in zip(prepared, planes):
        state[qubit] = plane

    started = time.perf_counter()
    for arity, qubits in program:
        if arity == 1:
            state[qubits[0]] ^= ALL_ONES
        elif arity == 2:
            state[qubits[1]] ^= state[qubits[0]]
        else:
            state[qubits[2]] ^= state[qubits[0]] & state[qubits[1]]
    elapsed = time.perf_counter() - started

    dirty = np.zeros(batch // 64, dtype=np.uint64)
    for qubit in oracle.work_qubits:
        dirty |= state[qubit]
    for qubit, plane in zip(prepared, planes):
        dirty |= state[qubit] ^ plane
    return _popcount(state[oracle.objective_qubit]), batch, _popcount(dirty), elapsed


def enumerated_price(model) -> float:
    """Exact price of the rounded integer model, walked without the module."""

    spec = model.spec
    scale = 1 << model.multiplier_fraction_bits
    strike_sum = spec.n_dates * spec.strike_integer
    payoff_total = 0
    for digits in itertools.product((0, 1), repeat=spec.n_dates):
        price = model.initial_price
        total = 0
        for digit in digits:
            price = (price * model.price_factors[digit] + scale - 1) // scale
            total += price
        payoff_total += max(total - strike_sum, 0)
    mean_payoff = payoff_total / ((1 << spec.n_dates) * spec.n_dates * spec.price_scale)
    return math.exp(-RATE * MATURITY) * mean_payoff


def model_record(model, oracle, fired: int, batch: int, dirty: int,
                 seconds: float) -> dict:
    """Everything the artifact keeps about one built oracle."""

    spec = model.spec
    record = {
        "spec": dataclasses.asdict(spec),
        "multiplier_fraction_bits": model.multiplier_fraction_bits,
        "factor_scale": model.factor_scale,
        "price_factors": list(model.price_factors),
        "initial_price": model.initial_price,
        "initial_geometric": model.initial_geometric,
        "value_bits": model.value_bits,
        "multiplier_bits": model.multiplier_bits,
        "product_bits": model.product_bits,
        "geometric_product_bits": model.geometric_product_bits,
        "total_bits": model.total_bits,
        "residual_bits": model.residual_bits,
        "threshold_bits": model.threshold_bits,
        "requested_residual_cap_numerator": model.requested_residual_cap_numerator,
        "normalization_numerator": model.normalization_numerator,
        "normalization_dollars": model.normalization_dollars,
        "geometric_dp_peak_states": model.geometric_dp_peak_states,
        "circuit_qubits": oracle.circuit.num_qubits,
        "shock_qubits": len(oracle.shock_qubits),
        "threshold_qubits": len(oracle.threshold_qubits),
        "work_qubits": len(oracle.work_qubits),
        "prepared_basis_states": batch,
        "objective_states_fired": fired,
        "objective_probability": fired / batch,
        "dirty_basis_states": dirty,
        "decode_seconds": seconds,
    }
    if spec.geometric_leg != "none":
        record["geometric_control_undiscounted"] = model.geometric_control_undiscounted
    return record


def three_way(n_dates: int, s0: float) -> dict:
    print(
        f"=== N={n_dates}, S0=K={s0:g}, price_scale={PRICE_SCALE}, "
        f"multiplier_fraction_bits={FRACTION_BITS} ==="
    )
    discount = math.exp(-RATE * MATURITY)

    raw_model = build_arithmetic_asian_model(
        _spec(n_dates, s0, "none"), multiplier_fraction_bits=FRACTION_BITS
    )
    raw_oracle = build_arithmetic_asian_oracle(raw_model)
    raw_fired, raw_batch, raw_dirty, raw_seconds = exact_objective_probability(raw_oracle)
    if raw_dirty:
        raise AssertionError("the raw oracle left a dirty branch")
    raw_price = raw_oracle.post_process(raw_fired / raw_batch)

    cv_model = build_arithmetic_asian_model(
        _spec(n_dates, s0, "collapsed"), multiplier_fraction_bits=FRACTION_BITS
    )
    cv_oracle = build_arithmetic_asian_oracle(cv_model)
    cv_fired, cv_batch, cv_dirty, cv_seconds = exact_objective_probability(cv_oracle)
    if cv_dirty:
        raise AssertionError("the residual oracle left a dirty branch")
    cv_residual_price = discount * cv_model.normalization_dollars * (cv_fired / cv_batch)
    cv_control_price = discount * cv_model.geometric_control_undiscounted
    cv_price = cv_oracle.post_process(cv_fired / cv_batch)

    exact = enumerated_price(raw_model)
    module = discount * enumerate_arithmetic_asian(raw_model).arithmetic_payoff_undiscounted

    print(
        f"  raw       qubits={raw_oracle.circuit.num_qubits:>4} "
        f"threshold_bits={raw_model.threshold_bits:>3} "
        f"cap={raw_model.requested_residual_cap_numerator} "
        f"states={raw_batch} dirty={raw_dirty} decode={raw_seconds:.2f}s"
    )
    print(
        f"  residual  qubits={cv_oracle.circuit.num_qubits:>4} "
        f"threshold_bits={cv_model.threshold_bits:>3} "
        f"cap={cv_model.requested_residual_cap_numerator} "
        f"states={cv_batch} dirty={cv_dirty} decode={cv_seconds:.2f}s"
    )
    print(f"  (a) raw circuit                   {raw_price!r}")
    print(f"  (b) residual circuit              {cv_residual_price!r}")
    print(f"      exact DP geometric control  + {cv_control_price!r}")
    print(f"      total                         {cv_price!r}")
    print(f"  (c) classical enumeration         {exact!r}")
    print(f"  gap (a) - (c) = {raw_price - exact:+.3e}")
    print(f"  gap (b) - (c) = {cv_price - exact:+.3e}")
    if module != exact:
        raise AssertionError("module enumeration disagrees with the inline walk")
    for label, gap in (("a", raw_price - exact), ("b", cv_price - exact)):
        if abs(gap) > 1e-10:
            raise AssertionError(f"price ({label}) misses enumeration by {gap:.3e}")
    print()

    gap_raw = raw_price - exact
    gap_residual = cv_price - exact
    return {
        "n_dates": n_dates,
        "s0": s0,
        "strike": s0,
        "price_scale": PRICE_SCALE,
        "multiplier_fraction_bits": FRACTION_BITS,
        "rate": RATE,
        "volatility": VOLATILITY,
        "maturity": MATURITY,
        "discount_factor": discount,
        "raw_oracle": model_record(
            raw_model, raw_oracle, raw_fired, raw_batch, raw_dirty, raw_seconds
        ),
        "residual_oracle": model_record(
            cv_model, cv_oracle, cv_fired, cv_batch, cv_dirty, cv_seconds
        ),
        "prices": {
            "a_raw_circuit_discounted": raw_price,
            "b_residual_circuit_discounted": cv_residual_price,
            "b_exact_dp_geometric_control_discounted": cv_control_price,
            "b_restored_price_discounted": cv_price,
            "c_classical_enumeration_discounted": exact,
            "c_module_enumeration_discounted": module,
        },
        "gaps": {
            "note": "as measured, not rounded; the manuscript caption should be "
                    "read against these values rather than against 'zero in "
                    "double precision'",
            "a_raw_minus_enumeration": gap_raw,
            "a_raw_minus_enumeration_ulps": ulps(gap_raw, exact),
            "b_restored_minus_enumeration": gap_residual,
            "b_restored_minus_enumeration_ulps": ulps(gap_residual, exact),
            "a_minus_b": raw_price - cv_price,
            "module_enumeration_equals_inline_walk": module == exact,
            "double_precision_ulp_of_enumeration": math.ulp(exact),
        },
        "dirty_qubit_check": {
            "convention": "a prepared basis state is dirty if any work qubit is "
                          "left at 1, or any shock/threshold qubit is not restored "
                          "to its prepared value, after the full A",
            "raw_dirty_basis_states": raw_dirty,
            "residual_dirty_basis_states": cv_dirty,
            "clean": raw_dirty == 0 and cv_dirty == 0,
        },
    }


def mps_crosscheck(n_dates: int, s0: float) -> list[dict]:
    print(f"--- MPS cross-check of the bit-parallel decode, N={n_dates}, S0={s0:g} ---")
    rows = []
    for leg in ("none", "collapsed"):
        model = build_arithmetic_asian_model(
            _spec(n_dates, s0, leg), multiplier_fraction_bits=FRACTION_BITS
        )
        oracle = build_arithmetic_asian_oracle(model)
        fired, batch, dirty, _ = exact_objective_probability(oracle)
        mps, leakage = arithmetic_objective_probability_from_mps(oracle)
        print(
            f"  leg={leg:<9} bit_parallel={fired / batch!r} mps={mps!r} "
            f"diff={mps - fired / batch:+.3e} work_hamming={leakage:.3e} dirty={dirty}"
        )
        rows.append({
            "geometric_leg": leg,
            "n_dates": n_dates,
            "s0": s0,
            "bit_parallel_objective_probability": fired / batch,
            "mps_objective_probability": mps,
            "difference": mps - fired / batch,
            "work_register_hamming_leakage": leakage,
            "dirty_basis_states": dirty,
        })
    print()
    return rows


def main() -> None:
    created_at_start = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.perf_counter()

    crosscheck = None
    if "--mps-crosscheck" in sys.argv:
        crosscheck = mps_crosscheck(3, 1.0)
    cases = {
        "n_dates_3": three_way(3, 100.0),
        "n_dates_4": three_way(4, 100.0),
    }
    print("OK: raw circuit, residual circuit plus control, and enumeration agree.")

    artifact = {
        "schema_version": "three-way-price-validation-v1",
        "created_at_start": created_at_start,
        "command": "scripts/v20_three_way_price_validation.py",
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "purpose": "give the manuscript's three-way price table a traceable "
                   "artifact: the same at-the-money Asian call priced by the raw "
                   "oracle, by the residual oracle plus its exactly computed "
                   "geometric control, and by classical enumeration of the same "
                   "rounded integer model",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
        },
        "git": git_info(),
        "estimand": "discounted price of a discretely monitored at-the-money "
                    "arithmetic-average Asian call on a uniform binary +/-1 shock "
                    "tree, in the rounded fixed-point integer model the circuit "
                    "implements, not the continuous Black-Scholes model",
        "method": {
            "decode": "exact bit-parallel permutation decode of the transpiled A: "
                      "every gate is H, X, CX or Toffoli and the H gates act only "
                      "on the shock and threshold registers, so fixing those "
                      "registers makes the rest of A a permutation of "
                      "computational basis states; all 2**(n*q + m) prepared basis "
                      "states are advanced together as uint64 bit-planes and the "
                      "objective probability is the exact fraction that fire",
            "transpilation": "basis_gates=['h','x','cx','ccx'], optimization_level=0",
            "raw_leg": "geometric_leg='none', payoff_cap=None; the comparator "
                       "encodes the arithmetic payoff, so decoding is the bare "
                       "normalisation with no control added back",
            "residual_leg": "geometric_leg='collapsed', uncapped; decoding gives "
                            "the residual expectation and the exact "
                            "dynamic-programming geometric control is added back "
                            "classically",
            "enumeration": "every shock path of the same rounded integer model, "
                           "walked with plain Python integers, cross-checked "
                           "against enumerate_arithmetic_asian",
            "mps_crosscheck": "qiskit-aer MPS decode at S0=1, run only with "
                              "--mps-crosscheck",
        },
        "assertions": {
            "no_dirty_branch": "both oracles must leave every work qubit at zero "
                               "and restore every prepared qubit on every branch",
            "module_matches_inline_walk": "enumerate_arithmetic_asian must equal "
                                          "the inline integer walk exactly",
            "prices_match_enumeration": "|price - enumeration| <= 1e-10 for both "
                                        "the raw and the restored residual price",
            "all_passed": True,
        },
        "cases": cases,
        "mps_crosscheck": crosscheck,
        "source_hashes": {
            name: sha256(ROOT / name) for name in HASHED_SOURCES
        },
        "runtime_seconds": time.perf_counter() - started,
        "created_at_end": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
