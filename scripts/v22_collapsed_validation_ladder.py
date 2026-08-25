"""Committed validation ladder for the collapsed geometric leg.

The manuscript's oracle is ``AsianGridSpec.geometric_leg='collapsed'``: the
geometric control is evaluated by accumulating the weighted shock sum
``s = sum_d (N - d) b_d`` into one small register and then applying one
controlled constant multiplication per bit of ``s``.  Its validation evidence
previously lived only in unit tests and in unrecorded scratch runs.  This
script produces the evidence as an artifact with provenance.

Five rungs, each recording exactly what was executed rather than inferred.

RUNG 1  Exhaustive enumeration of the integer model, N = 2..5.  An independent
        numpy path recurrence is compared with the module's own recurrence as
        integers, the residual is checked nonnegative, the control identity
        ``arithmetic = N * geometric + residual`` is checked, and the
        directed-rounding sandwich that makes the residual nonnegative
        (``total >= N * G * price_scale >= N * encoded_geometric``) is verified
        pathwise in 80-digit decimal against the exact real geometric mean.

RUNG 2  The TRANSPILED circuit against that mirror, same sizes.  Every gate of
        ``A`` is H, X, CX or Toffoli and the H gates act only on the shock and
        threshold registers, so fixing those registers makes the rest of ``A``
        a permutation of computational basis states.  All ``2**(N + m)``
        prepared basis states are therefore advanced together as uint64 bit
        planes; that is an exact decode of a permutation circuit, not a
        truncation.  The check is per shock path, not aggregate: the number of
        threshold values that fire must equal ``min(residual, cap)`` exactly as
        an integer.  Work-register cleanliness is checked on every branch.

RUNG 3  The struck contract at depth.  The published deep sweep
        (results/v9/arithmetic_oracle_scaleup.json) runs at K = 0, where both
        hinge selectors are constant across the superposition, so the strike
        hinge is never exercised at depth.  This rung runs struck, at-the-money
        contracts with an active geometric control and reports, for each hinge,
        how many paths its branch selector sends down the positive branch, which
        is what proves the hinge is genuinely two-sided.  It runs first at full
        residual resolution and then with a cap that narrows the threshold
        register and buys depth.

RUNG 4  Price decode at the finest quantisation that runs, for the collapsed
        leg and for the control-free 'none' leg, against the exact classical
        price of the same integer model.

RUNG 5  The ordering guarantee at production scale.  The residual encoding
        needs the encoded arithmetic sum never to fall below N times the
        encoded geometric value.  That follows from directed rounding, ceiling
        on the arithmetic side and floor on the geometric side, but the module
        rounds with a 32-ulp tolerance and never checks that the loaded
        constants actually landed on the required side.  Using the module's own
        constants at N = 252 and all three published precisions, every loaded
        constant is checked in 80-digit decimal against the exact real quantity
        it approximates, and the floored chain is checked against the exact
        geometric value on every one of the 31,879 reachable weighted sums.

Usage:  python scripts/v22_collapsed_validation_ladder.py [--only 1,2,5]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import qiskit
from qiskit import transpile

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    ArithmeticAsianOracle,
    _path_values,
    arithmetic_objective_probability_from_mps,
    build_arithmetic_asian_model,
    build_arithmetic_asian_oracle,
    enumerate_arithmetic_asian,
    estimate_arithmetic_asian_resources,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v20" / "collapsed_validation_ladder.json"
SCHEMA = "collapsed-validation-ladder-v1"

RATE = 0.05
VOLATILITY = 0.20
MATURITY = 1.0
S0 = 100.0
ALL_ONES = np.uint64(0xFFFF_FFFF_FFFF_FFFF)
DECIMAL_DIGITS = 80

getcontext().prec = DECIMAL_DIGITS


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------


def spec(
    n_dates: int,
    *,
    price_scale: int,
    leg: str,
    s0: float = S0,
    strike: float | None = None,
    residual_cap: float | None = None,
    payoff_cap: float | None = None,
) -> AsianGridSpec:
    return AsianGridSpec(
        n_dates=n_dates,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=s0,
        strike=s0 if strike is None else strike,
        rate=RATE,
        volatility=VOLATILITY,
        maturity=MATURITY,
        shock_scale=1,
        price_scale=price_scale,
        geometric_leg=leg,
        residual_payoff_cap=residual_cap,
        payoff_cap=payoff_cap,
    )


# --------------------------------------------------------------------------
# independent classical mirror
# --------------------------------------------------------------------------


def mirror(model) -> dict[str, np.ndarray]:
    """Re-derive every path from the model's integer constants, independently.

    This does not call the module's ``_path_values``; it rebuilds the same
    integer recurrence with vectorised numpy so that the two implementations can
    be compared as integers.  Path index ``i`` has shock bit ``b_d = (i >> d) &
    1``, which is the same convention the circuit's shock register uses.
    """

    grid = model.spec
    n = grid.n_dates
    fraction_bits = model.multiplier_fraction_bits
    scale = np.int64(1) << np.int64(fraction_bits)
    paths = 1 << n
    index = np.arange(paths, dtype=np.int64)

    def digit(date: int) -> np.ndarray:
        # one date at a time: holding all n digit vectors at once would cost
        # n * 2**n bytes, which at the deep end of rung 3 rivals the decoder.
        return ((index >> date) & 1).astype(np.int8)

    price = np.full(paths, model.initial_price, dtype=np.int64)
    total = np.zeros(paths, dtype=np.int64)
    factors = np.asarray(model.price_factors, dtype=np.int64)
    for date in range(n):
        price = (price * factors[digit(date)] + scale - 1) >> fraction_bits
        total += price

    leg = grid.geometric_leg
    weighted = np.zeros(paths, dtype=np.int64)
    if leg == "none":
        geometric = np.zeros(paths, dtype=np.int64)
    elif leg == "collapsed":
        for date in range(n):
            weighted += np.int64(n - date) * digit(date)
        geometric = np.full(paths, model.initial_geometric, dtype=np.int64)
        for bit, factor in enumerate(model.geometric_chain_factors):
            active = ((weighted >> bit) & 1).astype(bool)
            geometric = np.where(
                active, (geometric * np.int64(factor)) >> fraction_bits, geometric
            )
    else:
        geometric = np.full(paths, model.initial_geometric, dtype=np.int64)
        for date in range(n):
            row = np.asarray(model.geometric_factors[date], dtype=np.int64)
            geometric = (geometric * row[digit(date)]) >> fraction_bits

    strike_integer = grid.strike_integer
    arithmetic_payoff = np.maximum(total - n * strike_integer, 0)
    geometric_payoff = (
        np.zeros(paths, dtype=np.int64)
        if leg == "none"
        else np.maximum(geometric - strike_integer, 0)
    )
    residual = arithmetic_payoff - n * geometric_payoff
    encoded = arithmetic_payoff if leg == "none" else residual
    return {
        "weighted_sum": weighted,
        "total": total,
        "geometric": geometric,
        "arithmetic_payoff": arithmetic_payoff,
        "geometric_payoff": geometric_payoff,
        "residual": residual,
        "encoded": encoded,
        "expected_objective_count": np.minimum(
            encoded, model.requested_residual_cap_numerator
        ),
    }


def module_recurrence_agrees(model, values: dict[str, np.ndarray]) -> bool:
    """Compare the mirror with the module's own path recurrence, as integers."""

    n = model.spec.n_dates
    for digits in itertools.product((0, 1), repeat=n):
        # itertools orders the most significant date first; the mirror indexes
        # path i by b_d = (i >> d) & 1, so rebuild that index explicitly.
        index = sum(digit << date for date, digit in enumerate(digits))
        arithmetic, geometric, residual, total = _path_values(model, digits)
        if (
            total != values["total"][index]
            or arithmetic != values["arithmetic_payoff"][index]
            or geometric != values["geometric_payoff"][index]
            or residual != values["residual"][index]
        ):
            return False
    return True


def exact_integer_model_price(model, values: dict[str, np.ndarray]) -> float:
    """Discounted price of the rounded integer model, from the mirror alone."""

    grid = model.spec
    paths = 1 << grid.n_dates
    denominator = paths * grid.n_dates * grid.price_scale
    return math.exp(-RATE * MATURITY) * (
        int(values["arithmetic_payoff"].sum()) / denominator
    )


# --------------------------------------------------------------------------
# exact bit-parallel decode of the transpiled circuit
# --------------------------------------------------------------------------


def primitive_program(oracle) -> tuple[list[tuple[int, tuple[int, ...]]], int, dict]:
    """Transpile to H/X/CX/Toffoli and assert the permutation structure.

    The transpiled circuit object is released before returning so that it is not
    resident while the bit planes are allocated.
    """

    circuit = transpile(
        oracle.circuit, basis_gates=["h", "x", "cx", "ccx"], optimization_level=0
    )
    prepared = set(oracle.shock_qubits) | set(oracle.threshold_qubits)
    program: list[tuple[int, tuple[int, ...]]] = []
    counts = {"h": 0, "x": 0, "cx": 0, "ccx": 0}
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if name == "h":
            if qubits[0] not in prepared:
                raise AssertionError("H acts outside the prepared registers")
            counts["h"] += 1
            continue
        if name not in ("x", "cx", "ccx"):
            raise AssertionError(f"unexpected primitive {name}")
        counts[name] += 1
        program.append((len(qubits), qubits))
    if counts["h"] != len(prepared):
        raise AssertionError("state preparation is not one H per prepared qubit")
    num_qubits = int(circuit.num_qubits)
    del circuit
    gc.collect()
    return program, num_qubits, counts


def decode(
    program: list[tuple[int, tuple[int, ...]]],
    num_qubits: int,
    counts: dict,
    *,
    shock_qubits: tuple[int, ...],
    threshold_qubits: tuple[int, ...],
    work_qubits: tuple[int, ...],
    objective_qubit: int,
) -> dict:
    """Advance every prepared basis state through the transpiled circuit.

    Returns per-shock-path objective counts, the dirty-branch count, and timings.
    """

    n_shock = len(shock_qubits)
    n_threshold = len(threshold_qubits)
    prepared = [*shock_qubits, *threshold_qubits]
    batch = 1 << len(prepared)
    if batch % 64:
        raise ValueError("this decoder needs at least 64 prepared basis states")
    words = batch // 64
    index = np.arange(batch, dtype=np.int64)
    planes = [
        np.packbits(((index >> position) & 1).astype(np.uint8), bitorder="little").view(
            np.uint64
        )
        for position in range(len(prepared))
    ]
    del index
    state = np.zeros((num_qubits, words), dtype=np.uint64)
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
    seconds = time.perf_counter() - started

    dirty = np.zeros(words, dtype=np.uint64)
    for qubit in work_qubits:
        dirty |= state[qubit]
    for qubit, plane in zip(prepared, planes):
        dirty |= state[qubit] ^ plane
    dirty_states = int(np.unpackbits(dirty.view(np.uint8), bitorder="little").sum())
    objective = np.unpackbits(
        state[objective_qubit].view(np.uint8), bitorder="little"
    )
    # index = shock + 2**n_shock * threshold, so the C-order reshape is
    # [threshold, shock] and the column sum counts the firing thresholds.
    per_path = objective.reshape(1 << n_threshold, 1 << n_shock).sum(
        axis=0, dtype=np.int64
    )
    result = {
        "qubits": num_qubits,
        "primitive_counts": dict(counts),
        "primitive_gates_executed": len(program),
        "prepared_basis_states": batch,
        "shock_qubits": n_shock,
        "threshold_qubits": n_threshold,
        "dirty_states": dirty_states,
        "objective_fired": int(per_path.sum()),
        "per_path_counts": per_path,
        "decode_seconds": seconds,
        "uint64_word_operations": len(program) * words,
    }
    del state, planes, dirty, objective
    gc.collect()
    return result


SPOT_CHECK_SAMPLES = 12


def scalar_spot_check(
    program: list[tuple[int, tuple[int, ...]]],
    num_qubits: int,
    *,
    shock_qubits: tuple[int, ...],
    threshold_qubits: tuple[int, ...],
    work_qubits: tuple[int, ...],
    objective_qubit: int,
    expected: np.ndarray,
    samples: int = SPOT_CHECK_SAMPLES,
) -> dict:
    """Re-run selected basis states one at a time, with scalar bits.

    The bit-plane decoder packs 64 basis states per uint64 word, so a mistake in
    the packing or in the shock/threshold index convention would be invisible to
    a check that only compares against the same packing.  This walks a spread of
    individual basis states through the same primitive program with plain
    integers and checks the objective bit, the work registers and the prepared
    registers one state at a time.  The states are chosen by a fixed stride, so
    the check is deterministic and carries no seed.
    """

    prepared = [*shock_qubits, *threshold_qubits]
    batch = 1 << len(prepared)
    step = max(1, batch // samples)
    indices = sorted(
        {0, 1, batch - 2, batch - 1, *(position * step for position in range(samples))}
    )
    shock_mask = (1 << len(shock_qubits)) - 1
    started = time.perf_counter()
    for index in indices:
        state = [0] * num_qubits
        for position, qubit in enumerate(prepared):
            state[qubit] = (index >> position) & 1
        for arity, qubits in program:
            if arity == 1:
                state[qubits[0]] ^= 1
            elif arity == 2:
                state[qubits[1]] ^= state[qubits[0]]
            else:
                state[qubits[2]] ^= state[qubits[0]] & state[qubits[1]]
        path = index & shock_mask
        threshold_value = index >> len(shock_qubits)
        want = 1 if threshold_value < int(expected[path]) else 0
        if state[objective_qubit] != want:
            raise AssertionError(
                f"scalar spot check: basis state {index} gave objective "
                f"{state[objective_qubit]}, expected {want}"
            )
        if any(state[qubit] for qubit in work_qubits):
            raise AssertionError(f"scalar spot check: dirty work at state {index}")
        for position, qubit in enumerate(prepared):
            if state[qubit] != ((index >> position) & 1):
                raise AssertionError(
                    f"scalar spot check: prepared qubit moved at state {index}"
                )
    return {
        "basis_states_rechecked_one_at_a_time": len(indices),
        "scalar_spot_check_passed": True,
        "scalar_spot_check_seconds": time.perf_counter() - started,
    }


def circuit_case(model, *, values: dict[str, np.ndarray] | None = None) -> dict:
    """Build, transpile, decode, and check one model against its mirror."""

    started = time.perf_counter()
    oracle = build_arithmetic_asian_oracle(model)
    build_seconds = time.perf_counter() - started
    if values is None:
        values = mirror(model)
    program, num_qubits, counts = primitive_program(oracle)
    layout = {
        "shock_qubits": oracle.shock_qubits,
        "threshold_qubits": oracle.threshold_qubits,
        "work_qubits": oracle.work_qubits,
        "objective_qubit": oracle.objective_qubit,
    }
    del oracle
    gc.collect()
    decoded = decode(program, num_qubits, counts, **layout)
    expected = values["expected_objective_count"]
    spot = scalar_spot_check(program, num_qubits, expected=expected, **layout)
    del program
    gc.collect()
    decoded.update(spot)
    per_path = decoded.pop("per_path_counts")
    exact = bool(np.array_equal(per_path, expected))
    mismatches = int((per_path != expected).sum())
    ledger = estimate_arithmetic_asian_resources(model)
    decoded.update(
        {
            "build_seconds": build_seconds,
            "per_path_objective_count_matches_min_residual_cap_exactly": exact,
            "per_path_mismatches": mismatches,
            "work_registers_clean_on_every_branch": decoded["dirty_states"] == 0,
            "objective_probability": decoded["objective_fired"]
            / decoded["prepared_basis_states"],
            "executable_counts_match_compositional_ledger": (
                decoded["primitive_counts"]
                == {
                    "h": ledger.a_counts.h,
                    "x": ledger.a_counts.x,
                    "cx": ledger.a_counts.cx,
                    "ccx": ledger.a_counts.ccx,
                }
            ),
        }
    )
    if not exact:
        raise AssertionError(
            f"circuit disagrees with the mirror on {mismatches} shock paths"
        )
    if decoded["dirty_states"]:
        raise AssertionError(
            f"{decoded['dirty_states']} branches left a dirty work register"
        )
    return decoded


def model_summary(model) -> dict:
    grid = model.spec
    return {
        "n_dates": grid.n_dates,
        "geometric_leg": grid.geometric_leg,
        "s0": grid.s0,
        "strike": grid.strike,
        "price_units_per_dollar": grid.price_scale,
        "multiplier_fraction_bits": model.multiplier_fraction_bits,
        "value_bits": model.value_bits,
        "product_bits": model.product_bits,
        "geometric_product_bits": model.geometric_product_bits,
        "total_bits": model.total_bits,
        "threshold_bits": model.threshold_bits,
        "shock_weight_sum": model.shock_weight_sum,
        "shock_weight_bits": model.shock_weight_bits,
        "requested_cap_numerator": model.requested_residual_cap_numerator,
        "normalization_numerator": model.normalization_numerator,
        "cap_comparator_engaged": (
            model.requested_residual_cap_numerator < model.normalization_numerator
        ),
    }


def hinge_counts(model, values: dict[str, np.ndarray]) -> dict:
    """Count the branch each positive-part hinge selects, path by path.

    ``_append_positive_part`` forms ``input - subtract`` in two's complement and
    copies the result only when the sign bit is clear, so the branch selector is
    ``[input - subtract >= 0]``, not ``[payoff > 0]``.  The two differ exactly at
    a tie, and at K=0 they differ everywhere: the selector is constantly 1 while
    the payoff can still be zero.  Both are reported.
    """

    grid = model.spec
    paths = 1 << grid.n_dates
    strike_integer = grid.strike_integer
    raw = grid.geometric_leg == "none"
    arithmetic_selector = int(
        (values["total"] - grid.n_dates * strike_integer >= 0).sum()
    )
    geometric_selector = (
        None if raw else int((values["geometric"] - strike_integer >= 0).sum())
    )
    return {
        "paths": paths,
        "arithmetic_hinge_selects_positive_branch_on_paths": arithmetic_selector,
        "geometric_hinge_selects_positive_branch_on_paths": geometric_selector,
        "arithmetic_hinge_two_sided": 0 < arithmetic_selector < paths,
        "geometric_hinge_two_sided": (
            None if raw else bool(0 < geometric_selector < paths)
        ),
        "arithmetic_payoff_strictly_positive_paths": int(
            (values["arithmetic_payoff"] > 0).sum()
        ),
        "geometric_payoff_strictly_positive_paths": (
            None if raw else int((values["geometric_payoff"] > 0).sum())
        ),
    }


# --------------------------------------------------------------------------
# high-precision reference for the directed-rounding preconditions
# --------------------------------------------------------------------------


def decimal_reference(grid: AsianGridSpec, source: str) -> dict[str, Decimal]:
    """Exact real quantities the module's constants are supposed to bracket.

    ``module_doubles`` interprets the double-precision intermediates the module
    actually computed (``dt``, ``drift``, ``diffusion``) as exact rationals and
    does every later step exactly.  That is the reference the ordering proof
    needs, because AM--GM holds exactly for any consistent set of reals and the
    encoded model is the one those doubles define.  ``exact_base_parameters``
    instead derives ``dt``, ``drift`` and ``diffusion`` exactly from the
    double-precision inputs, which additionally absorbs their rounding.
    """

    n = grid.n_dates
    if source == "module_doubles":
        dt = Decimal(grid.maturity / n)
        drift = Decimal(grid.rate - 0.5 * grid.volatility**2)
        diffusion = Decimal(grid.volatility * math.sqrt(grid.maturity / n))
    elif source == "exact_base_parameters":
        dt = Decimal(grid.maturity) / Decimal(n)
        drift = Decimal(grid.rate) - Decimal(grid.volatility) ** 2 / 2
        diffusion = Decimal(grid.volatility) * dt.sqrt()
    else:
        raise ValueError(f"unknown decimal reference {source}")
    return {
        "dt": dt,
        "drift": drift,
        "diffusion": diffusion,
        "average_drift_time": dt * (Decimal(n) + 1) / 2,
    }


def exact_geometric_value(model, reference: dict[str, Decimal], weighted: int) -> Decimal:
    """``s0 * exp(drift * t_bar + diffusion * (low * W + (high - low) * s) / N)``."""

    grid = model.spec
    low, high = (Decimal(grid.shock_points[0]), Decimal(grid.shock_points[1]))
    n = Decimal(grid.n_dates)
    exponent = reference["drift"] * reference["average_drift_time"] + reference[
        "diffusion"
    ] * (low * Decimal(model.shock_weight_sum) + (high - low) * Decimal(weighted)) / n
    return Decimal(grid.s0) * exponent.exp()


def as_float(value: Decimal) -> float:
    return float(value)


# --------------------------------------------------------------------------
# rung 1
# --------------------------------------------------------------------------


RUNG12_PRICE_SCALE = 64
RUNG12_FRACTION_BITS = 14
RUNG12_SIZES = (2, 3, 4, 5)


def rung_one() -> dict:
    rows = []
    for n in RUNG12_SIZES:
        grid = spec(n, price_scale=RUNG12_PRICE_SCALE, leg="collapsed")
        model = build_arithmetic_asian_model(
            grid, multiplier_fraction_bits=RUNG12_FRACTION_BITS
        )
        values = mirror(model)
        paths = 1 << n

        agrees = module_recurrence_agrees(model, values)
        identity = np.array_equal(
            values["arithmetic_payoff"],
            n * values["geometric_payoff"] + values["residual"],
        )
        reference_module = enumerate_arithmetic_asian(model)

        # the defining structural property of the collapsed leg: the encoded
        # geometric value depends on the path only through the weighted sum
        collapses = all(
            len(set(values["geometric"][values["weighted_sum"] == weighted].tolist()))
            <= 1
            for weighted in range(model.shock_weight_sum + 1)
        )

        # directed-rounding sandwich, pathwise, in 80-digit decimal
        decimal_rows = {}
        for source in ("module_doubles", "exact_base_parameters"):
            reference = decimal_reference(grid, source)
            worst_arithmetic = None
            worst_geometric = None
            for weighted in range(model.shock_weight_sum + 1):
                selection = values["weighted_sum"] == weighted
                if not selection.any():
                    continue
                exact = exact_geometric_value(model, reference, weighted) * Decimal(
                    grid.price_scale
                )
                lowest_total = Decimal(int(values["total"][selection].min()))
                encoded_geometric = Decimal(int(values["geometric"][selection].max()))
                arithmetic_slack = lowest_total - Decimal(n) * exact
                geometric_slack = exact - encoded_geometric
                if worst_arithmetic is None or arithmetic_slack < worst_arithmetic:
                    worst_arithmetic = arithmetic_slack
                if worst_geometric is None or geometric_slack < worst_geometric:
                    worst_geometric = geometric_slack
            decimal_rows[source] = {
                "worst_total_minus_n_times_exact_geometric": as_float(worst_arithmetic),
                "worst_exact_geometric_minus_encoded_geometric": as_float(
                    worst_geometric
                ),
                "both_nonnegative": bool(
                    worst_arithmetic >= 0 and worst_geometric >= 0
                ),
            }

        rows.append(
            {
                "model": model_summary(model),
                "paths_enumerated": paths,
                "independent_mirror_matches_module_recurrence_exactly": bool(agrees),
                "control_identity_arithmetic_equals_n_geometric_plus_residual": bool(
                    identity
                ),
                "residual_nonnegative_on_every_path": bool(
                    (values["residual"] >= 0).all()
                ),
                "encoded_geometric_is_a_function_of_the_weighted_shock_sum_alone": bool(
                    collapses
                ),
                "minimum_residual_numerator": int(values["residual"].min()),
                "minimum_residual_numerator_over_in_the_money_paths": int(
                    values["residual"][values["arithmetic_payoff"] > 0].min()
                ),
                "maximum_residual_numerator": int(values["residual"].max()),
                "minimum_residual_dollars": int(values["residual"].min())
                / (n * grid.price_scale),
                "module_enumeration_minimum_residual_numerator": (
                    reference_module.minimum_residual_numerator
                ),
                "module_enumeration_maximum_residual_numerator": (
                    reference_module.maximum_residual_numerator
                ),
                "hinges": hinge_counts(model, values),
                "directed_rounding_sandwich_80_digit_decimal": decimal_rows,
            }
        )
        if not (agrees and identity and collapses and (values["residual"] >= 0).all()):
            raise AssertionError(f"rung 1 failed at N={n}")
        for source, row in decimal_rows.items():
            if not row["both_nonnegative"]:
                raise AssertionError(f"rung 1 sandwich failed at N={n} under {source}")
    return {
        "what_was_executed": (
            "exhaustive integer enumeration of every 2**N shock path of the "
            "collapsed model, compared with the module's own path recurrence and "
            "with an 80-digit decimal evaluation of the exact real geometric mean"
        ),
        "quantisation": {
            "price_units_per_dollar": RUNG12_PRICE_SCALE,
            "multiplier_fraction_bits": RUNG12_FRACTION_BITS,
            "note": (
                "coarser than the manuscript's 30-fraction-bit production "
                "encoding; chosen so that rungs 1 and 2 describe the same models"
            ),
        },
        "contract": "struck at the money, S0 = K = 100, uncapped residual",
        "sizes": list(RUNG12_SIZES),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# rung 2
# --------------------------------------------------------------------------


def mps_crosscheck_of_the_decoder() -> dict:
    """Check the bit-parallel decoder against a different simulation engine.

    The decoder's exactness follows from the asserted structure of ``A``, but the
    assertion is checked by this script's own transpile-and-extract step.  This
    runs qiskit-aer's matrix-product-state decode of the same built oracle at a
    size where MPS is tractable, so the two engines can disagree.  The instance
    is the one the module's own unit tests use for the collapsed leg, which has
    different market parameters from the rest of this ladder.
    """

    grid = AsianGridSpec(
        n_dates=2,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=3.0,
        strike=1.0,
        rate=0.03,
        volatility=0.4,
        maturity=1.25,
        shock_scale=1,
        price_scale=2,
        geometric_leg="collapsed",
    )
    model = build_arithmetic_asian_model(grid, multiplier_fraction_bits=3)
    values = mirror(model)
    case = circuit_case(model, values=values)
    oracle = build_arithmetic_asian_oracle(model)
    started = time.perf_counter()
    probability, leakage = arithmetic_objective_probability_from_mps(oracle)
    seconds = time.perf_counter() - started
    difference = probability - case["objective_probability"]
    return {
        "instance": (
            "the collapsed instance from tests/test_arithmetic_asian_oracle.py: "
            "N=2, S0=3, K=1, rate 0.03, volatility 0.4, maturity 1.25, "
            "price_scale=2, 3 multiplier fraction bits"
        ),
        "qubits": case["qubits"],
        "prepared_basis_states": case["prepared_basis_states"],
        "bit_parallel_objective_probability": case["objective_probability"],
        "matrix_product_state_objective_probability": probability,
        "difference": difference,
        "mps_expected_work_hamming_weight": leakage,
        "mps_seconds": seconds,
        "agreement_note": (
            "the residual difference is MPS truncation noise; the bit-parallel "
            "value is exact, being a count of basis states"
        ),
        "why_only_here": (
            "an MPS decode of the 281-qubit N=2 instance at this ladder's own "
            "quantisation did not complete within a 600-second trial, so the "
            "independent-engine cross-check is run at the tractable size only"
        ),
    }


def rung_two() -> dict:
    rows = []
    for n in RUNG12_SIZES:
        grid = spec(n, price_scale=RUNG12_PRICE_SCALE, leg="collapsed")
        model = build_arithmetic_asian_model(
            grid, multiplier_fraction_bits=RUNG12_FRACTION_BITS
        )
        values = mirror(model)
        case = circuit_case(model, values=values)
        case["model"] = model_summary(model)
        case["hinges"] = hinge_counts(model, values)
        rows.append(case)
    return {
        "what_was_executed": (
            "the transpiled H/X/CX/Toffoli circuit, advanced over every prepared "
            "basis state as uint64 bit planes; per shock path the number of "
            "threshold values that fire is compared with min(residual, cap) as an "
            "exact integer, and every work qubit and prepared qubit is checked on "
            "every branch"
        ),
        "decode_method": (
            "exact permutation decode of the transpiled circuit; not a truncated "
            "or sampled simulation"
        ),
        "quantisation": {
            "price_units_per_dollar": RUNG12_PRICE_SCALE,
            "multiplier_fraction_bits": RUNG12_FRACTION_BITS,
        },
        "sizes": list(RUNG12_SIZES),
        "rows": rows,
        "decoder_crosscheck_against_matrix_product_state": (
            mps_crosscheck_of_the_decoder()
        ),
    }


# --------------------------------------------------------------------------
# rung 3
# --------------------------------------------------------------------------


PUBLISHED_SWEEP_SIZES = (2, 3, 4, 5)
RUNG3_UNCAPPED = (6, 7, 8, 9, 10)
RUNG3_PRICE_SCALE = 8
RUNG3_FRACTION_BITS = 8
# (n_dates, cap numerator) with the cap numerator strictly below 2**threshold_bits
# so the capped comparator branch of the encoder is genuinely exercised.
RUNG3_CAPPED = ((12, 5), (14, 5), (16, 5), (18, 5), (20, 5), (22, 3), (23, 3))
RUNG3_PROJECTED = ((24, 3), (25, 3), (26, 3))


def machine_memory_bytes() -> int | None:
    try:
        return int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:  # noqa: BLE001 - provenance only
        return None


def published_k0_diagnostic() -> dict:
    """What the K=0 sweep in results/v9 actually exercises, computed classically."""

    rows = []
    for n in PUBLISHED_SWEEP_SIZES:
        grid = AsianGridSpec(
            n_dates=n,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.5, 0.5),
            s0=1.0,
            strike=0.0,
            rate=0.05,
            volatility=0.1,
            maturity=1.0,
            shock_scale=1,
            price_scale=1,
        )
        model = build_arithmetic_asian_model(grid, multiplier_fraction_bits=1)
        values = mirror(model)
        rows.append(
            {
                "n_dates": n,
                "geometric_leg": grid.geometric_leg,
                "hinges": hinge_counts(model, values),
                "paths_with_nonzero_geometric_control": int(
                    (values["geometric_payoff"] > 0).sum()
                ),
                "encoded_geometric_control_undiscounted": (
                    model.geometric_control_undiscounted
                ),
            }
        )
    collapsed_buildable = True
    collapsed_error = None
    try:
        build_arithmetic_asian_model(
            spec(5, price_scale=1, leg="collapsed", s0=1.0, strike=0.0),
            multiplier_fraction_bits=1,
        )
    except Exception as error:  # noqa: BLE001
        collapsed_buildable = False
        collapsed_error = str(error)
    return {
        "configuration": (
            "the published deep sweep, scripts/v13_arithmetic_oracle_scaleup.py -> "
            "results/v9/arithmetic_oracle_scaleup.json: per_date leg, S0=1, K=0, "
            "price_scale=1, 1 multiplier fraction bit"
        ),
        "finding": (
            "at K=0 both positive-part hinges select the same branch on every path, "
            "so neither hinge is exercised across the superposition; the geometric "
            "control is nonzero on a single path at this quantisation"
        ),
        "rows": rows,
        "collapsed_leg_buildable_at_that_quantisation": collapsed_buildable,
        "collapsed_leg_build_error": collapsed_error,
    }


def rung_three() -> dict:
    uncapped_rows = []
    for n in RUNG3_UNCAPPED:
        grid = spec(n, price_scale=RUNG3_PRICE_SCALE, leg="collapsed")
        model = build_arithmetic_asian_model(
            grid, multiplier_fraction_bits=RUNG3_FRACTION_BITS
        )
        values = mirror(model)
        clipped = int(
            (values["residual"] > model.requested_residual_cap_numerator).sum()
        )
        case = circuit_case(model, values=values)
        case["model"] = model_summary(model)
        case["hinges"] = hinge_counts(model, values)
        case["paths_clipped_by_the_cap"] = clipped
        uncapped_rows.append(case)
        print(
            f"  rung3 uncapped N={n}: qubits={case['qubits']} "
            f"states={case['prepared_basis_states']} "
            f"hinge={case['hinges']['arithmetic_hinge_selects_positive_branch_on_paths']}/"
            f"{case['hinges']['geometric_hinge_selects_positive_branch_on_paths']} "
            f"decode={case['decode_seconds']:.1f}s",
            flush=True,
        )

    capped_rows = []
    ceiling = None
    stopped_by = None
    for n, cap_numerator in RUNG3_CAPPED:
        cap = cap_numerator / (n * RUNG3_PRICE_SCALE)
        grid = spec(n, price_scale=RUNG3_PRICE_SCALE, leg="collapsed", residual_cap=cap)
        model = build_arithmetic_asian_model(
            grid, multiplier_fraction_bits=RUNG3_FRACTION_BITS
        )
        if model.requested_residual_cap_numerator != cap_numerator:
            raise AssertionError("the requested cap did not round to its numerator")
        values = mirror(model)
        clipped = int(
            (values["residual"] > model.requested_residual_cap_numerator).sum()
        )
        try:
            case = circuit_case(model, values=values)
        except MemoryError as error:  # pragma: no cover - machine dependent
            stopped_by = f"MemoryError at N={n}: {error}"
            break
        case["model"] = model_summary(model)
        case["hinges"] = hinge_counts(model, values)
        case["paths_clipped_by_the_cap"] = clipped
        case["residual_cap_dollars_of_the_average"] = cap
        capped_rows.append(case)
        ceiling = n
        print(
            f"  rung3 capped N={n}: qubits={case['qubits']} "
            f"states={case['prepared_basis_states']} "
            f"hinge={case['hinges']['arithmetic_hinge_selects_positive_branch_on_paths']}/"
            f"{case['hinges']['geometric_hinge_selects_positive_branch_on_paths']} "
            f"decode={case['decode_seconds']:.1f}s",
            flush=True,
        )

    projections = []
    for n, cap_numerator in RUNG3_PROJECTED:
        cap = cap_numerator / (n * RUNG3_PRICE_SCALE)
        model = build_arithmetic_asian_model(
            spec(n, price_scale=RUNG3_PRICE_SCALE, leg="collapsed", residual_cap=cap),
            multiplier_fraction_bits=RUNG3_FRACTION_BITS,
        )
        ledger = estimate_arithmetic_asian_resources(model)
        gates = ledger.a_counts.x + ledger.a_counts.cx + ledger.a_counts.ccx
        states = 1 << (n + model.threshold_bits)
        projections.append(
            {
                "n_dates": n,
                "threshold_bits": model.threshold_bits,
                "qubits": ledger.a_qubits,
                "prepared_basis_states": states,
                "bit_plane_bytes": ledger.a_qubits * states // 8,
                "projected_uint64_word_operations": gates * states // 64,
                "executed": False,
            }
        )

    top = capped_rows[-1] if capped_rows else None
    limit_note = (
        "the decoder holds one uint64 bit plane per qubit over all 2**(N + m) "
        "prepared basis states, so its resident footprint is qubits * 2**(N + m) / 8 "
        "bytes and its work is (primitive gates) * 2**(N + m) / 64 word operations.  "
        "Both double with each extra date at fixed threshold width m, so the ladder "
        "is stopped by memory before it is stopped by time.  The next rungs were not "
        "attempted; their footprints are projected below from the compositional "
        "ledger without building the circuits, and are to be read against "
        "machine_bytes_of_ram, on a machine that was concurrently running another "
        "job.  Narrowing the threshold register further would buy one or two more "
        "dates at the cost of an encoder that no longer resolves the residual."
    )
    return {
        "what_was_executed": (
            "struck at-the-money collapsed contracts with an active geometric "
            "control, decoded exactly over every prepared basis state, reporting "
            "for each hinge how many paths its branch selector sends down the "
            "positive branch"
        ),
        "published_k0_sweep_diagnostic": published_k0_diagnostic(),
        "quantisation": {
            "price_units_per_dollar": RUNG3_PRICE_SCALE,
            "multiplier_fraction_bits": RUNG3_FRACTION_BITS,
            "note": (
                "coarse by design: this rung buys circuit depth, not price "
                "accuracy.  The capped rows encode min(residual, cap) with a cap "
                "of a few encoded units, so most paths saturate and the decoded "
                "number is not a price.  What they establish is that the circuit "
                "computes that quantity exactly on every prepared basis state, "
                "with both hinges genuinely two-sided and every work qubit "
                "returned to zero, at depths the price-accurate configurations "
                "cannot reach.  The price check is rung 4."
            ),
        },
        "full_residual_resolution_uncapped": uncapped_rows,
        "capped_threshold_register": capped_rows,
        "ceiling_n_dates": ceiling,
        "ceiling_qubits": None if top is None else top["qubits"],
        "ceiling_prepared_basis_states": None
        if top is None
        else top["prepared_basis_states"],
        "not_executed_projections": projections,
        "machine_bytes_of_ram": machine_memory_bytes(),
        "what_stopped_it": stopped_by or limit_note,
    }


# --------------------------------------------------------------------------
# rung 4
# --------------------------------------------------------------------------


# (n_dates, price_scale, multiplier_fraction_bits)
RUNG4_CASES = ((2, 16384, 30), (3, 16384, 30), (4, 4096, 24))
RUNG4_PROJECTED = ((4, 16384, 30), (5, 16384, 30))


def rung_four() -> dict:
    rows = []
    for n, price_scale, fraction_bits in RUNG4_CASES:
        entry = {
            "n_dates": n,
            "price_units_per_dollar": price_scale,
            "multiplier_fraction_bits": fraction_bits,
            "legs": {},
        }
        for leg in ("collapsed", "none"):
            grid = spec(n, price_scale=price_scale, leg=leg)
            model = build_arithmetic_asian_model(
                grid, multiplier_fraction_bits=fraction_bits
            )
            values = mirror(model)
            case = circuit_case(model, values=values)
            # decode with the module's own convention.  post_process reads only
            # self.model, so it is called unbound on a stand-in holding the
            # model; that avoids keeping the transpiled circuit resident.
            oracle_price = ArithmeticAsianOracle.post_process(
                SimpleNamespace(model=model), case["objective_probability"]
            )
            replicated = math.exp(-RATE * MATURITY) * (
                model.normalization_dollars * case["objective_probability"]
                + (
                    0.0
                    if leg == "none"
                    else model.geometric_control_undiscounted
                )
            )
            if oracle_price != replicated:
                raise AssertionError("post_process disagrees with the decode formula")
            exact = exact_integer_model_price(model, values)
            # third route to the same number: the module's own enumerator, which
            # also re-checks the control identity and residual sign at this
            # precision, where rung 1 did not run.
            module_reference = enumerate_arithmetic_asian(model)
            module_price = math.exp(-RATE * MATURITY) * (
                module_reference.arithmetic_payoff_undiscounted
            )
            agrees = module_recurrence_agrees(model, values)
            if not agrees:
                raise AssertionError(
                    "the mirror and the module recurrence disagree at "
                    f"N={n}, price_scale={price_scale}"
                )
            if module_price != exact:
                raise AssertionError(
                    "the module enumerator and the mirror disagree on the price"
                )
            case["independent_mirror_matches_module_recurrence_exactly"] = agrees
            case["module_enumeration_price_discounted"] = module_price
            case["model"] = model_summary(model)
            case["hinges"] = hinge_counts(model, values)
            case["paths_clipped_by_the_cap"] = int(
                (values["encoded"] > model.requested_residual_cap_numerator).sum()
            )
            case["decoded_price_discounted"] = oracle_price
            case["exact_integer_model_price_discounted"] = exact
            case["absolute_gap"] = oracle_price - exact
            case["relative_gap"] = (oracle_price - exact) / exact
            case["geometric_control_added_back_undiscounted"] = (
                0.0 if leg == "none" else model.geometric_control_undiscounted
            )
            entry["legs"][leg] = case
            print(
                f"  rung4 N={n} ps={price_scale} fb={fraction_bits} leg={leg}: "
                f"qubits={case['qubits']} gap={case['absolute_gap']:+.3e} "
                f"decode={case['decode_seconds']:.1f}s",
                flush=True,
            )
        collapsed = entry["legs"]["collapsed"]
        raw = entry["legs"]["none"]
        entry["residual_minus_raw_price"] = (
            collapsed["decoded_price_discounted"] - raw["decoded_price_discounted"]
        )
        rows.append(entry)

    projections = []
    for n, price_scale, fraction_bits in RUNG4_PROJECTED:
        model = build_arithmetic_asian_model(
            spec(n, price_scale=price_scale, leg="collapsed"),
            multiplier_fraction_bits=fraction_bits,
        )
        ledger = estimate_arithmetic_asian_resources(model)
        gates = ledger.a_counts.x + ledger.a_counts.cx + ledger.a_counts.ccx
        states = 1 << (n + model.threshold_bits)
        projections.append(
            {
                "n_dates": n,
                "price_units_per_dollar": price_scale,
                "multiplier_fraction_bits": fraction_bits,
                "geometric_leg": "collapsed",
                "threshold_bits": model.threshold_bits,
                "qubits": ledger.a_qubits,
                "prepared_basis_states": states,
                "bit_plane_bytes": ledger.a_qubits * states // 8,
                "projected_uint64_word_operations": gates * states // 64,
                "executed": False,
            }
        )

    return {
        "what_was_executed": (
            "the price decoded from the exactly computed objective probability of "
            "the transpiled circuit, against the exact classical price of the same "
            "rounded integer model, for the collapsed residual leg and for the "
            "control-free raw leg"
        ),
        "finest_quantisation_executed": {
            "price_units_per_dollar": 16384,
            "multiplier_fraction_bits": 30,
            "note": (
                "this is the manuscript's production encoding precision, executed "
                "here at N=2 and N=3 only; the 252-date circuit is not materialised"
            ),
        },
        "rows": rows,
        "not_executed_projections": projections,
        "what_stopped_it": (
            "the same bit-plane footprint as rung 3.  At the production precision "
            "the threshold register is 21 bits wide at N=3, so each extra date "
            "doubles a decode that already holds 1.2 GB of bit planes; the first "
            "unexecuted case is listed below with its projected footprint."
        ),
    }


# --------------------------------------------------------------------------
# rung 5
# --------------------------------------------------------------------------


RUNG5_N_DATES = 252
RUNG5_RESIDUAL_CAP = 2.864
# (price_scale, multiplier_fraction_bits) as published in results/v8
RUNG5_PRECISIONS = ((1024, 18), (4096, 24), (16384, 30))


def rung_five() -> dict:
    rows = []
    for price_scale, fraction_bits in RUNG5_PRECISIONS:
        grid = spec(
            RUNG5_N_DATES,
            price_scale=price_scale,
            leg="collapsed",
            residual_cap=RUNG5_RESIDUAL_CAP,
        )
        model = build_arithmetic_asian_model(
            grid, multiplier_fraction_bits=fraction_bits
        )
        two_fraction = Decimal(1 << fraction_bits)
        scale = Decimal(price_scale)
        s0 = Decimal(grid.s0)

        # the module's own floored chain, recomputed here on every reachable
        # weighted sum in increasing order of the sum
        chain = [model.initial_geometric]
        for factor in model.geometric_chain_factors:
            chain = (
                chain + [(value * factor) >> fraction_bits for value in chain]
            )[: model.shock_weight_sum + 1]
        if len(chain) != model.shock_weight_sum + 1:
            raise AssertionError("weighted-sum enumeration lost a reachable state")

        # the module rounds with _ceil_stable / _floor_stable, which shift the
        # argument by 32 ulps before rounding.  Recording that tolerance in the
        # same units shows how much of the observed margin it could have eaten.
        dt_float = grid.maturity / grid.n_dates
        drift_float = grid.rate - 0.5 * grid.volatility**2
        diffusion_float = grid.volatility * math.sqrt(dt_float)
        factor_scale = float(1 << fraction_bits)
        pre_rounding = {
            "initial_price": grid.s0 * grid.price_scale,
            "price_factors": [
                math.exp(drift_float * dt_float + diffusion_float * shock)
                * factor_scale
                for shock in grid.shock_points
            ],
            "initial_geometric": grid.s0
            * math.exp(
                drift_float * dt_float * (grid.n_dates + 1) / 2.0
                + diffusion_float
                * grid.shock_points[0]
                * model.shock_weight_sum
                / grid.n_dates
            )
            * grid.price_scale,
            "chain_factors": [
                math.exp(
                    diffusion_float
                    * (grid.shock_points[1] - grid.shock_points[0])
                    / grid.n_dates
                    * (1 << bit)
                )
                * factor_scale
                for bit in range(model.shock_weight_bits)
            ],
        }
        tolerances = {
            "initial_price": 32.0 * math.ulp(max(1.0, pre_rounding["initial_price"])),
            "price_factors": [
                32.0 * math.ulp(max(1.0, value))
                for value in pre_rounding["price_factors"]
            ],
            "initial_geometric": 32.0
            * math.ulp(max(1.0, pre_rounding["initial_geometric"])),
            "chain_factors": [
                32.0 * math.ulp(max(1.0, value))
                for value in pre_rounding["chain_factors"]
            ],
        }

        references = {}
        for source in ("module_doubles", "exact_base_parameters"):
            reference = decimal_reference(grid, source)
            price_factor_slacks = [
                Decimal(model.price_factors[position])
                - (
                    reference["drift"] * reference["dt"]
                    + reference["diffusion"] * Decimal(shock)
                ).exp()
                * two_fraction
                for position, shock in enumerate(grid.shock_points)
            ]
            initial_price_slack = Decimal(model.initial_price) - s0 * scale
            low, high = (
                Decimal(grid.shock_points[0]),
                Decimal(grid.shock_points[1]),
            )
            alpha = reference["diffusion"] * (high - low) / Decimal(grid.n_dates)
            chain_factor_slacks = [
                (alpha * Decimal(1 << bit)).exp() * two_fraction - Decimal(factor)
                for bit, factor in enumerate(model.geometric_chain_factors)
            ]
            initial_geometric_slack = exact_geometric_value(
                model, reference, 0
            ) * scale - Decimal(model.initial_geometric)

            started = time.perf_counter()
            worst_slack = None
            worst_state = None
            worst_relative = None
            for weighted in range(model.shock_weight_sum + 1):
                exact = exact_geometric_value(model, reference, weighted) * scale
                slack = exact - Decimal(chain[weighted])
                if worst_slack is None or slack < worst_slack:
                    worst_slack = slack
                    worst_state = weighted
                    worst_relative = slack / exact
            sweep_seconds = time.perf_counter() - started

            references[source] = {
                "initial_price_ceiling_slack": as_float(initial_price_slack),
                "initial_price_ceiling_holds": bool(initial_price_slack >= 0),
                "price_factor_ceiling_slacks": [
                    as_float(value) for value in price_factor_slacks
                ],
                "price_factor_ceiling_worst_slack": as_float(min(price_factor_slacks)),
                "price_factor_ceiling_holds": bool(min(price_factor_slacks) >= 0),
                "initial_geometric_floor_slack": as_float(initial_geometric_slack),
                "initial_geometric_floor_holds": bool(initial_geometric_slack >= 0),
                "chain_factor_floor_slacks": [
                    as_float(value) for value in chain_factor_slacks
                ],
                "chain_factor_floor_worst_slack": as_float(min(chain_factor_slacks)),
                "chain_factor_floor_holds": bool(min(chain_factor_slacks) >= 0),
                "chain_sweep_states_checked": model.shock_weight_sum + 1,
                "chain_sweep_worst_slack_encoded_units": as_float(worst_slack),
                "chain_sweep_worst_slack_relative": as_float(worst_relative),
                "chain_sweep_worst_state": worst_state,
                "chain_sweep_holds": bool(worst_slack >= 0),
                "chain_sweep_seconds": sweep_seconds,
            }
            # s0 * price_scale is exactly representable here, so the initial
            # price is an exact tie rather than a margin; the transcendental
            # constants are the ones the 32-ulp tolerance could have flipped.
            margins = [
                *(
                    as_float(slack) / tolerance
                    for slack, tolerance in zip(
                        price_factor_slacks, tolerances["price_factors"]
                    )
                ),
                as_float(initial_geometric_slack) / tolerances["initial_geometric"],
                *(
                    as_float(slack) / tolerance
                    for slack, tolerance in zip(
                        chain_factor_slacks, tolerances["chain_factors"]
                    )
                ),
            ]
            references[source][
                "narrowest_transcendental_margin_in_units_of_the_32_ulp_tolerance"
            ] = min(margins)
            references[source]["initial_price_is_an_exact_tie"] = bool(
                initial_price_slack == 0
            )
            references[source]["directed_rounding_tolerance_units"] = {
                "initial_price": tolerances["initial_price"],
                "price_factors": tolerances["price_factors"],
                "initial_geometric": tolerances["initial_geometric"],
                "chain_factors": tolerances["chain_factors"],
            }
            for key in (
                "initial_price_ceiling_holds",
                "price_factor_ceiling_holds",
                "initial_geometric_floor_holds",
                "chain_factor_floor_holds",
                "chain_sweep_holds",
            ):
                if not references[source][key]:
                    raise AssertionError(
                        f"rung 5 precondition {key} failed at "
                        f"price_scale={price_scale} under {source}"
                    )
        rows.append(
            {
                "model": model_summary(model),
                "residual_payoff_cap_dollars": RUNG5_RESIDUAL_CAP,
                "geometric_chain_factors": list(model.geometric_chain_factors),
                "price_factors": list(model.price_factors),
                "initial_price": model.initial_price,
                "initial_geometric": model.initial_geometric,
                "references": references,
            }
        )
        print(
            f"  rung5 price_scale={price_scale} fb={fraction_bits}: "
            f"states={model.shock_weight_sum + 1} checked",
            flush=True,
        )
    return {
        "what_was_executed": (
            "every loaded constant of the N=252 collapsed model checked in "
            f"{DECIMAL_DIGITS}-digit decimal against the exact real quantity it "
            "approximates, and the floored geometric chain checked against the "
            "exact geometric value on every reachable weighted shock sum"
        ),
        "why_it_matters": (
            "the encoded residual is nonnegative because the arithmetic leg rounds "
            "up and the geometric leg rounds down, so that total >= N * G * "
            "price_scale >= N * encoded_geometric.  The module rounds with a "
            "32-ulp tolerance and never checks that the constants landed on the "
            "required side, so the two preconditions are verified here instead."
        ),
        "preconditions": [
            "initial_price >= s0 * price_scale, and every price factor at least "
            "the exact real factor scaled by 2**fraction_bits",
            "initial_geometric at most the exact real initial geometric value "
            "scaled by price_scale, and every chain factor at most the exact real "
            "factor scaled by 2**fraction_bits",
        ],
        "n_dates": RUNG5_N_DATES,
        "reachable_weighted_sums": RUNG5_N_DATES * (RUNG5_N_DATES + 1) // 2 + 1,
        "decimal_digits": DECIMAL_DIGITS,
        "narrowest_margin_by_fraction_bits": {
            str(row["model"]["multiplier_fraction_bits"]): row["references"][
                "module_doubles"
            ]["narrowest_transcendental_margin_in_units_of_the_32_ulp_tolerance"]
            for row in rows
        },
        "caveat": (
            "every precondition holds at all three published precisions, but "
            "nothing in the module enforces them.  The margin is the distance "
            "between a transcendental and the integer grid, so it is not "
            "predictable from the precision alone and it shrinks as the "
            "precision grows.  It must be rechecked after any change to the "
            "model parameters, the precision, or the rounding helpers."
        ),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def provenance() -> dict:
    dirty = git("status", "--porcelain")
    return {
        "script": "scripts/v22_collapsed_validation_ladder.py",
        "script_sha256": sha256(Path(__file__).resolve()),
        "module_sha256": {
            "src/qc_option_pricing/quantum/arithmetic_asian_oracle.py": sha256(
                ROOT / "src/qc_option_pricing/quantum/arithmetic_asian_oracle.py"
            ),
            "src/qc_option_pricing/quantum/asian_oracle.py": sha256(
                ROOT / "src/qc_option_pricing/quantum/asian_oracle.py"
            ),
        },
        "git_rev": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "git_dirty_paths": dirty.splitlines(),
        "cwd": str(ROOT),
        "command": "python scripts/v22_collapsed_validation_ladder.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
        },
        "seeds": (
            "none: every rung is exhaustive and deterministic.  No Monte Carlo, "
            "no sampling, no random state anywhere in this script."
        ),
    }


RUNGS = {
    "rung_1_exhaustive_enumeration": rung_one,
    "rung_2_circuit_against_mirror": rung_two,
    "rung_3_struck_contract_at_depth": rung_three,
    "rung_4_price_decode": rung_four,
    "rung_5_ordering_guarantee_at_252_dates": rung_five,
}


def summarise(document: dict) -> dict:
    """Scannable headline, derived only from rungs present in the document."""

    summary: dict = {}
    states: dict = {}

    one = document.get("rung_1_exhaustive_enumeration")
    if one:
        summary["rung_1"] = {
            "sizes": one["sizes"],
            "quantisation": one["quantisation"],
            "paths_enumerated": sum(row["paths_enumerated"] for row in one["rows"]),
            "minimum_residual_numerator_by_n": {
                str(row["model"]["n_dates"]): row["minimum_residual_numerator"]
                for row in one["rows"]
            },
            "minimum_residual_numerator_over_in_the_money_paths_by_n": {
                str(row["model"]["n_dates"]): row[
                    "minimum_residual_numerator_over_in_the_money_paths"
                ]
                for row in one["rows"]
            },
            "all_checks_passed": all(
                row["independent_mirror_matches_module_recurrence_exactly"]
                and row["control_identity_arithmetic_equals_n_geometric_plus_residual"]
                and row["residual_nonnegative_on_every_path"]
                for row in one["rows"]
            ),
        }
        states["collapsed_integer_model_N2_to_N5"] = (
            "VERIFIED by exhaustive enumeration of every path against an "
            "independent recurrence and an 80-digit decimal AM-GM sandwich"
        )

    two = document.get("rung_2_circuit_against_mirror")
    if two:
        summary["rung_2"] = {
            "sizes": two["sizes"],
            "qubits_by_n": {
                str(row["model"]["n_dates"]): row["qubits"] for row in two["rows"]
            },
            "basis_states_covered": sum(
                row["prepared_basis_states"] for row in two["rows"]
            ),
            "all_per_path_exact": all(
                row["per_path_objective_count_matches_min_residual_cap_exactly"]
                for row in two["rows"]
            ),
            "all_branches_clean": all(row["dirty_states"] == 0 for row in two["rows"]),
        }
        states["collapsed_executable_A_N2_to_N5"] = (
            "VERIFIED per shock path by an exact permutation decode of the "
            "transpiled circuit over every prepared basis state"
        )

    three = document.get("rung_3_struck_contract_at_depth")
    if three:
        executed = [*three["full_residual_resolution_uncapped"], *three["capped_threshold_register"]]
        deepest = max(executed, key=lambda row: row["model"]["n_dates"])
        summary["rung_3"] = {
            "deepest_struck_n_dates": deepest["model"]["n_dates"],
            "deepest_qubits": deepest["qubits"],
            "deepest_prepared_basis_states": deepest["prepared_basis_states"],
            "deepest_hinges": deepest["hinges"],
            "deepest_threshold_bits": deepest["model"]["threshold_bits"],
            "deepest_uncapped_n_dates": max(
                row["model"]["n_dates"]
                for row in three["full_residual_resolution_uncapped"]
            ),
            "every_hinge_two_sided": all(
                row["hinges"]["arithmetic_hinge_two_sided"]
                and row["hinges"]["geometric_hinge_two_sided"]
                for row in executed
            ),
            "what_stopped_it": three["what_stopped_it"],
        }
        states["struck_two_sided_hinges"] = (
            f"VERIFIED to N={deepest['model']['n_dates']} at "
            f"{deepest['qubits']} qubits over "
            f"{deepest['prepared_basis_states']} prepared basis states, at the "
            "coarse quantisation recorded in rung 3; deeper N was not executed"
        )

    four = document.get("rung_4_price_decode")
    if four:
        gaps = [
            (leg["absolute_gap"], leg["relative_gap"], row["n_dates"], name)
            for row in four["rows"]
            for name, leg in row["legs"].items()
        ]
        worst = max(gaps, key=lambda item: abs(item[0]))
        summary["rung_4"] = {
            "cases": [
                {
                    "n_dates": row["n_dates"],
                    "price_units_per_dollar": row["price_units_per_dollar"],
                    "multiplier_fraction_bits": row["multiplier_fraction_bits"],
                    "collapsed_absolute_gap": row["legs"]["collapsed"]["absolute_gap"],
                    "none_absolute_gap": row["legs"]["none"]["absolute_gap"],
                    "residual_minus_raw_price": row["residual_minus_raw_price"],
                }
                for row in four["rows"]
            ],
            "worst_absolute_gap": worst[0],
            "worst_relative_gap": worst[1],
            "worst_case": {"n_dates": worst[2], "leg": worst[3]},
        }
        states["price_decode_at_production_precision"] = (
            "VERIFIED at N=2 and N=3 with 30 multiplier fraction bits and "
            "16384 price units per dollar, for the collapsed leg and the raw "
            "leg; deeper N at that precision was not executed"
        )

    five = document.get("rung_5_ordering_guarantee_at_252_dates")
    if five:
        summary["rung_5"] = {
            "n_dates": five["n_dates"],
            "precisions": [
                row["model"]["multiplier_fraction_bits"] for row in five["rows"]
            ],
            "states_checked_per_precision": five["reachable_weighted_sums"],
            "all_preconditions_hold": all(
                reference["price_factor_ceiling_holds"]
                and reference["initial_price_ceiling_holds"]
                and reference["initial_geometric_floor_holds"]
                and reference["chain_factor_floor_holds"]
                and reference["chain_sweep_holds"]
                for row in five["rows"]
                for reference in row["references"].values()
            ),
            "narrowest_margin_by_fraction_bits": five.get(
                "narrowest_margin_by_fraction_bits"
            ),
            "worst_chain_sweep_slack_encoded_units": min(
                reference["chain_sweep_worst_slack_encoded_units"]
                for row in five["rows"]
                for reference in row["references"].values()
            ),
        }
        states["252_date_ordering_preconditions"] = (
            "VERIFIED in 80-digit decimal for all three published precisions at "
            "N=252, on every reachable weighted shock sum; the preconditions are "
            "not enforced anywhere in the module and must be rechecked after any "
            "change to parameters, precision, or the rounding helpers"
        )

    states["252_date_circuit_execution"] = (
        "NOT-ASSESSABLE here; the 252-date circuit is intentionally not "
        "materialised by this script"
    )
    states["continuous_model_accuracy"] = (
        "OUT OF SCOPE for this artifact; every number here is a statement about "
        "the rounded integer model, not about continuous Black-Scholes"
    )
    summary["evidence_states"] = states
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="1,2,3,4,5",
        help="comma-separated rung numbers to run; results merge into the artifact",
    )
    arguments = parser.parse_args()
    selected = [int(value) for value in arguments.only.split(",") if value.strip()]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document: dict = {}
    if OUTPUT.exists():
        document = json.loads(OUTPUT.read_text())
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    document.setdefault("schema_version", SCHEMA)
    document["estimand"] = (
        "correctness of the collapsed geometric leg of the arithmetic Asian "
        "control-variate oracle: its integer model, its executable circuit, its "
        "struck-contract behaviour at depth, its price decode, and the "
        "directed-rounding preconditions of its residual encoding at 252 dates"
    )
    document["created_at_start"] = started_at
    document["provenance"] = provenance()
    document.setdefault("rung_wall_seconds", {})
    document.setdefault("rungs_executed_at", {})

    for number, (name, function) in enumerate(RUNGS.items(), start=1):
        if number not in selected:
            continue
        print(f"=== {name} ===", flush=True)
        started = time.perf_counter()
        document[name] = function()
        elapsed = time.perf_counter() - started
        document["rung_wall_seconds"][name] = elapsed
        document["rungs_executed_at"][name] = datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        )
        document["created_at_end"] = (
            datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        )
        document["summary"] = summarise(document)
        OUTPUT.write_text(json.dumps(document, indent=1) + "\n")
        print(f"  {name}: {elapsed:.1f}s", flush=True)

    document["summary"] = summarise(document)
    document["created_at_end"] = (
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    )
    OUTPUT.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
