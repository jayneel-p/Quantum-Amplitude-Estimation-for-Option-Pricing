"""Independent and executable checks for the arithmetic Asian QCV oracle."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import itertools
import math
import unittest
from dataclasses import replace

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    _append_threshold_encoder,
    _floor_product,
    _path_values,
    _weighted_shock_sum,
    arithmetic_objective_probability_from_mps,
    arithmetic_roundtrip_leakage_from_mps,
    build_arithmetic_asian_model,
    build_arithmetic_asian_oracle,
    build_selected_fixed_multiplier_gate,
    enumerate_arithmetic_asian,
    estimate_arithmetic_asian_resources,
    iter_reachable_path_values,
    primitive_counts_from_circuit,
    uniform_normal_midpoint_grid,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec


def _basis_value(state: Statevector) -> int:
    probabilities = np.abs(state.data) ** 2
    index = int(np.argmax(probabilities))
    if not math.isclose(float(probabilities[index]), 1.0, abs_tol=1e-12):
        raise AssertionError("basis-state test produced a superposition")
    return index


def _reversible_classical_program(oracle) -> tuple[list[tuple[int, tuple[int, ...]]], int]:
    """Transpile ``A`` into a reversible classical program on basis states.

    Every gate of ``A`` is H, X, CX or Toffoli and the H gates act only on the
    shock and threshold registers.  Fixing those registers therefore turns the
    remainder into a permutation of computational basis states, which can be
    executed exactly with integers and no simulator.
    """

    from qiskit import transpile

    circuit = transpile(
        oracle.circuit, basis_gates=["h", "x", "cx", "ccx"], optimization_level=0
    )
    prepared = set(oracle.shock_qubits) | set(oracle.threshold_qubits)
    program: list[tuple[int, tuple[int, ...]]] = []
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if name == "h":
            if qubits[0] not in prepared:
                raise AssertionError("H acts outside the prepared registers")
            continue
        if name not in ("x", "cx", "ccx"):
            raise AssertionError(f"unexpected primitive {name}")
        program.append((len(qubits), qubits))
    return program, circuit.num_qubits


def _run_reversible_program(program, num_qubits: int, assignment) -> bytearray:
    state = bytearray(num_qubits)
    for index, value in assignment.items():
        state[index] = value
    for arity, qubits in program:
        if arity == 1:
            state[qubits[0]] ^= 1
        elif arity == 2:
            state[qubits[1]] ^= state[qubits[0]]
        else:
            state[qubits[2]] ^= state[qubits[0]] & state[qubits[1]]
    return state


def _decimal_integer(value: Decimal, rounding: str) -> int:
    """Convert a positive high-precision value with an explicit direction."""

    mode = ROUND_CEILING if rounding == "ceil" else ROUND_FLOOR
    return int(value.to_integral_value(rounding=mode))


def _independent_finite_reference(
    spec: AsianGridSpec, *, multiplier_fraction_bits: int
) -> dict[str, float | int]:
    """Recompute the encoded model without using the oracle model helpers.

    This deliberately uses ``Decimal.exp`` and its own path loop.  It checks
    the finite model compiled from the public specification, rather than
    replaying the implementation's floating-point rounding helpers.
    """

    if any(probability != 0.5 for probability in spec.shock_probabilities):
        raise ValueError("the independent reference is specialized to binary uniform shocks")
    with localcontext() as context:
        context.prec = 80
        n_dates = spec.n_dates
        scale = 1 << multiplier_fraction_bits
        decimal = lambda value: Decimal(str(value))
        dt = decimal(spec.maturity) / Decimal(n_dates)
        drift = decimal(spec.rate) - decimal(spec.volatility) ** 2 / Decimal(2)
        diffusion = decimal(spec.volatility) * dt.sqrt()
        price_factors = tuple(
            _decimal_integer(
                (drift * dt + diffusion * decimal(shock)).exp() * scale,
                "ceil",
            )
            for shock in spec.shock_points
        )
        geometric_factors = tuple(
            tuple(
                _decimal_integer(
                    (
                        diffusion
                        * Decimal(n_dates - date)
                        * decimal(shock)
                        / Decimal(n_dates)
                    ).exp()
                    * scale,
                    "floor",
                )
                for shock in spec.shock_points
            )
            for date in range(n_dates)
        )
        initial_price = _decimal_integer(decimal(spec.s0) * spec.price_scale, "ceil")
        initial_geometric = _decimal_integer(
            decimal(spec.s0)
            * (drift * dt * Decimal(n_dates + 1) / Decimal(2)).exp()
            * spec.price_scale,
            "floor",
        )

    arithmetic_sum = 0
    geometric_sum = 0
    residual_sum = 0
    minimum_residual = math.inf
    maximum_residual = 0
    residuals: list[int] = []
    path_count = 2**spec.n_dates
    for digits in itertools.product(range(2), repeat=spec.n_dates):
        price = initial_price
        geometric = initial_geometric
        total = 0
        for date, digit in enumerate(digits):
            price = (price * price_factors[digit] + scale - 1) >> multiplier_fraction_bits
            geometric = (
                geometric * geometric_factors[date][digit]
            ) >> multiplier_fraction_bits
            total += price
        arithmetic = max(total - spec.n_dates * spec.strike_integer, 0)
        geometric_payoff = max(geometric - spec.strike_integer, 0)
        residual = arithmetic - spec.n_dates * geometric_payoff
        if residual < 0:
            raise AssertionError("independent directed-rounding reference violated AM--GM")
        arithmetic_sum += arithmetic
        geometric_sum += geometric_payoff
        residual_sum += residual
        residuals.append(residual)
        minimum_residual = min(minimum_residual, residual)
        maximum_residual = max(maximum_residual, residual)

    price_upper = initial_price
    safe_raw_payoff_upper = 0
    for _ in range(spec.n_dates):
        price_upper = max(
            (price_upper * factor + scale - 1) >> multiplier_fraction_bits
            for factor in price_factors
        )
        safe_raw_payoff_upper += price_upper
    safe_residual_upper = max(
        safe_raw_payoff_upper - spec.n_dates * spec.strike_integer, 0
    )
    if spec.residual_payoff_cap is None:
        cap_numerator = safe_residual_upper
    else:
        cap_numerator = int(
            round(spec.residual_payoff_cap * spec.n_dates * spec.price_scale)
        )
    clipped_residual_sum = sum(min(residual, cap_numerator) for residual in residuals)
    threshold_bits = max(1, (cap_numerator - 1).bit_length())
    denominator = path_count * spec.n_dates * spec.price_scale
    return {
        "price_factors": price_factors,
        "geometric_factors": geometric_factors,
        "initial_price": initial_price,
        "initial_geometric": initial_geometric,
        "safe_residual_upper": safe_residual_upper,
        "requested_cap_numerator": cap_numerator,
        "arithmetic_payoff_undiscounted": arithmetic_sum / denominator,
        "geometric_payoff_undiscounted": spec.n_dates * geometric_sum / denominator,
        "residual_payoff_undiscounted": residual_sum / denominator,
        "clipped_residual_payoff_undiscounted": clipped_residual_sum / denominator,
        "objective_probability": clipped_residual_sum / (path_count * (1 << threshold_bits)),
        "minimum_residual_numerator": int(minimum_residual),
        "maximum_residual_numerator": maximum_residual,
    }


class SelectedFixedMultiplierTests(unittest.TestCase):
    def _check_gate(
        self,
        *,
        value_bits: int,
        shock_bits: int,
        product_bits: int,
        fraction_bits: int,
        factors: tuple[int, ...],
        rounding: str,
    ) -> None:
        gate = build_selected_fixed_multiplier_gate(
            value_bits=value_bits,
            shock_bits=shock_bits,
            product_bits=product_bits,
            fraction_bits=fraction_bits,
            factors=factors,
            rounding=rounding,
        )
        product_start = value_bits + shock_bits
        work_start = product_start + product_bits
        for value in range(1 << value_bits):
            for shock in range(1 << shock_bits):
                with self.subTest(value=value, shock=shock, rounding=rounding):
                    circuit = QuantumCircuit(gate.num_qubits)
                    for bit in range(value_bits):
                        if (value >> bit) & 1:
                            circuit.x(bit)
                    for bit in range(shock_bits):
                        if (shock >> bit) & 1:
                            circuit.x(value_bits + bit)
                    circuit.append(gate, range(gate.num_qubits))
                    encoded = _basis_value(Statevector.from_instruction(circuit))
                    product = sum(
                        ((encoded >> (product_start + bit)) & 1) << bit
                        for bit in range(product_bits)
                    )
                    offset = (1 << fraction_bits) - 1 if rounding == "ceil" else 0
                    self.assertEqual(product, value * factors[shock] + offset)
                    self.assertEqual(encoded >> work_start, 0)
                    self.assertEqual(encoded & ((1 << value_bits) - 1), value)
                    recovered = Statevector.from_instruction(
                        circuit.compose(circuit.inverse())
                    )
                    self.assertEqual(_basis_value(recovered), 0)

    def test_binary_ceil_multiplier_exhaustively(self) -> None:
        self._check_gate(
            value_bits=2,
            shock_bits=1,
            product_bits=5,
            fraction_bits=2,
            factors=(3, 5),
            rounding="ceil",
        )

    def test_two_qubit_floor_multiplier_exhaustively(self) -> None:
        self._check_gate(
            value_bits=1,
            shock_bits=2,
            product_bits=4,
            fraction_bits=2,
            factors=(3, 4, 5, 6),
            rounding="floor",
        )


class UniformNormalMidpointGridTests(unittest.TestCase):
    def test_binary_grid_reduces_to_plus_or_minus_one(self) -> None:
        points, probabilities = uniform_normal_midpoint_grid(1)
        self.assertAlmostEqual(points[0], -1.0, places=15)
        self.assertAlmostEqual(points[1], 1.0, places=15)
        self.assertEqual(probabilities, (0.5, 0.5))

    def test_uniform_grid_has_the_encoded_first_two_moments(self) -> None:
        for shock_qubits in (1, 2, 3):
            points, probabilities = uniform_normal_midpoint_grid(shock_qubits)
            weights = np.asarray(probabilities)
            values = np.asarray(points)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=15)
            self.assertAlmostEqual(float(weights @ values), 0.0, places=15)
            self.assertAlmostEqual(float(weights @ values**2), 1.0, places=15)


class ThresholdEncoderTests(unittest.TestCase):
    def test_uniform_threshold_probability_and_cleanup(self) -> None:
        width = 2
        for residual_value in range(1 << width):
            with self.subTest(residual=residual_value):
                threshold = QuantumRegister(width, "threshold")
                residual = QuantumRegister(width, "residual")
                objective = QuantumRegister(1, "objective")
                scratch = QuantumRegister(width + 1, "scratch")
                zero = QuantumRegister(1, "zero")
                helper = QuantumRegister(1, "helper")
                residual_flag = QuantumRegister(1, "residual_flag")
                cap_flag = QuantumRegister(1, "cap_flag")
                constant = QuantumRegister(width + 1, "constant")
                circuit = QuantumCircuit(
                    threshold,
                    residual,
                    objective,
                    scratch,
                    zero,
                    helper,
                    residual_flag,
                    cap_flag,
                    constant,
                )
                for bit in range(width):
                    if (residual_value >> bit) & 1:
                        circuit.x(residual[bit])
                circuit.h(threshold)
                _append_threshold_encoder(
                    circuit,
                    threshold,
                    residual,
                    objective[0],
                    scratch,
                    constant,
                    zero[0],
                    helper[0],
                    residual_flag[0],
                    cap_flag[0],
                    1 << width,
                )
                state = Statevector.from_instruction(circuit)
                objective_index = circuit.find_bit(objective[0]).index
                objective_probability = float(
                    state.probabilities([objective_index])[1]
                )
                self.assertAlmostEqual(
                    objective_probability, residual_value / (1 << width), places=13
                )
                for qubit in [
                    *scratch,
                    zero[0],
                    helper[0],
                    residual_flag[0],
                    cap_flag[0],
                    *constant,
                ]:
                    index = circuit.find_bit(qubit).index
                    self.assertLess(float(state.probabilities([index])[1]), 1e-13)

    def test_explicit_non_power_of_two_cap(self) -> None:
        width = 2
        residual_width = 3
        cap = 3
        for residual_value in range(1 << residual_width):
            threshold = QuantumRegister(width, "threshold")
            residual = QuantumRegister(residual_width, "residual")
            objective = QuantumRegister(1, "objective")
            scratch = QuantumRegister(residual_width + 1, "scratch")
            constant = QuantumRegister(residual_width + 1, "constant")
            zero = QuantumRegister(1, "zero")
            helper = QuantumRegister(1, "helper")
            residual_flag = QuantumRegister(1, "residual_flag")
            cap_flag = QuantumRegister(1, "cap_flag")
            circuit = QuantumCircuit(
                threshold,
                residual,
                objective,
                scratch,
                constant,
                zero,
                helper,
                residual_flag,
                cap_flag,
            )
            for bit in range(residual_width):
                if (residual_value >> bit) & 1:
                    circuit.x(residual[bit])
            circuit.h(threshold)
            _append_threshold_encoder(
                circuit,
                threshold,
                residual,
                objective[0],
                scratch,
                constant,
                zero[0],
                helper[0],
                residual_flag[0],
                cap_flag[0],
                cap,
            )
            state = Statevector.from_instruction(circuit)
            objective_index = circuit.find_bit(objective[0]).index
            self.assertAlmostEqual(
                float(state.probabilities([objective_index])[1]),
                min(residual_value, cap) / (1 << width),
                places=13,
            )
            for qubit in [
                *scratch,
                *constant,
                zero[0],
                helper[0],
                residual_flag[0],
                cap_flag[0],
            ]:
                index = circuit.find_bit(qubit).index
                self.assertLess(float(state.probabilities([index])[1]), 1e-13)


class ArithmeticAsianOracleTests(unittest.TestCase):
    @staticmethod
    def _small_spec() -> AsianGridSpec:
        return AsianGridSpec(
            n_dates=2,
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

    @staticmethod
    def _small_collapsed_spec() -> AsianGridSpec:
        """Small, but with a nondegenerate exponential chain and control."""

        return AsianGridSpec(
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

    def test_full_a_matches_independent_enumeration_and_cleans_work(self) -> None:
        model = build_arithmetic_asian_model(
            self._small_spec(), multiplier_fraction_bits=1
        )
        reference = enumerate_arithmetic_asian(model)
        oracle = build_arithmetic_asian_oracle(model)
        probability, work_hamming = arithmetic_objective_probability_from_mps(oracle)
        self.assertAlmostEqual(
            probability, reference.objective_probability, delta=1e-10
        )
        self.assertLess(work_hamming, 1e-10)
        self.assertAlmostEqual(
            oracle.post_process(probability),
            math.exp(-model.spec.rate * model.spec.maturity)
            * reference.arithmetic_payoff_undiscounted,
            delta=4e-10,
        )

    def test_full_a_inverse_roundtrip(self) -> None:
        model = build_arithmetic_asian_model(
            self._small_spec(), multiplier_fraction_bits=1
        )
        oracle = build_arithmetic_asian_oracle(model)
        self.assertLess(arithmetic_roundtrip_leakage_from_mps(oracle), 1e-10)

    def test_full_a_implements_explicit_residual_cap(self) -> None:
        spec = replace(self._small_spec(), residual_payoff_cap=1.5)
        model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=1)
        reference = enumerate_arithmetic_asian(model)
        oracle = build_arithmetic_asian_oracle(model)
        probability, work_hamming = arithmetic_objective_probability_from_mps(oracle)
        self.assertEqual(model.requested_residual_cap_numerator, 3)
        self.assertEqual(model.normalization_numerator, 4)
        self.assertAlmostEqual(
            probability, reference.objective_probability, delta=1e-10
        )
        self.assertLess(work_hamming, 1e-10)
        self.assertLess(
            reference.clipped_residual_payoff_undiscounted,
            reference.residual_payoff_undiscounted,
        )

    def test_executable_primitive_counts_equal_compositional_ledger(self) -> None:
        model = build_arithmetic_asian_model(
            self._small_spec(), multiplier_fraction_bits=1
        )
        oracle = build_arithmetic_asian_oracle(model)
        actual = primitive_counts_from_circuit(oracle)
        estimated = estimate_arithmetic_asian_resources(model)
        self.assertEqual(actual, estimated.a_counts)
        self.assertEqual(estimated.qrom_rows, 0)
        self.assertEqual(estimated.arbitrary_rotations, 0)

    def test_directed_rounding_preserves_pathwise_control_identity(self) -> None:
        for n_dates in (2, 3, 4):
            for price_scale in (1, 2, 4):
                spec = AsianGridSpec(
                    n_dates=n_dates,
                    shock_points=(-1.0, 1.0),
                    shock_probabilities=(0.5, 0.5),
                    s0=3.0,
                    strike=1.0,
                    rate=0.03,
                    volatility=0.4,
                    maturity=1.25,
                    shock_scale=1,
                    price_scale=price_scale,
                )
                model = build_arithmetic_asian_model(
                    spec, multiplier_fraction_bits=4
                )
                for _, (arithmetic, geometric, residual, _) in iter_reachable_path_values(
                    model
                ):
                    self.assertGreaterEqual(residual, 0)
                    self.assertEqual(
                        arithmetic, n_dates * geometric + residual
                    )

    def test_model_matches_independent_high_precision_finite_reference(self) -> None:
        """Check finite-model compilation against Decimal arithmetic and path loops."""

        cases = (
            (
                AsianGridSpec(
                    n_dates=2,
                    shock_points=(-1.0, 1.0),
                    shock_probabilities=(0.5, 0.5),
                    s0=3.0,
                    strike=1.0,
                    rate=0.02,
                    volatility=0.35,
                    maturity=0.75,
                    shock_scale=1,
                    price_scale=8,
                    residual_payoff_cap=0.5,
                ),
                8,
            ),
            (
                AsianGridSpec(
                    n_dates=3,
                    shock_points=(-1.0, 1.0),
                    shock_probabilities=(0.5, 0.5),
                    s0=4.0,
                    strike=2.0,
                    rate=0.05,
                    volatility=0.30,
                    maturity=1.0,
                    shock_scale=1,
                    price_scale=16,
                ),
                9,
            ),
            (
                AsianGridSpec(
                    n_dates=4,
                    shock_points=(-1.0, 1.0),
                    shock_probabilities=(0.5, 0.5),
                    s0=2.0,
                    strike=1.0,
                    rate=0.0,
                    volatility=0.55,
                    maturity=1.25,
                    shock_scale=1,
                    price_scale=4,
                    residual_payoff_cap=1.0,
                ),
                10,
            ),
        )
        for spec, fraction_bits in cases:
            with self.subTest(n_dates=spec.n_dates, fraction_bits=fraction_bits):
                expected = _independent_finite_reference(
                    spec, multiplier_fraction_bits=fraction_bits
                )
                model = build_arithmetic_asian_model(
                    spec, multiplier_fraction_bits=fraction_bits
                )
                actual = enumerate_arithmetic_asian(model)
                self.assertEqual(model.price_factors, expected["price_factors"])
                self.assertEqual(model.geometric_factors, expected["geometric_factors"])
                self.assertEqual(model.initial_price, expected["initial_price"])
                self.assertEqual(model.initial_geometric, expected["initial_geometric"])
                self.assertEqual(model.maximum_residual, expected["safe_residual_upper"])
                self.assertEqual(
                    model.requested_residual_cap_numerator,
                    expected["requested_cap_numerator"],
                )
                for field in (
                    "arithmetic_payoff_undiscounted",
                    "geometric_payoff_undiscounted",
                    "residual_payoff_undiscounted",
                    "clipped_residual_payoff_undiscounted",
                    "objective_probability",
                ):
                    self.assertAlmostEqual(
                        getattr(actual, field), expected[field], places=15
                    )
                self.assertEqual(
                    actual.minimum_residual_numerator,
                    expected["minimum_residual_numerator"],
                )
                self.assertEqual(
                    actual.maximum_residual_numerator,
                    expected["maximum_residual_numerator"],
                )

    def test_nonuniform_state_preparation_is_rejected(self) -> None:
        spec = AsianGridSpec(
            n_dates=2,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.4, 0.6),
            s0=2.0,
            strike=1.0,
            rate=0.0,
            volatility=0.2,
            maturity=1.0,
            shock_scale=1,
            price_scale=1,
        )
        with self.assertRaisesRegex(ValueError, "uniform shocks"):
            build_arithmetic_asian_model(spec)

    def test_externally_calibrated_geometric_control_skips_state_dp(self) -> None:
        model = build_arithmetic_asian_model(
            self._small_spec(),
            multiplier_fraction_bits=1,
            geometric_control_undiscounted_override=0.25,
            geometric_control_standard_error_undiscounted=0.01,
        )
        self.assertEqual(model.geometric_dp_peak_states, 0)
        self.assertEqual(model.geometric_payoff_count_denominator, 0)
        self.assertEqual(model.geometric_control_undiscounted, 0.25)
        self.assertEqual(model.geometric_control_standard_error_undiscounted, 0.01)
        oracle = build_arithmetic_asian_oracle(model)
        probability, work_hamming = arithmetic_objective_probability_from_mps(oracle)
        self.assertLess(work_hamming, 1e-10)
        reference = enumerate_arithmetic_asian(model)
        self.assertAlmostEqual(
            oracle.post_process(probability),
            math.exp(-model.spec.rate * model.spec.maturity)
            * (0.25 + reference.residual_payoff_undiscounted),
            delta=4e-10,
        )

    def test_external_control_requires_a_finite_uncertainty(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires its standard error"):
            build_arithmetic_asian_model(
                self._small_spec(), geometric_control_undiscounted_override=0.25
            )
        with self.assertRaisesRegex(ValueError, "requires an override"):
            build_arithmetic_asian_model(
                self._small_spec(), geometric_control_standard_error_undiscounted=0.01
            )

    def test_geometric_leg_defaults_to_per_date_and_rejects_unknown_values(self) -> None:
        spec = self._small_spec()
        self.assertEqual(spec.geometric_leg, "per_date")
        model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=1)
        self.assertEqual(model.geometric_chain_factors, ())
        self.assertEqual(model.shock_weight_bits, 0)
        self.assertEqual(model.geometric_product_bits, model.product_bits)
        with self.assertRaisesRegex(ValueError, "geometric_leg"):
            replace(spec, geometric_leg="fast")
        with self.assertRaisesRegex(ValueError, "binary shock qubit"):
            build_arithmetic_asian_model(
                AsianGridSpec(
                    n_dates=2,
                    shock_points=(-1.5, -0.5, 0.5, 1.5),
                    shock_probabilities=(0.25,) * 4,
                    s0=2.0,
                    strike=1.0,
                    rate=0.0,
                    volatility=0.2,
                    maturity=1.0,
                    shock_scale=1,
                    price_scale=1,
                    geometric_leg="collapsed",
                ),
                multiplier_fraction_bits=4,
            )

    def test_collapsed_leg_keeps_the_arithmetic_leg_and_stays_under_the_exact_average(
        self,
    ) -> None:
        """The collapsed control changes only the geometric leg, and floors it."""

        for n_dates in (2, 3, 4, 5):
            spec = AsianGridSpec(
                n_dates=n_dates,
                shock_points=(-1.0, 1.0),
                shock_probabilities=(0.5, 0.5),
                s0=3.0,
                strike=1.0,
                rate=0.03,
                volatility=0.4,
                maturity=1.25,
                shock_scale=1,
                price_scale=4,
            )
            per_date = build_arithmetic_asian_model(spec, multiplier_fraction_bits=4)
            collapsed = build_arithmetic_asian_model(
                replace(spec, geometric_leg="collapsed"), multiplier_fraction_bits=4
            )
            with self.subTest(n_dates=n_dates):
                self.assertEqual(collapsed.price_factors, per_date.price_factors)
                self.assertEqual(collapsed.initial_price, per_date.initial_price)
                self.assertEqual(collapsed.product_bits, per_date.product_bits)
                self.assertEqual(
                    len(collapsed.geometric_chain_factors),
                    collapsed.shock_weight_bits,
                )
                self.assertEqual(
                    collapsed.shock_weight_sum, n_dates * (n_dates + 1) // 2
                )
                with localcontext() as context:
                    context.prec = 60
                    decimal = lambda value: Decimal(str(value))
                    dt = decimal(spec.maturity) / Decimal(n_dates)
                    drift = decimal(spec.rate) - decimal(spec.volatility) ** 2 / 2
                    diffusion = decimal(spec.volatility) * dt.sqrt()
                    for digits in itertools.product(range(2), repeat=n_dates):
                        weighted = sum(
                            (n_dates - date) * (2 * digit - 1)
                            for date, digit in enumerate(digits)
                        )
                        exact = (
                            decimal(spec.s0)
                            * (
                                drift * dt * Decimal(n_dates + 1) / 2
                                + diffusion * Decimal(weighted) / Decimal(n_dates)
                            ).exp()
                            * spec.price_scale
                        )
                        encoded = collapsed.initial_geometric
                        weighted_bits = _weighted_shock_sum(collapsed, digits)
                        for bit, factor in enumerate(
                            collapsed.geometric_chain_factors
                        ):
                            if (weighted_bits >> bit) & 1:
                                encoded = _floor_product(
                                    encoded,
                                    factor,
                                    collapsed.multiplier_fraction_bits,
                                )
                        arithmetic, geometric, residual, total = _path_values(
                            collapsed, digits
                        )
                        self.assertEqual(
                            geometric, max(encoded - spec.strike_integer, 0)
                        )
                        self.assertGreaterEqual(residual, 0)
                        self.assertEqual(arithmetic, n_dates * geometric + residual)
                        # the arithmetic leg is untouched by the collapsed control
                        self.assertEqual(total, _path_values(per_date, digits)[3])
                        # directed rounding: the chain never exceeds the exact
                        # geometric average, and the encoded averages stay ordered
                        self.assertLessEqual(Decimal(encoded), exact)
                        self.assertGreaterEqual(total, n_dates * encoded)

    def test_collapsed_leg_rejects_a_price_scale_that_floors_its_constant(self) -> None:
        with self.assertRaisesRegex(ValueError, "rounds to zero"):
            build_arithmetic_asian_model(
                replace(self._small_spec(), geometric_leg="collapsed"),
                multiplier_fraction_bits=4,
            )

    def test_none_leg_drops_every_control_field_and_rejects_control_configuration(
        self,
    ) -> None:
        spec = replace(self._small_collapsed_spec(), geometric_leg="none")
        model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=3)
        self.assertEqual(model.geometric_factors, ())
        self.assertEqual(model.geometric_chain_factors, ())
        self.assertEqual(model.shock_weight_sum, 0)
        self.assertEqual(model.shock_weight_bits, 0)
        self.assertEqual(model.initial_geometric, 0)
        self.assertEqual(model.maximum_geometric_values, ())
        self.assertEqual(model.geometric_product_bits, 0)
        self.assertEqual(model.geometric_dp_peak_states, 0)
        self.assertEqual(model.geometric_payoff_count_denominator, 0)
        # the raw cap defaults to the exact maximum payoff, with no truncation
        self.assertEqual(
            model.requested_residual_cap_numerator, model.maximum_residual
        )
        with self.assertRaisesRegex(ValueError, "no geometric control"):
            model.geometric_control_undiscounted
        with self.assertRaisesRegex(ValueError, "payoff_cap, not residual_payoff_cap"):
            build_arithmetic_asian_model(
                replace(spec, residual_payoff_cap=1.0), multiplier_fraction_bits=3
            )
        with self.assertRaisesRegex(ValueError, "no geometric control to calibrate"):
            build_arithmetic_asian_model(
                spec,
                multiplier_fraction_bits=3,
                geometric_control_undiscounted_override=0.25,
                geometric_control_standard_error_undiscounted=0.01,
            )
        with self.assertRaisesRegex(ValueError, "below one encoded payoff unit"):
            build_arithmetic_asian_model(
                replace(spec, payoff_cap=0.01), multiplier_fraction_bits=3
            )
        with self.assertRaisesRegex(ValueError, "exceeds the maximum encoded"):
            build_arithmetic_asian_model(
                replace(spec, payoff_cap=1000.0), multiplier_fraction_bits=3
            )

    def test_none_leg_is_the_bare_arithmetic_leg_and_encodes_the_capped_payoff(
        self,
    ) -> None:
        """'none' keeps the arithmetic leg byte-identical and has no control.

        The encoded quantity is ``min(arithmetic payoff, cap)``; the payoff is
        recomputed here with an inline integer price walk so the check does not
        route through the module's own path helper.
        """

        fraction_bits = 4
        for n_dates in (2, 3, 4, 5):
            spec = AsianGridSpec(
                n_dates=n_dates,
                shock_points=(-1.0, 1.0),
                shock_probabilities=(0.5, 0.5),
                s0=3.0,
                strike=1.0,
                rate=0.03,
                volatility=0.4,
                maturity=1.25,
                shock_scale=1,
                price_scale=4,
            )
            per_date = build_arithmetic_asian_model(
                spec, multiplier_fraction_bits=fraction_bits
            )
            raw = build_arithmetic_asian_model(
                replace(spec, geometric_leg="none"),
                multiplier_fraction_bits=fraction_bits,
            )
            capped = build_arithmetic_asian_model(
                replace(spec, geometric_leg="none", payoff_cap=1.25),
                multiplier_fraction_bits=fraction_bits,
            )
            cap = capped.requested_residual_cap_numerator
            with self.subTest(n_dates=n_dates):
                self.assertEqual(raw.price_factors, per_date.price_factors)
                self.assertEqual(raw.initial_price, per_date.initial_price)
                self.assertEqual(raw.product_bits, per_date.product_bits)
                self.assertEqual(raw.maximum_total, per_date.maximum_total)
                self.assertEqual(cap, round(1.25 * n_dates * spec.price_scale))
                self.assertLess(cap, raw.maximum_residual)
                clipped_sum = 0
                for digits in itertools.product(range(2), repeat=n_dates):
                    price = raw.initial_price
                    total = 0
                    for digit in digits:
                        price = (
                            price * raw.price_factors[digit]
                            + (1 << fraction_bits)
                            - 1
                        ) >> fraction_bits
                        total += price
                    payoff = max(total - n_dates * spec.strike_integer, 0)
                    arithmetic, geometric, residual, encoded_total = _path_values(
                        raw, digits
                    )
                    self.assertEqual(encoded_total, total)
                    self.assertEqual(arithmetic, payoff)
                    self.assertEqual(geometric, 0)
                    # with no control the encoded residual IS the payoff
                    self.assertEqual(residual, payoff)
                    # the arithmetic leg is untouched by dropping the control
                    self.assertEqual(total, _path_values(per_date, digits)[3])
                    self.assertEqual(
                        _path_values(capped, digits)[2], payoff
                    )
                    clipped_sum += min(payoff, cap)
                reference = enumerate_arithmetic_asian(capped)
                denominator = (1 << n_dates) * n_dates * spec.price_scale
                self.assertEqual(
                    reference.clipped_residual_payoff_undiscounted,
                    clipped_sum / denominator,
                )
                self.assertEqual(
                    reference.arithmetic_payoff_undiscounted,
                    enumerate_arithmetic_asian(per_date).arithmetic_payoff_undiscounted,
                )

    def test_none_full_a_matches_independent_enumeration_and_cleans_work(self) -> None:
        model = build_arithmetic_asian_model(
            replace(self._small_collapsed_spec(), geometric_leg="none"),
            multiplier_fraction_bits=3,
        )
        reference = enumerate_arithmetic_asian(model)
        oracle = build_arithmetic_asian_oracle(model)
        probability, work_hamming = arithmetic_objective_probability_from_mps(oracle)
        self.assertAlmostEqual(
            probability, reference.objective_probability, delta=1e-10
        )
        self.assertLess(work_hamming, 1e-10)
        # decoding is the bare normalization: no control is added back
        self.assertAlmostEqual(
            oracle.post_process(probability),
            math.exp(-model.spec.rate * model.spec.maturity)
            * reference.arithmetic_payoff_undiscounted,
            delta=4e-10,
        )

    def test_none_full_a_implements_explicit_payoff_cap(self) -> None:
        spec = replace(
            self._small_collapsed_spec(), geometric_leg="none", payoff_cap=2.5
        )
        model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=3)
        reference = enumerate_arithmetic_asian(model)
        oracle = build_arithmetic_asian_oracle(model)
        probability, work_hamming = arithmetic_objective_probability_from_mps(oracle)
        self.assertEqual(model.requested_residual_cap_numerator, 10)
        self.assertEqual(model.normalization_numerator, 16)
        self.assertAlmostEqual(
            probability, reference.objective_probability, delta=1e-10
        )
        self.assertLess(work_hamming, 1e-10)
        self.assertLess(
            reference.clipped_residual_payoff_undiscounted,
            reference.residual_payoff_undiscounted,
        )
        self.assertAlmostEqual(
            oracle.post_process(probability),
            math.exp(-spec.rate * spec.maturity)
            * reference.clipped_residual_payoff_undiscounted,
            delta=4e-10,
        )

    def test_none_primitive_counts_equal_compositional_ledger(self) -> None:
        for n_dates in (2, 3, 4, 5):
            model = build_arithmetic_asian_model(
                replace(
                    self._small_collapsed_spec(),
                    n_dates=n_dates,
                    geometric_leg="none",
                ),
                multiplier_fraction_bits=3,
            )
            oracle = build_arithmetic_asian_oracle(model)
            actual = primitive_counts_from_circuit(oracle)
            estimated = estimate_arithmetic_asian_resources(model)
            with self.subTest(n_dates=n_dates):
                self.assertEqual(actual, estimated.a_counts)
                self.assertEqual(oracle.circuit.num_qubits, estimated.a_qubits)
                for control_component in (
                    "geometric_selected_multipliers_in_compute",
                    "geometric_weighted_sum_in_compute",
                    "geometric_exponential_chain_in_compute",
                    "geometric_positive_part_in_compute",
                    "geometric_payoff_scaling_in_compute",
                    "residual_subtraction_in_compute",
                ):
                    self.assertNotIn(
                        control_component, estimated.component_counts
                    )

    def test_collapsed_full_a_matches_independent_enumeration_and_cleans_work(
        self,
    ) -> None:
        model = build_arithmetic_asian_model(
            self._small_collapsed_spec(), multiplier_fraction_bits=3
        )
        self.assertEqual(model.geometric_factors, ())
        self.assertEqual(model.geometric_chain_factors, (10, 15))
        self.assertGreater(model.geometric_control_undiscounted, 0.0)
        reference = enumerate_arithmetic_asian(model)
        oracle = build_arithmetic_asian_oracle(model)
        probability, work_hamming = arithmetic_objective_probability_from_mps(oracle)
        self.assertAlmostEqual(
            probability, reference.objective_probability, delta=1e-10
        )
        self.assertLess(work_hamming, 1e-10)
        self.assertAlmostEqual(
            oracle.post_process(probability),
            math.exp(-model.spec.rate * model.spec.maturity)
            * reference.arithmetic_payoff_undiscounted,
            delta=4e-10,
        )

    def test_collapsed_primitive_counts_equal_compositional_ledger(self) -> None:
        model = build_arithmetic_asian_model(
            self._small_collapsed_spec(), multiplier_fraction_bits=3
        )
        oracle = build_arithmetic_asian_oracle(model)
        actual = primitive_counts_from_circuit(oracle)
        estimated = estimate_arithmetic_asian_resources(model)
        self.assertEqual(actual, estimated.a_counts)
        self.assertEqual(oracle.circuit.num_qubits, estimated.a_qubits)
        self.assertIn(
            "geometric_weighted_sum_in_compute", estimated.component_counts
        )
        self.assertIn(
            "geometric_exponential_chain_in_compute", estimated.component_counts
        )
        self.assertNotIn(
            "geometric_selected_multipliers_in_compute", estimated.component_counts
        )

    def test_circuit_encodes_the_mirror_residual_on_every_path(self) -> None:
        """Execute A as a reversible classical program, one path at a time.

        Basis-state execution recovers ``min(residual, cap)`` exactly, so this
        checks the built circuit against the classical mirror as integers, and
        also that every work qubit returns to zero on every branch.  Under
        ``geometric_leg='none'`` the recovered quantity is the capped
        arithmetic payoff itself.
        """

        for leg, n_dates, payoff_cap in (
            ("per_date", 2, None),
            ("collapsed", 2, None),
            ("collapsed", 3, None),
            ("none", 2, None),
            ("none", 3, None),
            ("none", 2, 2.5),
        ):
            spec = replace(
                self._small_collapsed_spec(),
                n_dates=n_dates,
                geometric_leg=leg,
                payoff_cap=payoff_cap,
            )
            model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=3)
            oracle = build_arithmetic_asian_oracle(model)
            program, num_qubits = _reversible_classical_program(oracle)
            width = len(oracle.threshold_qubits)
            cap = min(model.requested_residual_cap_numerator, 1 << width)
            with self.subTest(geometric_leg=leg, n_dates=n_dates, payoff_cap=payoff_cap):
                for digits in itertools.product(range(2), repeat=n_dates):
                    fired = 0
                    for threshold in range(1 << width):
                        assignment = {
                            oracle.shock_qubits[date]: digit
                            for date, digit in enumerate(digits)
                        }
                        assignment.update(
                            {
                                oracle.threshold_qubits[bit]: (threshold >> bit) & 1
                                for bit in range(width)
                            }
                        )
                        state = _run_reversible_program(
                            program, num_qubits, assignment
                        )
                        fired += state[oracle.objective_qubit]
                        self.assertEqual(
                            sum(state[index] for index in oracle.work_qubits), 0
                        )
                    self.assertEqual(
                        fired, min(_path_values(model, digits)[2], cap)
                    )

    def test_complete_252_date_collapsed_resource_ledger(self) -> None:
        spec = AsianGridSpec(
            n_dates=252,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.5, 0.5),
            s0=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            shock_scale=1,
            price_scale=1024,
            residual_payoff_cap=2.864,
            geometric_leg="collapsed",
        )
        model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=18)
        estimate = estimate_arithmetic_asian_resources(model)
        self.assertEqual(
            (
                model.value_bits,
                model.product_bits,
                model.geometric_product_bits,
                model.shock_weight_bits,
                model.total_bits,
                model.residual_bits,
                model.threshold_bits,
            ),
            (22, 41, 43, 15, 28, 28, 20),
        )
        # the weighted-sum dynamic program is exact and enumerates every
        # reachable sum, so no calibrated control override is needed here
        self.assertEqual(model.geometric_dp_peak_states, 31_879)
        self.assertEqual(estimate.a_qubits, 11_533)
        self.assertEqual(estimate.a_counts.ccx, 9_823_413)
        self.assertEqual(estimate.a_counts.t, 68_763_891)
        self.assertEqual(estimate.q_counts.t, 137_689_209)
        self.assertEqual(
            estimate.component_counts["geometric_weighted_sum_in_compute"].ccx,
            37_800,
        )
        self.assertEqual(
            estimate.component_counts["geometric_exponential_chain_in_compute"].ccx,
            285_120,
        )

    def test_complete_252_date_none_resource_ledger(self) -> None:
        """The raw oracle at the production accuracy-qualified configuration.

        This is the control-free baseline the paper compares against: same
        arithmetic leg as the residual oracle, arithmetic payoff encoded
        directly, threshold register sized from the raw cap of 45.85 dollars
        (28 bits) rather than the residual cap of 2.864 dollars (24 bits).
        The collapsed residual oracle is rebuilt here at the identical grid so
        the per-query control overhead the paper prints is locked to both
        ledgers rather than to a hand-copied number.
        """

        spec = AsianGridSpec(
            n_dates=252,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.5, 0.5),
            s0=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            shock_scale=1,
            price_scale=16384,
            payoff_cap=45.85,
            geometric_leg="none",
        )
        model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=30)
        estimate = estimate_arithmetic_asian_resources(model)
        self.assertEqual(
            (
                model.value_bits,
                model.product_bits,
                model.geometric_product_bits,
                model.shock_weight_bits,
                model.total_bits,
                model.residual_bits,
                model.threshold_bits,
            ),
            (26, 57, 0, 0, 32, 32, 28),
        )
        self.assertEqual(model.requested_residual_cap_numerator, 189_304_013)
        self.assertEqual(model.geometric_dp_peak_states, 0)
        self.assertEqual(estimate.a_qubits, 14_837)
        self.assertEqual(estimate.a_counts.ccx, 15_024_057)
        self.assertEqual(estimate.a_counts.t, 105_168_399)
        self.assertEqual(estimate.q_counts.t, 210_544_481)
        self.assertEqual(estimate.qrom_rows, 0)
        self.assertEqual(estimate.arbitrary_rotations, 0)

        residual_model = build_arithmetic_asian_model(
            replace(
                spec,
                payoff_cap=None,
                residual_payoff_cap=2.864,
                geometric_leg="collapsed",
            ),
            multiplier_fraction_bits=30,
        )
        residual_estimate = estimate_arithmetic_asian_resources(residual_model)
        self.assertEqual(residual_model.threshold_bits, 24)
        self.assertEqual(residual_estimate.a_counts.ccx, 16_039_589)
        self.assertEqual(residual_estimate.a_counts.t, 112_277_123)
        self.assertEqual(residual_estimate.a_qubits, 15_857)
        # the arithmetic leg is shared byte for byte; only the control differs
        for shared in (
            "price_selected_multipliers_in_compute",
            "arithmetic_price_sum_in_compute",
            "arithmetic_positive_part_in_compute",
        ):
            self.assertEqual(
                estimate.component_counts[shared],
                residual_estimate.component_counts[shared],
            )
        self.assertAlmostEqual(
            residual_estimate.a_counts.t / estimate.a_counts.t, 1.0676, places=4
        )
        self.assertAlmostEqual(
            residual_estimate.a_qubits / estimate.a_qubits, 1.0687, places=4
        )

    def test_complete_252_date_resource_ledger(self) -> None:
        spec = AsianGridSpec(
            n_dates=252,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.5, 0.5),
            s0=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            shock_scale=1,
            price_scale=1024,
            residual_payoff_cap=2.864,
        )
        model = build_arithmetic_asian_model(
            spec, multiplier_fraction_bits=18
        )
        estimate = estimate_arithmetic_asian_resources(model)
        self.assertEqual(
            (
                model.value_bits,
                model.product_bits,
                model.total_bits,
                model.residual_bits,
                model.threshold_bits,
            ),
            (22, 41, 28, 28, 20),
        )
        self.assertEqual(estimate.a_qubits, 21_203)
        self.assertEqual(estimate.a_counts.ccx, 18_314_085)
        self.assertEqual(estimate.a_counts.t, 128_198_595)
        self.assertEqual(estimate.q_counts.t, 256_693_997)
        self.assertEqual(estimate.qrom_rows, 0)
        self.assertEqual(estimate.arbitrary_rotations, 0)


if __name__ == "__main__":
    unittest.main()
