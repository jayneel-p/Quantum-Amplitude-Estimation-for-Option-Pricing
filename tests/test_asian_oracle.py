"""Correctness tests for the complete finite-grid Asian oracle.

Run with:
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from qc_option_pricing.quantum.asian_oracle import (
    AsianGridSpec,
    build_asian_model,
    build_asian_oracle,
    enumerate_encoded_asian,
    estimate_asian_oracle_resources,
    gauss_hermite_normal_grid,
    inverse_roundtrip_leakage_from_mps,
    objective_probability_from_mps,
    objective_probability_from_statevector,
)


def _independent_path_reference(spec: AsianGridSpec) -> dict[str, float]:
    """Direct formulas; deliberately does not read any compiled model table."""
    dt = spec.maturity / spec.n_dates
    drift = spec.rate - 0.5 * spec.volatility**2
    shock_integers = tuple(round(point * spec.shock_scale) for point in spec.shock_points)
    strike_integer = round(spec.strike * spec.price_scale)
    arithmetic = 0.0
    geometric = 0.0
    residual = 0.0
    minimum_residual = math.inf
    maximum_raw_numerator = 0
    path_data: list[tuple[float, int, int]] = []

    for digits in itertools.product(range(len(shock_integers)), repeat=spec.n_dates):
        probability = math.prod(spec.shock_probabilities[digit] for digit in digits)
        cumulative = 0
        total = 0
        weighted = 0
        for date, digit in enumerate(digits, start=1):
            shock = shock_integers[digit]
            cumulative += shock
            stock = spec.s0 * math.exp(
                drift * date * dt
                + spec.volatility * math.sqrt(dt) * cumulative / spec.shock_scale
            )
            total += math.ceil(stock * spec.price_scale - 1e-12)
            weighted += (spec.n_dates - date + 1) * shock
        geometric_stock = spec.s0 * math.exp(
            drift * dt * (spec.n_dates + 1) / 2
            + spec.volatility
            * math.sqrt(dt)
            * weighted
            / (spec.n_dates * spec.shock_scale)
        )
        geometric_integer = math.floor(geometric_stock * spec.price_scale + 1e-12)
        arithmetic_numerator = max(total - spec.n_dates * strike_integer, 0)
        geometric_numerator = spec.n_dates * max(geometric_integer - strike_integer, 0)
        residual_numerator = arithmetic_numerator - geometric_numerator
        minimum_residual = min(minimum_residual, residual_numerator)
        maximum_raw_numerator = max(maximum_raw_numerator, arithmetic_numerator)
        denominator = spec.n_dates * spec.price_scale
        arithmetic += probability * arithmetic_numerator / denominator
        geometric += probability * geometric_numerator / denominator
        residual += probability * residual_numerator / denominator
        path_data.append((probability, arithmetic_numerator, residual_numerator))

    raw_cap = (
        maximum_raw_numerator
        if spec.payoff_cap is None
        else round(spec.payoff_cap * spec.n_dates * spec.price_scale)
    )
    if raw_cap == 0:
        raise ValueError("the encoded payoff is identically zero on the configured range")
    raw_probability = sum(p * min(a, raw_cap) / raw_cap for p, a, _ in path_data)
    residual_cap = (
        raw_cap
        if spec.residual_payoff_cap is None
        else round(spec.residual_payoff_cap * spec.n_dates * spec.price_scale)
    )
    if residual_cap == 0:
        raise ValueError("the encoded residual is identically zero on the configured range")
    qcv_probability = sum(p * min(r, residual_cap) / residual_cap for p, _, r in path_data)
    return {
        "arithmetic": arithmetic,
        "geometric": geometric,
        "residual": residual,
        "minimum_residual": minimum_residual,
        "raw_probability": raw_probability,
        "qcv_probability": qcv_probability,
    }


class TestAsianOracle(unittest.TestCase):
    @staticmethod
    def _spec(payoff_cap: float | None = None) -> AsianGridSpec:
        return AsianGridSpec(
            n_dates=2,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.4, 0.6),
            s0=2.0,
            strike=1.0,
            rate=0.0,
            volatility=0.3,
            maturity=1.0,
            shock_scale=1,
            price_scale=1,
            payoff_cap=payoff_cap,
        )

    def test_gauss_hermite_rule_has_normal_moments(self) -> None:
        points, probabilities = gauss_hermite_normal_grid(2)
        points_array = np.asarray(points)
        probabilities_array = np.asarray(probabilities)
        self.assertAlmostEqual(float(probabilities_array.sum()), 1.0, places=15)
        self.assertAlmostEqual(float(probabilities_array @ points_array), 0.0, places=15)
        self.assertAlmostEqual(float(probabilities_array @ points_array**2), 1.0, places=14)
        self.assertAlmostEqual(float(probabilities_array @ points_array**3), 0.0, places=14)
        self.assertAlmostEqual(float(probabilities_array @ points_array**4), 3.0, places=13)

    def test_compiled_tables_match_independent_formulas(self) -> None:
        spec = self._spec()
        model = build_asian_model(spec)
        direct = _independent_path_reference(spec)
        enumerated = enumerate_encoded_asian(model)
        self.assertAlmostEqual(enumerated.probability_mass, 1.0, places=15)
        self.assertAlmostEqual(
            enumerated.arithmetic_payoff_undiscounted, direct["arithmetic"], places=14
        )
        self.assertAlmostEqual(
            enumerated.geometric_payoff_undiscounted, direct["geometric"], places=14
        )
        self.assertAlmostEqual(
            enumerated.residual_payoff_undiscounted, direct["residual"], places=14
        )
        self.assertGreaterEqual(direct["minimum_residual"], 0)
        self.assertAlmostEqual(
            enumerated.arithmetic_payoff_undiscounted,
            enumerated.geometric_payoff_undiscounted
            + enumerated.residual_payoff_undiscounted,
            places=14,
        )

    def test_raw_circuit_matches_direct_path_sum_and_cleans_work(self) -> None:
        spec = self._spec()
        direct = _independent_path_reference(spec)
        oracle = build_asian_oracle(spec, "raw")
        probability, work_leakage = objective_probability_from_statevector(oracle)
        self.assertAlmostEqual(probability, direct["raw_probability"], places=11)
        self.assertLess(work_leakage, 1e-12)
        expected_price = math.exp(-spec.rate * spec.maturity) * direct["arithmetic"]
        self.assertAlmostEqual(oracle.post_process(probability), expected_price, places=11)

    def test_qcv_circuit_matches_direct_path_sum_and_cleans_work(self) -> None:
        spec = self._spec()
        direct = _independent_path_reference(spec)
        oracle = build_asian_oracle(spec, "qcv")
        probability, work_leakage = objective_probability_from_mps(oracle)
        self.assertAlmostEqual(probability, direct["qcv_probability"], places=8)
        self.assertLess(work_leakage, 1e-9)
        self.assertAlmostEqual(
            oracle.geometric_control_undiscounted, direct["geometric"], places=14
        )
        expected_price = math.exp(-spec.rate * spec.maturity) * direct["arithmetic"]
        self.assertAlmostEqual(oracle.post_process(probability), expected_price, places=8)

    def test_factorized_qcv_matches_reference_and_cleans_work(self) -> None:
        spec = self._spec()
        direct = _independent_path_reference(spec)
        oracle = build_asian_oracle(
            spec, "qcv", residual_method="factorized_arithmetic"
        )
        probability, work_leakage = objective_probability_from_mps(oracle)
        self.assertAlmostEqual(probability, direct["qcv_probability"], places=8)
        self.assertLess(work_leakage, 1e-9)
        self.assertEqual(oracle.residual_method, "factorized_arithmetic")

    def test_explicit_cap_has_exact_clipped_semantics(self) -> None:
        spec = self._spec(payoff_cap=0.5)
        direct = _independent_path_reference(spec)
        raw = build_asian_oracle(spec, "raw")
        qcv = build_asian_oracle(spec, "qcv")
        raw_probability, raw_leakage = objective_probability_from_statevector(raw)
        qcv_probability, qcv_leakage = objective_probability_from_mps(qcv)
        self.assertAlmostEqual(raw_probability, direct["raw_probability"], places=11)
        self.assertAlmostEqual(qcv_probability, direct["qcv_probability"], places=8)
        self.assertLess(raw_leakage, 1e-12)
        self.assertLess(qcv_leakage, 1e-9)

    def test_independent_residual_cap_has_exact_clipped_semantics(self) -> None:
        base = self._spec()
        spec = AsianGridSpec(
            **{
                **base.__dict__,
                "payoff_cap": 1.0,
                "residual_payoff_cap": 0.5,
            }
        )
        direct = _independent_path_reference(spec)
        qcv = build_asian_oracle(spec, "qcv")
        probability, leakage = objective_probability_from_mps(qcv)
        self.assertAlmostEqual(probability, direct["qcv_probability"], places=8)
        self.assertLess(leakage, 1e-9)

    def test_invalid_scalar_and_grid_inputs_are_rejected(self) -> None:
        base = self._spec().__dict__
        invalid_overrides = (
            {"n_dates": 1.5},
            {"n_dates": True},
            {"shock_scale": 1.5},
            {"price_scale": True},
            {"s0": math.nan},
            {"strike": math.inf},
            {"rate": math.nan},
            {"volatility": math.inf},
            {"maturity": math.nan},
            {"payoff_cap": math.nan},
            {"residual_payoff_cap": math.inf},
        )
        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                AsianGridSpec(**{**base, **override})

    def test_qcv_a_inverse_roundtrip(self) -> None:
        oracle = build_asian_oracle(self._spec(), "qcv")
        self.assertLess(inverse_roundtrip_leakage_from_mps(oracle), 1e-9)

    def test_resource_estimator_matches_small_oracle_structure(self) -> None:
        model = build_asian_model(self._spec())
        raw = build_asian_oracle(model, "raw")
        qcv = build_asian_oracle(model, "qcv")
        raw_estimate = estimate_asian_oracle_resources(model, "raw")
        qcv_estimate = estimate_asian_oracle_resources(model, "qcv")
        self.assertEqual(raw_estimate.total_qubits, raw.circuit.num_qubits)
        self.assertEqual(raw_estimate.lookup_rows, raw.lookup_rows)
        self.assertEqual(raw_estimate.lookup_bit_toggles, raw.lookup_bit_toggles)
        self.assertEqual(raw_estimate.controlled_rotations, raw.controlled_rotations)
        self.assertEqual(qcv_estimate.total_qubits, qcv.circuit.num_qubits)
        self.assertGreaterEqual(qcv_estimate.lookup_rows, qcv.lookup_rows)
        self.assertGreaterEqual(qcv_estimate.lookup_bit_toggles, qcv.lookup_bit_toggles)

        factorized = build_asian_oracle(
            model, "qcv", residual_method="factorized_arithmetic"
        )
        factorized_estimate = estimate_asian_oracle_resources(
            model, "qcv", residual_method="factorized_arithmetic"
        )
        self.assertEqual(factorized_estimate.total_qubits, factorized.circuit.num_qubits)
        self.assertEqual(factorized_estimate.lookup_rows, factorized.lookup_rows)
        self.assertEqual(
            factorized_estimate.lookup_bit_toggles, factorized.lookup_bit_toggles
        )

    def test_252_date_structural_estimate_has_no_register_overflow(self) -> None:
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
            price_scale=1,
        )
        model = build_asian_model(spec)
        self.assertGreaterEqual(model.prefix_min, -(1 << (model.prefix_bits - 1)))
        self.assertLessEqual(model.prefix_max, (1 << (model.prefix_bits - 1)) - 1)
        self.assertGreaterEqual(model.weighted_min, -(1 << (model.weighted_bits - 1)))
        self.assertLessEqual(model.weighted_max, (1 << (model.weighted_bits - 1)) - 1)
        self.assertLess(model.maximum_total, 1 << model.total_bits)
        raw = estimate_asian_oracle_resources(model, "raw")
        qcv = estimate_asian_oracle_resources(model, "qcv")
        factorized = estimate_asian_oracle_resources(
            model, "qcv", residual_method="factorized_arithmetic"
        )
        self.assertEqual((raw.total_qubits, raw.lookup_rows), (307, 258_048))
        self.assertEqual((qcv.total_qubits, qcv.lookup_rows), (357, 23_883_377_236))
        self.assertEqual((factorized.total_qubits, factorized.lookup_rows), (375, 648_220))
        self.assertLess(factorized.lookup_rows, qcv.lookup_rows / 30_000)

    def test_randomized_discrete_am_gm_stress(self) -> None:
        rng = np.random.default_rng(20260713)
        for _ in range(100):
            n_dates = int(rng.integers(2, 6))
            volatility = float(rng.uniform(0.05, 0.9))
            rate = float(rng.uniform(-0.05, 0.15))
            s0 = float(rng.integers(2, 8))
            strike = float(rng.integers(0, 8))
            first_probability = float(rng.uniform(0.1, 0.9))
            spec = AsianGridSpec(
                n_dates=n_dates,
                shock_points=(-1.0, 1.0),
                shock_probabilities=(first_probability, 1.0 - first_probability),
                s0=s0,
                strike=strike,
                rate=rate,
                volatility=volatility,
                maturity=float(rng.uniform(0.25, 2.0)),
                shock_scale=1,
                price_scale=1,
            )
            try:
                direct = _independent_path_reference(spec)
                enumerated = enumerate_encoded_asian(spec)
            except ValueError as exc:
                # Some random ranges are identically OTM and intentionally
                # rejected because no payoff normalization exists.
                self.assertIn("identically zero", str(exc))
                continue
            self.assertGreaterEqual(direct["minimum_residual"], 0)
            self.assertGreaterEqual(enumerated.minimum_residual_numerator, 0)
            self.assertAlmostEqual(
                direct["arithmetic"], direct["geometric"] + direct["residual"], places=12
            )


if __name__ == "__main__":
    unittest.main()
