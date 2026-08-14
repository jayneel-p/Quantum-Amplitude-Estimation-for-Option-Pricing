"""Independent and circuit-level checks for the general blocked oracle."""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np
from scipy.stats import norm

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    arithmetic_objective_probability_from_mps,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec
from qc_option_pricing.quantum.blocked_asian_oracle import (
    black_scholes_binary_spec,
    blocked_asian_payoffs_from_digits,
    blocked_asian_path_values,
    build_black_scholes_blocked_model,
    build_blocked_asian_model,
    build_blocked_asian_oracle,
    enumerate_blocked_asian,
    estimate_block_control_price_rqmc,
    estimate_encoded_block_control_price_rqmc,
    estimate_blocked_asian_resources,
    primitive_counts_from_blocked_asian_circuit,
    sample_blocked_residuals,
)
from qc_option_pricing.quantum.telescoping_asian_ladder import (
    build_k2_ladder_model,
    estimate_k2_ladder_resources,
    k2_ladder_path_values,
)


class GeneralBlockedAsianOracleTests(unittest.TestCase):
    @staticmethod
    def _spec(n_dates: int = 6, price_scale: int = 16) -> AsianGridSpec:
        return AsianGridSpec(
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
            geometric_leg="collapsed",
        )

    def test_block_count_validation(self) -> None:
        spec = self._spec()
        for invalid in (0, True, 4, 7):
            with self.subTest(block_count=invalid):
                with self.assertRaises(ValueError):
                    build_blocked_asian_model(spec, invalid)

    def test_every_small_path_is_nonnegative_and_reconstructs_target(self) -> None:
        spec = self._spec()
        for block_count in (1, 2, 3, 6):
            model = build_blocked_asian_model(
                spec,
                block_count,
                multiplier_fraction_bits=10,
            )
            with self.subTest(block_count=block_count):
                self.assertTrue(
                    all(
                        certificate.rounds_down_everywhere
                        for certificate in model.shared.rounding_certificates
                    )
                )
                for digits in itertools.product(range(2), repeat=spec.n_dates):
                    values = blocked_asian_path_values(model.shared, digits)
                    self.assertGreaterEqual(values.control, 0)
                    self.assertGreaterEqual(values.residual, 0)
                    self.assertEqual(values.target, values.control + values.residual)

    def test_general_k2_matches_frozen_k2_path_values_and_resources(self) -> None:
        spec = self._spec(n_dates=4)
        general = build_blocked_asian_model(
            spec,
            2,
            multiplier_fraction_bits=10,
            residual_cap_dollars=0.75,
        )
        reference = build_k2_ladder_model(
            spec,
            "blocked_to_target",
            multiplier_fraction_bits=10,
            increment_cap_dollars=0.75,
        )
        for digits in itertools.product(range(2), repeat=spec.n_dates):
            observed = blocked_asian_path_values(general.shared, digits)
            expected = k2_ladder_path_values(reference.shared, digits)
            self.assertEqual(observed.target, expected.target)
            self.assertEqual(observed.control, expected.blocked_control)
            self.assertEqual(observed.residual, expected.blocked_to_target)
        general_resources = estimate_blocked_asian_resources(general)
        reference_resources = estimate_k2_ladder_resources(reference)
        self.assertEqual(general_resources.a_counts, reference_resources.a_counts)
        self.assertEqual(general_resources.q_counts, reference_resources.q_counts)
        self.assertEqual(general_resources.a_qubits, reference_resources.a_qubits)
        self.assertEqual(
            general_resources.q_qubits_with_clean_reflection_ladder,
            reference_resources.q_qubits_with_clean_reflection_ladder,
        )

    def test_vectorized_sampler_matches_scalar_recurrence(self) -> None:
        model = build_blocked_asian_model(
            self._spec(n_dates=6),
            3,
            multiplier_fraction_bits=10,
        )
        observed = sample_blocked_residuals(
            model.shared,
            paths=37,
            seed=9876,
            chunk=37,
        )
        rng = np.random.default_rng(9876)
        draws = [
            rng.integers(0, 2, size=37, dtype=np.int8)
            for _ in range(model.spec.n_dates)
        ]
        expected = [
            blocked_asian_path_values(
                model.shared,
                tuple(int(draws[date][path]) for date in range(model.spec.n_dates)),
            ).residual
            for path in range(37)
        ]
        self.assertEqual(observed.tolist(), expected)

    def test_explicit_bit_matrix_matches_scalar_and_rqmc_exact_small_grid(self) -> None:
        model = build_blocked_asian_model(
            self._spec(n_dates=3, price_scale=8),
            3,
            multiplier_fraction_bits=8,
        )
        digits = np.asarray(list(itertools.product(range(2), repeat=3)), dtype=np.int8)
        vectorized = blocked_asian_payoffs_from_digits(model.shared, digits)
        scalar = [
            blocked_asian_path_values(model.shared, tuple(int(bit) for bit in row))
            for row in digits
        ]
        self.assertEqual(vectorized.target.tolist(), [value.target for value in scalar])
        self.assertEqual(vectorized.control.tolist(), [value.control for value in scalar])
        self.assertEqual(vectorized.residual.tolist(), [value.residual for value in scalar])
        exact = enumerate_blocked_asian(model)
        estimate = estimate_encoded_block_control_price_rqmc(
            model.shared,
            log2_paths=3,
            replicates=4,
            seed=2468,
        )
        discounted_exact = math.exp(-model.spec.rate * model.spec.maturity) * (
            exact.control_undiscounted
        )
        self.assertAlmostEqual(estimate.discounted_mean, discounted_exact, places=14)
        self.assertLess(estimate.discounted_standard_error, 1e-14)

    def test_compositional_counts_equal_transpiled_small_circuits(self) -> None:
        spec = self._spec(n_dates=3, price_scale=4)
        for block_count in (1, 3):
            with self.subTest(block_count=block_count):
                model = build_blocked_asian_model(
                    spec,
                    block_count,
                    multiplier_fraction_bits=4,
                )
                oracle = build_blocked_asian_oracle(model)
                counted = primitive_counts_from_blocked_asian_circuit(oracle)
                estimated = estimate_blocked_asian_resources(model)
                self.assertEqual(counted, estimated.a_counts)
                self.assertEqual(oracle.circuit.num_qubits, estimated.a_qubits)
                self.assertEqual(estimated.qrom_rows, 0)
                self.assertEqual(estimated.arbitrary_rotations, 0)

    def test_executable_oracle_matches_enumeration_and_cleans_work(self) -> None:
        model = build_blocked_asian_model(
            self._spec(n_dates=3, price_scale=4),
            3,
            multiplier_fraction_bits=4,
            residual_cap_dollars=1.0,
        )
        reference = enumerate_blocked_asian(model)
        oracle = build_blocked_asian_oracle(model)
        probability, work_hamming = arithmetic_objective_probability_from_mps(oracle)
        self.assertAlmostEqual(
            probability,
            reference.objective_probability,
            delta=2e-10,
        )
        self.assertLess(work_hamming, 1e-10)
        decoded = oracle.post_process(
            probability,
            control_undiscounted=reference.control_undiscounted,
        )
        expected = math.exp(-model.spec.rate * model.spec.maturity) * (
            reference.control_undiscounted
            + reference.clipped_residual_undiscounted
        )
        price_tolerance = (
            math.exp(-model.spec.rate * model.spec.maturity)
            * model.normalization_dollars
            * 2e-10
        )
        self.assertAlmostEqual(decoded, expected, delta=price_tolerance)

    def test_convenience_interface_exposes_bs_and_encoding_inputs(self) -> None:
        model = build_black_scholes_blocked_model(
            n_dates=12,
            block_count=3,
            s0=90.0,
            strike=95.0,
            rate=0.02,
            volatility=0.3,
            maturity=2.0,
            price_scale=1024,
            multiplier_fraction_bits=18,
            residual_cap_dollars=2.0,
        )
        self.assertEqual(model.block_count, 3)
        self.assertEqual(model.spec.n_dates, 12)
        self.assertEqual(model.spec.s0, 90.0)
        self.assertEqual(model.spec.volatility, 0.3)
        self.assertEqual(model.requested_cap_dollars, 2.0)

    def test_rqmc_k1_agrees_with_independent_geometric_asian_formula(self) -> None:
        spec = black_scholes_binary_spec(
            n_dates=12,
            s0=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
        )
        estimate = estimate_block_control_price_rqmc(
            spec,
            1,
            log2_points=14,
            replicates=8,
            seed=1234,
        )
        n = spec.n_dates
        mean = (
            math.log(spec.s0)
            + (spec.rate - 0.5 * spec.volatility**2)
            * spec.maturity
            * (n + 1)
            / (2 * n)
        )
        variance = (
            spec.volatility**2
            * spec.maturity
            * (n + 1)
            * (2 * n + 1)
            / (6 * n * n)
        )
        standard_deviation = math.sqrt(variance)
        d2 = (mean - math.log(spec.strike)) / standard_deviation
        d1 = d2 + standard_deviation
        expected = math.exp(-spec.rate * spec.maturity) * (
            math.exp(mean + 0.5 * variance) * norm.cdf(d1)
            - spec.strike * norm.cdf(d2)
        )
        self.assertAlmostEqual(estimate.discounted_mean, expected, delta=2e-4)

    def test_zero_volatility_control_price_is_deterministic(self) -> None:
        spec = black_scholes_binary_spec(
            n_dates=12,
            s0=100.0,
            strike=90.0,
            rate=0.05,
            volatility=0.0,
            maturity=1.0,
        )
        estimate = estimate_block_control_price_rqmc(
            spec,
            3,
            log2_points=4,
            replicates=4,
        )
        self.assertEqual(estimate.discounted_standard_error, 0.0)
        self.assertEqual(
            estimate.discounted_ci95_low,
            estimate.discounted_ci95_high,
        )


if __name__ == "__main__":
    unittest.main()
