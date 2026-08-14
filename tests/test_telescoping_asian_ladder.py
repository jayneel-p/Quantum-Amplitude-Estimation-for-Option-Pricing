"""Executable and independent checks for the two-block control ladder."""

from __future__ import annotations

import itertools
import math
import unittest
from dataclasses import replace

import numpy as np

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    arithmetic_objective_probability_from_mps,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec
from qc_option_pricing.quantum.k2_ladder_pilot import (
    _SortedCapSample,
    _smallest_cap_for_bias,
    sample_encoded_increments,
)
from qc_option_pricing.quantum.telescoping_asian_ladder import (
    _partition_common_numerator,
    build_k2_ladder_model,
    build_k2_ladder_oracle,
    enumerate_k2_ladder_increment,
    estimate_k2_ladder_resources,
    iter_k2_ladder_path_values,
    k2_ladder_path_values,
    k2_ladder_price_from_probabilities,
    primitive_counts_from_k2_ladder_circuit,
)


class K2TelescopingAsianLadderTests(unittest.TestCase):
    @staticmethod
    def _spec(n_dates: int = 4, price_scale: int = 8) -> AsianGridSpec:
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

    def test_uncorrected_rounding_breaks_order_but_certified_shift_repairs_it(
        self,
    ) -> None:
        model = build_k2_ladder_model(
            self._spec(), "coarse_to_blocked", multiplier_fraction_bits=8
        )
        partitions = model.shared.partition_map
        unshifted_coarse = replace(
            partitions[1], strike_adjustment_units=0
        )
        unshifted_increments = []
        shifted_increments = []
        for digits in itertools.product(range(2), repeat=model.spec.n_dates):
            blocked = _partition_common_numerator(
                model.shared, partitions[2], digits
            )
            unshifted_coarse_value = _partition_common_numerator(
                model.shared, unshifted_coarse, digits
            )
            shifted_coarse_value = _partition_common_numerator(
                model.shared, partitions[1], digits
            )
            unshifted_increments.append(blocked - unshifted_coarse_value)
            shifted_increments.append(blocked - shifted_coarse_value)
        self.assertLess(min(unshifted_increments), 0)
        self.assertGreaterEqual(min(shifted_increments), 0)
        self.assertGreater(model.shared.coarse_strike_adjustment_units, 0)

    def test_every_small_path_is_ordered_and_telescopes_exactly(self) -> None:
        for n_dates in (2, 4, 6, 8):
            model = build_k2_ladder_model(
                self._spec(n_dates=n_dates, price_scale=16),
                "coarse_to_blocked",
                multiplier_fraction_bits=10,
            )
            with self.subTest(n_dates=n_dates):
                for _, values in iter_k2_ladder_path_values(model.shared):
                    self.assertGreaterEqual(values.coarse_control, 0)
                    self.assertGreaterEqual(
                        values.blocked_control, values.coarse_control
                    )
                    self.assertGreaterEqual(values.target, values.blocked_control)
                    self.assertEqual(
                        values.target,
                        values.coarse_control
                        + values.coarse_to_blocked
                        + values.blocked_to_target,
                    )

    def test_increment_expectations_equal_the_direct_residual(self) -> None:
        spec = self._spec(n_dates=6, price_scale=16)
        models = {
            increment: build_k2_ladder_model(
                spec, increment, multiplier_fraction_bits=10
            )
            for increment in (
                "coarse_to_blocked",
                "blocked_to_target",
                "coarse_to_target",
            )
        }
        references = {
            increment: enumerate_k2_ladder_increment(model)
            for increment, model in models.items()
        }
        self.assertAlmostEqual(
            references["coarse_to_blocked"].increment_undiscounted
            + references["blocked_to_target"].increment_undiscounted,
            references["coarse_to_target"].increment_undiscounted,
            places=15,
        )

    def test_price_reconstruction_from_two_objective_probabilities(self) -> None:
        spec = self._spec(n_dates=4, price_scale=16)
        first_model = build_k2_ladder_model(
            spec, "coarse_to_blocked", multiplier_fraction_bits=10
        )
        second_model = build_k2_ladder_model(
            spec, "blocked_to_target", multiplier_fraction_bits=10
        )
        first_reference = enumerate_k2_ladder_increment(first_model)
        second_reference = enumerate_k2_ladder_increment(second_model)
        first_oracle = build_k2_ladder_oracle(first_model)
        second_oracle = build_k2_ladder_oracle(second_model)
        reconstructed = k2_ladder_price_from_probabilities(
            first_oracle,
            first_reference.objective_probability,
            second_oracle,
            second_reference.objective_probability,
        )
        target_mean = math.fsum(
            values.target
            for _, values in iter_k2_ladder_path_values(first_model.shared)
        ) / ((1 << spec.n_dates) * spec.n_dates * spec.price_scale)
        expected = math.exp(-spec.rate * spec.maturity) * target_mean
        self.assertAlmostEqual(reconstructed, expected, places=14)

    def test_executable_oracles_match_enumeration_and_clean_work(self) -> None:
        spec = self._spec(n_dates=2, price_scale=4)
        for increment in (
            "coarse_to_blocked",
            "blocked_to_target",
            "coarse_to_target",
        ):
            with self.subTest(increment=increment):
                model = build_k2_ladder_model(
                    spec, increment, multiplier_fraction_bits=4
                )
                reference = enumerate_k2_ladder_increment(model)
                oracle = build_k2_ladder_oracle(model)
                probability, work_hamming = arithmetic_objective_probability_from_mps(
                    oracle
                )
                self.assertAlmostEqual(
                    probability, reference.objective_probability, delta=2e-10
                )
                self.assertLess(work_hamming, 1e-10)

    def test_compositional_counts_equal_transpiled_small_circuits(self) -> None:
        spec = self._spec(n_dates=2, price_scale=4)
        for increment in (
            "coarse_to_blocked",
            "blocked_to_target",
            "coarse_to_target",
        ):
            with self.subTest(increment=increment):
                model = build_k2_ladder_model(
                    spec, increment, multiplier_fraction_bits=4
                )
                oracle = build_k2_ladder_oracle(model)
                actual = primitive_counts_from_k2_ladder_circuit(oracle)
                estimated = estimate_k2_ladder_resources(model)
                self.assertEqual(actual, estimated.a_counts)
                self.assertEqual(oracle.circuit.num_qubits, estimated.a_qubits)
                self.assertEqual(estimated.qrom_rows, 0)
                self.assertEqual(estimated.arbitrary_rotations, 0)

    def test_production_rounding_certificate_is_locked(self) -> None:
        spec = AsianGridSpec(
            n_dates=252,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.5, 0.5),
            s0=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            maturity=1.0,
            shock_scale=1,
            price_scale=16384,
            geometric_leg="collapsed",
        )
        model = build_k2_ladder_model(
            spec, "coarse_to_blocked", multiplier_fraction_bits=30
        )
        fine_bounds = tuple(
            block.rounding_error_bound_units
            for block in model.shared.partition_map[2].blocks
        )
        self.assertEqual(fine_bounds, (32, 790))
        self.assertEqual(model.shared.coarse_strike_adjustment_units, 411)
        self.assertAlmostEqual(
            model.shared.coarse_strike_adjustment_dollars,
            411 / 16384,
            places=15,
        )
        resources = estimate_k2_ladder_resources(model)
        self.assertGreater(resources.a_counts.t, 0)
        self.assertGreater(resources.a_qubits, spec.n_dates)

    def test_vectorized_pilot_matches_scalar_integer_recurrences(self) -> None:
        spec = self._spec(n_dates=4, price_scale=16)
        shared = build_k2_ladder_model(
            spec, "coarse_to_blocked", multiplier_fraction_bits=10
        ).shared
        paths, seed = 32, 20260730
        vectorized = sample_encoded_increments(shared, paths=paths, seed=seed)
        rng = np.random.default_rng(seed)
        draws = [
            rng.integers(0, 2, size=paths, dtype=np.int8)
            for _ in range(spec.n_dates)
        ]
        scalar = [
            k2_ladder_path_values(
                shared, tuple(int(draws[date][path]) for date in range(spec.n_dates))
            )
            for path in range(paths)
        ]
        np.testing.assert_array_equal(
            vectorized.coarse_to_blocked,
            [values.coarse_to_blocked for values in scalar],
        )
        np.testing.assert_array_equal(
            vectorized.blocked_to_target,
            [values.blocked_to_target for values in scalar],
        )
        np.testing.assert_array_equal(
            vectorized.coarse_to_target,
            [values.coarse_to_target for values in scalar],
        )
        unshifted = replace(
            shared.partition_map[1], strike_adjustment_units=0
        )
        expected_uncorrected = []
        for path in range(paths):
            digits = tuple(
                int(draws[date][path]) for date in range(spec.n_dates)
            )
            expected_uncorrected.append(
                _partition_common_numerator(shared, shared.partition_map[2], digits)
                - _partition_common_numerator(shared, unshifted, digits)
            )
        np.testing.assert_array_equal(
            vectorized.uncorrected_coarse_to_blocked,
            expected_uncorrected,
        )

    def test_cap_selector_is_minimal_and_confidence_margin_is_monotone(self) -> None:
        values = np.asarray([0, 0, 2, 3, 5, 8, 13, 21], dtype=np.int64)
        sample = _SortedCapSample.from_values(values)
        budget = 1.2
        selected = _smallest_cap_for_bias(
            sample,
            discounted_budget=budget,
            denominator=1,
            discount=1.0,
            selection_z=0.0,
        )
        brute_force = min(
            cap
            for cap in range(1, int(values.max()) + 1)
            if float(np.maximum(values - cap, 0).mean()) <= budget
        )
        self.assertEqual(selected, brute_force)
        confidence_selected = _smallest_cap_for_bias(
            sample,
            discounted_budget=budget,
            denominator=1,
            discount=1.0,
            selection_z=1.96,
        )
        self.assertGreaterEqual(confidence_selected, selected)

    def test_invalid_ladder_specifications_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "even number"):
            build_k2_ladder_model(
                self._spec(n_dates=3),
                "coarse_to_blocked",
                multiplier_fraction_bits=8,
            )
        four_point = replace(
            self._spec(),
            shock_points=(-1.5, -0.5, 0.5, 1.5),
            shock_probabilities=(0.25,) * 4,
        )
        with self.assertRaisesRegex(ValueError, "binary shock"):
            build_k2_ladder_model(
                four_point,
                "coarse_to_blocked",
                multiplier_fraction_bits=8,
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_k2_ladder_model(
                self._spec(), "not-a-level", multiplier_fraction_bits=8
            )


if __name__ == "__main__":
    unittest.main()
