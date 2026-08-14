"""Correctness tests for the Wang-Kan weak-Euler Heston model.

Run with:
    .venv/bin/python -m unittest tests.test_heston_weak_euler -v
"""

from __future__ import annotations

import math
import unittest
from itertools import product

import numpy as np

from qc_option_pricing.classical.heston_weak_euler import (
    HestonWeakEulerSpec,
    enumerate_weak_euler,
    simulate_weak_euler_payoffs,
    variance_positivity_certificate,
    weak_euler_path_from_signs,
)

CALL_1 = HestonWeakEulerSpec(
    s0=100.0, v0=0.1, rate=0.03, rho=-0.1, kappa=2.0, theta=0.12, xi=0.3,
    maturity=1.0, n_steps=256, strike=90.0, option_type="call",
)
PUT_2 = HestonWeakEulerSpec(
    s0=100.0, v0=0.05, rate=0.05, rho=-0.1, kappa=2.0, theta=0.04, xi=0.2,
    maturity=1.0, n_steps=256, strike=110.0, option_type="put",
)


def small_spec(**overrides):
    base = dict(s0=100.0, v0=0.1, rate=0.03, rho=-0.1, kappa=2.0, theta=0.12,
                xi=0.3, maturity=1.0, n_steps=2, strike=90.0, option_type="call")
    base.update(overrides)
    return HestonWeakEulerSpec(**base)


class InputValidationTest(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        for overrides in (
            {"rho": 1.5}, {"rho": -1.0001}, {"n_steps": 0}, {"maturity": 0.0},
            {"s0": 0.0}, {"v0": -0.01}, {"option_type": "digital"},
            {"strike": -1.0}, {"xi": -0.2},
        ):
            with self.assertRaises(ValueError):
                small_spec(**overrides)


class OneStepHandCalculationTest(unittest.TestCase):
    def test_all_four_shock_pairs(self):
        spec = small_spec(n_steps=1)
        h = spec.maturity
        sqrt_h = math.sqrt(h)
        rho_bar = math.sqrt(1.0 - spec.rho**2)
        for a, b in product((-1, 1), repeat=2):
            path = weak_euler_path_from_signs(spec, [(a, b)])
            # independent hand arithmetic, written out term by term
            sq = math.sqrt(spec.v0)
            r1 = (spec.rate - 0.5 * spec.v0) * h + sq * (spec.rho * a + rho_bar * b) * sqrt_h
            v1 = spec.v0 + spec.kappa * (spec.theta - spec.v0) * h + spec.xi * sq * a * sqrt_h
            self.assertAlmostEqual(path["log_returns"][1], r1, places=14)
            self.assertAlmostEqual(path["variances"][1], v1, places=14)
            s1 = spec.s0 * math.exp(r1)
            self.assertAlmostEqual(path["arithmetic_average"], s1, places=12)
            self.assertAlmostEqual(path["geometric_average"], s1, places=12)


class ExhaustiveEnumerationTest(unittest.TestCase):
    def test_two_step_sixteen_paths(self):
        spec = small_spec(n_steps=2)
        ref = enumerate_weak_euler(spec)
        self.assertEqual(ref["path_count"], 16)
        self.assertAlmostEqual(ref["probability_mass"], 1.0, places=14)
        self.assertGreaterEqual(ref["minimum_average_gap"], 0.0)
        self.assertGreaterEqual(ref["minimum_residual"], 0.0)
        identity = (ref["raw_payoff_expectation"]
                    - ref["geometric_payoff_expectation"]
                    - ref["residual_expectation"])
        self.assertAlmostEqual(identity, 0.0, places=12)

    def test_put_residual_nonnegative_on_every_path(self):
        spec = small_spec(n_steps=2, strike=110.0, option_type="put")
        ref = enumerate_weak_euler(spec)
        self.assertGreaterEqual(ref["minimum_residual"], 0.0)
        identity = (ref["geometric_payoff_expectation"]
                    - ref["raw_payoff_expectation"]
                    - ref["residual_expectation"])
        self.assertAlmostEqual(identity, 0.0, places=12)


class ScalarVectorAgreementTest(unittest.TestCase):
    def test_fixed_shock_matrix_agrees_to_1e12(self):
        spec = small_spec(n_steps=8)
        rng = np.random.default_rng(20260718)
        signs = 2 * rng.integers(0, 2, size=(64, 8, 2), dtype=np.int8) - 1
        # scalar reference per path
        for i in range(signs.shape[0]):
            path = weak_euler_path_from_signs(
                spec, [tuple(int(s) for s in signs[i, n]) for n in range(8)])
            # re-run the vectorized recurrence on this one path
            h, sqrt_h = spec.step, math.sqrt(spec.step)
            rho_bar = math.sqrt(1.0 - spec.rho**2)
            r = np.zeros(1)
            v = np.full(1, spec.v0)
            sum_exp_r = np.zeros(1)
            sum_r = np.zeros(1)
            for n in range(8):
                sq = np.sqrt(v)
                a, b = signs[i, n, 0], signs[i, n, 1]
                r = r + (spec.rate - 0.5 * v) * h + sq * (spec.rho * a + rho_bar * b) * sqrt_h
                v = v + spec.kappa * (spec.theta - v) * h + spec.xi * sq * a * sqrt_h
                sum_exp_r += np.exp(r)
                sum_r += r
            arithmetic = float(spec.s0 * sum_exp_r[0] / 8)
            geometric = float(spec.s0 * math.exp(sum_r[0] / 8))
            self.assertAlmostEqual(arithmetic, path["arithmetic_average"], delta=1e-12)
            self.assertAlmostEqual(geometric, path["geometric_average"], delta=1e-12)


class PathwiseInequalityTest(unittest.TestCase):
    def test_amgm_and_parity_on_enumeration(self):
        for option_type, strike in (("call", 90.0), ("put", 110.0)):
            spec = small_spec(n_steps=3, strike=strike, option_type=option_type)
            ref = enumerate_weak_euler(spec)
            for row in ref["paths"]:
                a, g = row["arithmetic_average"], row["geometric_average"]
                self.assertGreaterEqual(a - g, -1e-12)
                self.assertGreaterEqual(row["residual"], -1e-12)
                # call-put parity per average: (z-K)+ - (K-z)+ == z-K
                for z in (a, g):
                    self.assertAlmostEqual(
                        max(z - strike, 0.0) - max(strike - z, 0.0),
                        z - strike, places=10)


class ConstantVarianceReductionTest(unittest.TestCase):
    def test_xi_zero_matches_independent_recurrence(self):
        spec = small_spec(n_steps=16, xi=0.0, v0=0.12, theta=0.12)
        rng = np.random.default_rng(7)
        signs = 2 * rng.integers(0, 2, size=(16, 2), dtype=np.int8) - 1
        path = weak_euler_path_from_signs(
            spec, [tuple(int(s) for s in signs[n]) for n in range(16)])
        # independent recurrence: constant v, R is a deterministic drift plus
        # scaled correlated binary walk
        h = spec.step
        v = 0.12
        vol = math.sqrt(v * h)
        rho_bar = math.sqrt(1.0 - spec.rho**2)
        r = 0.0
        for n in range(16):
            r += (spec.rate - 0.5 * v) * h + vol * (spec.rho * signs[n, 0] + rho_bar * signs[n, 1])
            self.assertAlmostEqual(path["log_returns"][n + 1], r, places=12)
            self.assertAlmostEqual(path["variances"][n + 1], v, places=14)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_reproduces(self):
        spec = small_spec(n_steps=32)
        a = simulate_weak_euler_payoffs(spec, paths=5000, seed=99, keep_samples=False)
        b = simulate_weak_euler_payoffs(spec, paths=5000, seed=99, keep_samples=False)
        for key in ("arithmetic_price_discounted", "residual_mean_discounted",
                    "payoff_variance", "minimum_average_gap"):
            self.assertEqual(a[key], b[key])


class ProductionConfigurationTest(unittest.TestCase):
    def test_certificates_and_no_negative_variance(self):
        for spec in (CALL_1, PUT_2):
            certificate = variance_positivity_certificate(spec)
            self.assertTrue(certificate["certified"], certificate)
            out = simulate_weak_euler_payoffs(spec, paths=20000, seed=2026071801,
                                              keep_samples=False)
            self.assertEqual(out["negative_variance_count"], 0)
            self.assertEqual(out["nonfinite_count"], 0)
            self.assertGreaterEqual(out["minimum_average_gap"], 0.0)
            self.assertGreaterEqual(out["minimum_residual"], 0.0)
            self.assertGreater(out["minimum_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
