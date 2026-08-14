"""Independent checks for the exact-model Wang--Kan geometric control."""

from __future__ import annotations

import cmath
from itertools import product
import unittest

from qc_option_pricing.classical.heston_weak_euler import (
    HestonWeakEulerSpec,
    enumerate_weak_euler,
    weak_euler_path_from_signs,
)
from qc_option_pricing.classical.heston_weak_euler_geometric import (
    reachable_variance_intervals,
    weak_euler_geometric_asian_price,
    weak_euler_geometric_transform,
)


def make_spec(option_type="call", n_steps=4, xi=0.3, strike=None):
    return HestonWeakEulerSpec(
        s0=100.0, v0=0.1, rate=0.03, rho=-0.1, kappa=2.0,
        theta=0.12, xi=xi, maturity=1.0, n_steps=n_steps,
        strike=(90.0 if option_type == "call" else 110.0)
        if strike is None else strike,
        option_type=option_type,
    )


def enumerated_transform(spec, s):
    total = 0.0j
    count = 4 ** spec.n_steps
    for signs in product(product((-1, 1), repeat=2), repeat=spec.n_steps):
        path = weak_euler_path_from_signs(spec, signs)
        average_log_return = sum(path["log_returns"][1:]) / spec.n_steps
        total += cmath.exp(s * average_log_return) / count
    return total


class VarianceEnclosureTest(unittest.TestCase):
    def test_small_tree_is_inside_every_interval(self):
        spec = make_spec(n_steps=4)
        intervals = reachable_variance_intervals(spec)
        for signs in product(product((-1, 1), repeat=2), repeat=spec.n_steps):
            path = weak_euler_path_from_signs(spec, signs)
            for variance, (lo, hi) in zip(path["variances"], intervals):
                self.assertGreaterEqual(variance, lo - 1e-14)
                self.assertLessEqual(variance, hi + 1e-14)


class TransformEnumerationTest(unittest.TestCase):
    def test_recursion_matches_full_four_shock_tree(self):
        spec = make_spec(n_steps=4)
        for s in (0.0j, 1.0 + 0.0j, 1.0j, 1.0 + 2.0j, 10.0j):
            got = weak_euler_geometric_transform(s, spec, variance_nodes=64)
            want = enumerated_transform(spec, s)
            self.assertLess(abs(got - want), 2e-10, (s, got, want))

    def test_deterministic_variance_branch_matches_enumeration(self):
        spec = make_spec(n_steps=5, xi=0.0)
        for s in (1.0, 3.0j, 1.5 - 7.0j):
            got = weak_euler_geometric_transform(s, spec, variance_nodes=32)
            want = enumerated_transform(spec, s)
            self.assertLess(abs(got - want), 2e-12)


class PriceEnumerationTest(unittest.TestCase):
    def test_damped_fourier_price_matches_exact_payoff_enumeration(self):
        for option_type in ("call", "put"):
            spec = make_spec(option_type=option_type, n_steps=4)
            exact = enumerate_weak_euler(spec)
            want = spec.discount * exact["geometric_payoff_expectation"]
            got = weak_euler_geometric_asian_price(
                spec, variance_nodes=64, quadrature_order=4096, u_max=800.0,
            )["price"]
            # The finite binary distribution is atomic, so Fourier cutoff
            # convergence is algebraic rather than Heston-style exponential.
            self.assertLess(abs(got - want), 2e-4, (option_type, got, want))

    def test_same_model_control_identity_is_exact(self):
        for option_type in ("call", "put"):
            spec = make_spec(option_type=option_type, n_steps=3)
            exact = enumerate_weak_euler(spec)
            raw = spec.discount * exact["raw_payoff_expectation"]
            geometric = spec.discount * exact["geometric_payoff_expectation"]
            residual = spec.discount * exact["residual_expectation"]
            restored = (geometric + residual if option_type == "call"
                        else geometric - residual)
            self.assertAlmostEqual(restored, raw, places=13)


if __name__ == "__main__":
    unittest.main()
