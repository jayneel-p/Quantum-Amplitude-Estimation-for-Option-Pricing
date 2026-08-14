"""Unit tests for the residual-cap interface of the finite-grid Asian oracle.

Covers the three clipping regimes (none, partial, saturating) for the
independent ``residual_payoff_cap``, and the bias-budget selector.

Run with:
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

from qc_option_pricing.quantum.asian_oracle import (
    AsianGridSpec,
    build_asian_model,
    build_asian_oracle,
    enumerate_encoded_asian,
    objective_probability_from_mps,
    residual_cap_from_bias_budget,
)

BINARY = dict(n_dates=2, shock_points=(-1.0, 1.0), shock_probabilities=(0.4, 0.6),
              s0=2.0, strike=1.0, rate=0.0, volatility=0.3, maturity=1.0,
              shock_scale=1, price_scale=1)


class ResidualCapInterfaceTest(unittest.TestCase):
    def test_default_residual_cap_equals_raw_cap(self) -> None:
        model = build_asian_model(AsianGridSpec(**BINARY))
        self.assertEqual(model.residual_cap_numerator, model.raw_cap_numerator)

    def test_nonpositive_residual_cap_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AsianGridSpec(**BINARY, residual_payoff_cap=0.0)
        with self.assertRaises(ValueError):
            AsianGridSpec(**BINARY, residual_payoff_cap=-1.0)

    def test_selector_meets_budget_minimally(self) -> None:
        budget = 0.05
        cap = residual_cap_from_bias_budget(AsianGridSpec(**BINARY), budget)
        one_unit = 1.0 / (BINARY["n_dates"] * BINARY["price_scale"])

        def exact_bias(cap_dollars: float) -> float:
            ref = enumerate_encoded_asian(
                AsianGridSpec(**BINARY, residual_payoff_cap=cap_dollars))
            return (ref.residual_payoff_undiscounted
                    - ref.clipped_residual_payoff_undiscounted)

        self.assertLessEqual(exact_bias(cap), budget)
        if cap > one_unit:
            self.assertGreater(exact_bias(cap - one_unit), budget)

    def _qcv_case(self, spec: AsianGridSpec) -> tuple[float, float, float, float]:
        ref = enumerate_encoded_asian(spec)
        oracle = build_asian_oracle(spec, "qcv")
        probability, leakage = objective_probability_from_mps(oracle)
        price = oracle.post_process(probability)
        disc = math.exp(-spec.rate * spec.maturity)
        target = disc * (ref.clipped_residual_payoff_undiscounted
                         + ref.geometric_payoff_undiscounted)
        self.assertAlmostEqual(probability, ref.qcv_objective_probability, delta=1e-9)
        self.assertLess(leakage, 1e-9)
        self.assertAlmostEqual(price, target, delta=oracle.cap_dollars * 1e-9 + 1e-12)
        return probability, leakage, price, ref.residual_payoff_undiscounted

    def test_qcv_no_clipping(self) -> None:
        spec = AsianGridSpec(**BINARY, residual_payoff_cap=50.0)
        ref = enumerate_encoded_asian(spec)
        self.assertAlmostEqual(ref.clipped_residual_payoff_undiscounted,
                               ref.residual_payoff_undiscounted, delta=1e-12)
        self._qcv_case(spec)

    def test_qcv_partial_clipping(self) -> None:
        # price_scale=2 admits a cap strictly inside the residual support, so
        # some paths clip while others stay below the cap: interior amplitude
        # with active clipping.
        spec = AsianGridSpec(**dict(BINARY, price_scale=2),
                             residual_payoff_cap=0.5)
        ref = enumerate_encoded_asian(spec)
        self.assertLess(ref.clipped_residual_payoff_undiscounted,
                        ref.residual_payoff_undiscounted)
        probability, _, _, _ = self._qcv_case(spec)
        self.assertGreater(probability, 1e-9)
        self.assertLess(probability, 1.0 - 1e-9)

    def test_qcv_saturating_cap(self) -> None:
        one_unit = 1.0 / (BINARY["n_dates"] * BINARY["price_scale"])
        spec = AsianGridSpec(**BINARY, residual_payoff_cap=one_unit)
        probability, _, _, _ = self._qcv_case(spec)
        self.assertGreater(probability, 1.0 - 1e-9)


if __name__ == "__main__":
    unittest.main()
