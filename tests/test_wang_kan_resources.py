"""Reproduction tests for the Wang-Kan resource model.

Published values are rounded to two significant figures, so agreement is
asserted against the interval each rounded value implies.
"""

from __future__ import annotations

import unittest

from qc_option_pricing.quantum.wang_kan_resources import (
    calibrate_ppoly,
    delta_u2_control_variate,
    n_oracle_queries,
    t_add,
    t_add_const,
    t_arcsin_sqrt,
    t_mul_const,
    t_q,
    t_sqrt,
    t_toffoli,
    t_u1_weak,
    t_u2_asian,
    t_u3,
    t_usin,
)


def within_rounding(value: float, published: float) -> bool:
    """True when value rounds to the published two-significant-figure value."""
    import math
    exponent = math.floor(math.log10(abs(published)))
    step = 10.0 ** (exponent - 1)
    return abs(value - published) <= 0.5 * step + 1e-9


class PrimitiveTest(unittest.TestCase):
    def test_published_primitive_values(self):
        self.assertEqual(t_add(27), 104)
        self.assertEqual(t_add_const(27), 100)
        self.assertEqual(t_toffoli(3), 4)
        self.assertEqual(t_mul_const(27, 11), 2044)
        self.assertEqual(t_sqrt(27), 2008)


class U1ReproductionTest(unittest.TestCase):
    def test_instance_1_weak_euler(self):
        value = t_u1_weak(256, 27, 11)
        self.assertEqual(value, 6_445_056)
        self.assertTrue(within_rounding(value, 6.4e6))

    def test_instance_2_weak_euler(self):
        value = t_u1_weak(256, 27, 10)
        self.assertEqual(value, 6_373_376)
        self.assertTrue(within_rounding(value, 6.4e6))


class U2CalibrationTest(unittest.TestCase):
    def test_exp_cost_calibrates_within_published_rounding(self):
        # Instance 1: published U2 = 9.3e6 with N=256, n=27, p=11.
        fixed = ((256 - 1) * t_add(27) + t_mul_const(27, 11)
                 + t_add_const(27) + 27 * t_toffoli(3))
        implied_exp = (9.3e6 - fixed) / 256
        candidates = calibrate_ppoly(implied_exp, 27, 11)
        self.assertTrue(candidates, "no (M, d) reproduces the published U2")
        best = candidates[0]
        value = t_u2_asian(256, 27, 11, best["t_count"])
        self.assertTrue(within_rounding(value, 9.3e6), value)

    def test_u3_calibrates_within_published_rounding(self):
        # Instance 1: published U3 = 6.0e4 with eps_arcsin=1e-6, eps_sin=1e-8.
        usin = t_usin(27, 1e-8)
        target_arcsin = 6.0e4 - usin
        candidates = calibrate_ppoly(
            target_arcsin, 27, 11,
            builder=lambda n, p, m, d: t_arcsin_sqrt(n, p, m, d))
        self.assertTrue(candidates)
        value = t_u3(27, 11, candidates[0]["m_pieces"],
                     candidates[0]["degree"], 1e-8)
        self.assertTrue(within_rounding(value, 6.0e4), value)


class QueryBoundTest(unittest.TestCase):
    def test_published_7363_under_nearest_rounding(self):
        self.assertEqual(n_oracle_queries(1e-3, 0.1), 7363)

    def test_strict_ceiling_gives_7364(self):
        # documented discrepancy: a true upper bound must round up
        self.assertEqual(n_oracle_queries(1e-3, 0.1, rounding="ceil"), 7364)

    def test_monotone_in_epsilon(self):
        self.assertGreater(n_oracle_queries(1e-4, 0.1),
                           n_oracle_queries(1e-3, 0.1))


class TotalReproductionTest(unittest.TestCase):
    def test_instance_1_q_and_total(self):
        u1 = t_u1_weak(256, 27, 11)
        # calibrated EXP cost via published U2
        fixed = ((256 - 1) * t_add(27) + t_mul_const(27, 11)
                 + t_add_const(27) + 27 * t_toffoli(3))
        t_exp = calibrate_ppoly((9.3e6 - fixed) / 256, 27, 11)[0]["t_count"]
        u2 = t_u2_asian(256, 27, 11, t_exp)
        usin = t_usin(27, 1e-8)
        arc = calibrate_ppoly(6.0e4 - usin, 27, 11,
                              builder=lambda n, p, m, d: t_arcsin_sqrt(n, p, m, d))
        u3 = t_u3(27, 11, arc[0]["m_pieces"], arc[0]["degree"], 1e-8)
        t_a = u1 + u2 + u3
        q = t_q(t_a, 22_000)
        self.assertTrue(within_rounding(q, 3.2e7), q)
        # The published total 2.4e11 is the product of rounded inputs; the
        # implied interval from the published rounding is
        # 7363 x [3.15e7, 3.25e7].  Our point value sits inside it but
        # rounds to 2.3e11; the sub-rounding gap is recorded in the ledger.
        total = 7363 * q
        self.assertGreaterEqual(total, 7363 * 3.15e7)
        self.assertLessEqual(total, 7363 * 3.25e7)


class ControlVariateDeltaTest(unittest.TestCase):
    def test_delta_is_small_against_u2(self):
        fixed = ((256 - 1) * t_add(27) + t_mul_const(27, 11)
                 + t_add_const(27) + 27 * t_toffoli(3))
        t_exp = calibrate_ppoly((9.3e6 - fixed) / 256, 27, 11)[0]["t_count"]
        delta = delta_u2_control_variate(256, 27, 11, t_exp)
        u2 = t_u2_asian(256, 27, 11, t_exp)
        self.assertLess(delta / u2, 0.01)
        # the delta must charge exactly one extra exponential
        self.assertGreater(delta, t_exp)


if __name__ == "__main__":
    unittest.main()
