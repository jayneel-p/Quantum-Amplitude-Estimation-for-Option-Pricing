"""Validation of the affine-recursion geometric Asian pricer under Heston.

Four independent checks, none of which reuses the recursion under test:
closed-form Riccati step against direct ODE integration; the constant
variance (Black-Scholes) limit against an independently coded analytic
discrete geometric Asian formula; put-call parity on the geometric average;
and fine-substep Heston Monte Carlo at small fixing counts.
"""

from __future__ import annotations

import cmath
import math
import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm

from qc_option_pricing.classical.heston_geometric_asian import (
    HestonParams,
    geometric_asian_price,
    geometric_average_transform,
    step_coefficients,
)

CALL_1 = HestonParams(s0=100.0, v0=0.1, rate=0.03, rho=-0.1, kappa=2.0,
                      theta=0.12, xi=0.3)
PUT_2 = HestonParams(s0=100.0, v0=0.05, rate=0.05, rho=-0.1, kappa=2.0,
                     theta=0.04, xi=0.2)


def bs_discrete_geometric_asian_call(s0, strike, rate, sigma, maturity, n):
    """Independent analytic reference for constant volatility.

    Ybar is Gaussian with mean (r - sigma^2/2) T (n+1)/(2n) and variance
    sigma^2 T (n+1)(2n+1)/(6 n^2); the price is Black-Scholes on the
    lognormal geometric average.
    """
    mu = (rate - 0.5 * sigma * sigma) * maturity * (n + 1) / (2.0 * n)
    var = sigma * sigma * maturity * (n + 1) * (2 * n + 1) / (6.0 * n * n)
    sd = math.sqrt(var)
    d1 = (math.log(s0 / strike) + mu + var) / sd
    d2 = d1 - sd
    forward = s0 * math.exp(mu + 0.5 * var)
    return math.exp(-rate * maturity) * (forward * norm.cdf(d1)
                                         - strike * norm.cdf(d2))


class RiccatiStepTest(unittest.TestCase):
    def test_closed_form_matches_ode_integration(self):
        params = CALL_1
        h = 1.0 / 256
        rng = np.random.default_rng(20260718)
        for _ in range(6):
            z = complex(rng.uniform(-1, 2), rng.uniform(-3, 3))
            b_next = complex(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))
            a_step, b0 = step_coefficients(z, b_next, h, params)

            alpha = 0.5 * (z * z - z)
            beta = params.rho * params.xi * z - params.kappa
            gamma = 0.5 * params.xi * params.xi

            def odes(t, y):
                b = complex(y[0], y[1])
                a = alpha + beta * b + gamma * b * b
                da = params.rate * z + params.kappa * params.theta * b
                return [a.real, a.imag, da.real, da.imag]

            # integrate backward in time: tau runs from terminal to start
            sol = solve_ivp(odes, (0.0, h),
                            [b_next.real, b_next.imag, 0.0, 0.0],
                            rtol=1e-12, atol=1e-14)
            b_ode = complex(sol.y[0][-1], sol.y[1][-1])
            a_ode = complex(sol.y[2][-1], sol.y[3][-1])
            self.assertLess(abs(b_ode - b0), 1e-9)
            self.assertLess(abs(a_ode - a_step), 1e-9)


class BlackScholesLimitTest(unittest.TestCase):
    def test_tiny_vol_of_vol_matches_analytic(self):
        sigma = math.sqrt(0.09)
        for n, strike in ((16, 95.0), (256, 90.0)):
            params = HestonParams(s0=100.0, v0=0.09, rate=0.03, rho=0.0,
                                  kappa=2.0, theta=0.09, xi=1e-6)
            got = geometric_asian_price(params, strike, 1.0, n)["call_price"]
            want = bs_discrete_geometric_asian_call(100.0, strike, 0.03,
                                                    sigma, 1.0, n)
            self.assertLess(abs(got - want), 2e-4,
                            f"n={n}: got {got}, want {want}")


class ParityAndMomentTest(unittest.TestCase):
    def test_put_call_parity_and_transform_sanity(self):
        for params, strike in ((CALL_1, 90.0), (PUT_2, 110.0)):
            call = geometric_asian_price(params, strike, 1.0, 64, "call")
            put = geometric_asian_price(params, strike, 1.0, 64, "put")
            discount = math.exp(-params.rate)
            parity_gap = (call["call_price"] - put["price"]
                          - discount * (call["expected_geometric_average"]
                                        - strike))
            self.assertLess(abs(parity_gap), 1e-8)
            self.assertAlmostEqual(
                abs(geometric_average_transform(0.0j, params, 64, 1.0)), 1.0,
                places=12)


class MonteCarloCrossCheckTest(unittest.TestCase):
    def test_small_fixing_counts_against_fine_euler(self):
        rng = np.random.default_rng(20260719)
        for params, strike, n_fixings in ((CALL_1, 90.0, 4),
                                          (PUT_2, 110.0, 4)):
            paths, substeps = 400_000, 256
            dt = 1.0 / (n_fixings * substeps)
            sqrt_dt = math.sqrt(dt)
            log_s = np.zeros(paths)
            v = np.full(paths, params.v0)
            sum_log = np.zeros(paths)
            for j in range(n_fixings):
                for _ in range(substeps):
                    z1 = rng.standard_normal(paths)
                    z2 = (params.rho * z1 + math.sqrt(1 - params.rho**2)
                          * rng.standard_normal(paths))
                    v_pos = np.maximum(v, 0.0)
                    sq = np.sqrt(v_pos)
                    log_s += (params.rate - 0.5 * v_pos) * dt + sq * sqrt_dt * z2
                    v += (params.kappa * (params.theta - v_pos) * dt
                          + params.xi * sq * sqrt_dt * z1)
                sum_log += log_s
            g = params.s0 * np.exp(sum_log / n_fixings)
            discount = math.exp(-params.rate)
            payoff = np.maximum(g - strike, 0.0)
            mc = discount * float(payoff.mean())
            se = discount * float(payoff.std(ddof=1) / math.sqrt(paths))
            formula = geometric_asian_price(params, strike, 1.0,
                                            n_fixings)["call_price"]
            self.assertLess(abs(formula - mc), 4.0 * se + 0.02,
                            f"formula {formula} vs MC {mc} +/- {se}")


if __name__ == "__main__":
    unittest.main()
