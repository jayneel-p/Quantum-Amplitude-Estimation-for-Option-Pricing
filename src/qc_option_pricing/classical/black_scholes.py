"""Analytic European call under Black–Scholes (benchmark for Monte Carlo)."""

from __future__ import annotations

import math

from scipy.stats import norm


def european_call(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    if t <= 0:
        return max(s0 - k, 0.0)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s0 * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
