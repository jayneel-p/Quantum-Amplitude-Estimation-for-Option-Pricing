"""Arithmetic Asian (discrete fixings t_1..t_N; excludes S_0). Vanilla MC, geometric exact/MC, Kemna–Vorst CV."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from qc_option_pricing.classical.gbm import gbm_path


def _fixings(paths: np.ndarray) -> np.ndarray:
    """Return fixing-date prices S(t_1)...S(t_N), i.e. paths[:, 1:]."""
    return paths[:, 1:]


def _arith_avg(paths: np.ndarray) -> np.ndarray:
    """Arithmetic mean over fixing dates (excludes S_0)."""
    return _fixings(paths).mean(axis=1)


def _geo_avg(paths: np.ndarray) -> np.ndarray:
    """Geometric mean over fixing dates (excludes S_0)."""
    return np.exp(np.log(_fixings(paths)).mean(axis=1))


def arithmetic_asian_vanilla_from_paths(
    paths: np.ndarray, k: float, r: float, t: float
) -> tuple[float, float]:
    """
    Vanilla MC (estimate, stderr) from an existing path panel, shape (M, n_steps+1).
    """
    n_paths = paths.shape[0]
    discount = math.exp(-r * t)
    payoff = np.maximum(_arith_avg(paths) - k, 0.0) * discount
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(n_paths))


def arithmetic_asian_cv_from_paths(
    paths: np.ndarray,
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
) -> tuple[float, float]:
    """
    Kemna–Vorst CV (estimate, stderr) from the same path panel as vanilla.
    """
    n_paths = paths.shape[0]
    discount = math.exp(-r * t)
    arith_payoff = np.maximum(_arith_avg(paths) - k, 0.0) * discount
    geo_payoff = np.maximum(_geo_avg(paths) - k, 0.0) * discount
    geo_exact = geometric_asian_call_exact(s0, k, r, sigma, t, n_steps)
    cov_matrix = np.cov(arith_payoff, geo_payoff)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0
    adjusted = arith_payoff + beta * (geo_exact - geo_payoff)
    return float(adjusted.mean()), float(adjusted.std(ddof=1) / math.sqrt(n_paths))


def arithmetic_asian_vanilla_cv_shared(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    One ``gbm_path`` draw; return ``(vanilla_est, vanilla_se), (cv_est, cv_se)``.

    Use this when comparing vanilla vs KV on identical ω (common random numbers).
    """
    rng = rng or np.random.default_rng()
    paths = gbm_path(s0, r, sigma, t, n_steps, n_paths, rng=rng)
    v = arithmetic_asian_vanilla_from_paths(paths, k, r, t)
    cv = arithmetic_asian_cv_from_paths(paths, s0, k, r, sigma, t, n_steps)
    return v, cv


def asian_kv_payoff_correlation(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Pearson correlation between per-path discounted arithmetic and geometric
    Asian call payoffs (same fixing convention as KV). Used to interpret VR factors.
    """
    rng = rng or np.random.default_rng()
    paths = gbm_path(s0, r, sigma, t, n_steps, n_paths, rng=rng)
    discount = math.exp(-r * t)
    arith_pay = np.maximum(_arith_avg(paths) - k, 0.0) * discount
    geo_pay = np.maximum(_geo_avg(paths) - k, 0.0) * discount
    if float(np.std(arith_pay)) < 1e-14 or float(np.std(geo_pay)) < 1e-14:
        return float("nan")
    return float(np.corrcoef(arith_pay, geo_pay)[0, 1])


def arithmetic_asian_call_mc(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Payoff: max(A - K, 0) with A = (1/N) sum_{i=1}^{N} S(t_i).
    Returns (discounted_price_estimate, stderr).

    Simulates paths once (vanilla only). For vanilla+CV on the same paths, use
    ``arithmetic_asian_vanilla_cv_shared``.
    """
    rng = rng or np.random.default_rng()
    paths = gbm_path(s0, r, sigma, t, n_steps, n_paths, rng=rng)
    return arithmetic_asian_vanilla_from_paths(paths, k, r, t)


def geometric_asian_call_exact(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
) -> float:
    """
    Closed-form price for a discrete-monitoring geometric-average Asian call.

    The geometric mean of lognormals is lognormal, so BS-style formula applies
    with adjusted drift and volatility.

    Monitoring dates: t_1, t_2, ..., t_N equally spaced in (0, T],
    with N = n_steps.  S(t_0) is excluded.

    Adjusted parameters (discrete case):
        sigma_G = sigma * sqrt( (N+1)(2N+1) / (6 N^2) )
        mu_G    = (r - 0.5*sigma^2) * (N+1)/(2N) + 0.5*sigma_G^2
    Then price = exp(-rT) * [S0 exp(mu_G T) N(d1) - K N(d2)]
    with d1,d2 as in BS with sigma_G and mu_G.
    """
    n = n_steps

    sigma_g = sigma * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))
    mu_g = (r - 0.5 * sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_g ** 2

    d1 = (math.log(s0 / k) + (mu_g + 0.5 * sigma_g ** 2) * t) / (sigma_g * math.sqrt(t))
    d2 = d1 - sigma_g * math.sqrt(t)

    price = math.exp(-r * t) * (
        s0 * math.exp(mu_g * t) * norm.cdf(d1) - k * norm.cdf(d2)
    )
    return price


def geometric_asian_call_mc(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    MC price for geometric-average Asian call (same fixings as closed form).
    Useful for verifying geometric_asian_call_exact.
    Returns (discounted_price_estimate, stderr).
    """
    rng = rng or np.random.default_rng()
    paths = gbm_path(s0, r, sigma, t, n_steps, n_paths, rng=rng)
    avg = _geo_avg(paths)
    payoff = np.maximum(avg - k, 0.0)
    disc = np.exp(-r * t) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_paths))


def arithmetic_asian_call_cv(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Kemna–Vorst control-variate estimator for arithmetic Asian call.

    Simulates paths once (CV only). For vanilla+CV on the same paths, use
    ``arithmetic_asian_vanilla_cv_shared``.

    Returns (discounted_price_estimate, stderr).
    """
    rng = rng or np.random.default_rng()
    paths = gbm_path(s0, r, sigma, t, n_steps, n_paths, rng=rng)
    return arithmetic_asian_cv_from_paths(paths, s0, k, r, sigma, t, n_steps)
