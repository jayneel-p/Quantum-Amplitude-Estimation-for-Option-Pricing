"""Geometric Brownian motion under risk-neutral measure (Rebentrost et al. setup)."""

from __future__ import annotations

import numpy as np


def terminal_prices(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample S_T at maturity T for GBM: dS = r S dt + sigma S dW (Q-measure).

    S_T = S0 * exp((r - 0.5 sigma^2) T + sigma sqrt(T) Z), Z ~ N(0,1).
    """
    rng = rng or np.random.default_rng()
    z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * t
    diffusion = sigma * np.sqrt(t) * z
    return s0 * np.exp(drift + diffusion)


def gbm_path(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulated paths S(t_k), shape (n_paths, n_steps + 1), equally spaced in [0, T]."""
    rng = rng or np.random.default_rng()
    dt = t / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    inc = drift + sigma * np.sqrt(dt) * shocks
    log_s = np.zeros((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(s0)
    log_s[:, 1:] = np.log(s0) + np.cumsum(inc, axis=1)
    return np.exp(log_s)
