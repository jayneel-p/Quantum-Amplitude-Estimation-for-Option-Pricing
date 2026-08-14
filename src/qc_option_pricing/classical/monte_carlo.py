"""Classical Monte Carlo option pricing (cf. Rebentrost Sec. III; Stamatopoulos Sec. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qc_option_pricing.classical.black_scholes import european_call
from qc_option_pricing.classical.gbm import terminal_prices


@dataclass
class McResult:
    estimate: float
    stderr: float
    discounted_mean_payoff: float


def european_call_mc(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> McResult:
    rng = rng or np.random.default_rng()
    st = terminal_prices(s0, r, sigma, t, n_paths, rng=rng)
    payoff = np.maximum(st - k, 0.0)
    disc = np.exp(-r * t) * payoff
    mean = float(disc.mean())
    stderr = float(disc.std(ddof=1) / np.sqrt(n_paths))
    return McResult(estimate=mean, stderr=stderr, discounted_mean_payoff=mean)


def convergence_curve(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    path_counts: list[int],
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[float], list[float], float]:
    """Returns (ns, estimates, stderrs, analytic_price)."""
    rng = rng or np.random.default_rng()
    analytic = european_call(s0, k, r, sigma, t)
    ns, ys, ses = [], [], []
    for n in path_counts:
        res = european_call_mc(s0, k, r, sigma, t, n, rng=rng)
        ns.append(n)
        ys.append(res.estimate)
        ses.append(res.stderr)
    return ns, ys, ses, analytic
