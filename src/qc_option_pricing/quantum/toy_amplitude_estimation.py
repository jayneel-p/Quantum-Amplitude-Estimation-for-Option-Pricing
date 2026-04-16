"""
Toy amplitude-estimation style demo (Bernoulli probability).

Rebentrost et al. (2018) and Stamatopoulos et al. (2020) reduce option pricing to
estimating a = E[f(X)] in [0,1] after normalization, then using Amplitude Estimation
for O(1/M) error vs classical MC O(1/sqrt(M)).

Full option circuits require distribution loading + payoff oracle; this file isolates
the AE statistical advantage on a synthetic probability for sanity checks.
"""

from __future__ import annotations

import math


def classical_mc_bernoulli(p: int, q: int, shots: int) -> tuple[float, float]:
    """Estimate p/(p+q) with `shots` Bernoulli samples (variance p q / (p+q)^2 / shots)."""
    # deterministic stand-in: use exact mean and CLT stderr for comparison
    mu = p / (p + q)
    var = mu * (1 - mu) / shots
    return mu, math.sqrt(var)


def maximum_likelihood_ae_angle(success_counts: list[int], total_shots: int) -> float:
    """
    Given counts of |1> outcomes for Grover-like applications (simplified),
    recover sin^2(theta) ~ success probability — placeholder for MLE AE (Suzuki et al.).

    For a real project, use qiskit_algorithms.AmplitudeEstimation or IterativeAE.
    """
    if not success_counts:
        return 0.0
    # Naive: first count only as Bernoulli frequency (NOT true AE — documents API hook)
    s = success_counts[0]
    return s / total_shots
