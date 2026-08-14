"""Deterministic geometric-Asian control for Wang--Kan weak Euler.

This module prices the geometric control under *the same binary weak-Euler
probability model* used by :mod:`heston_weak_euler`.  It therefore restores an
exact control-variate identity at the model level; it is deliberately distinct
from the continuous-Heston geometric formula in ``heston_geometric_asian``.

Let ``q = s/N`` and define, for ``n = 0,...,N``,

    F_n(r, v) = E[exp(q sum_{j=n+1}^N R_j) | R_n=r, v_n=v].

Because the log-return update is additive, ``F_n`` factorizes as

    F_n(r, v) = exp(c_n r) f_n(v),       c_n = (N-n)s/N.

Writing ``h=T/N``, ``d(v)=(r-v/2)h``, ``w(v)=sqrt(vh)``, and

    V_a(v) = v + kappa(theta-v)h + xi a sqrt(vh),  a in {-1,+1},

the independent binary ``b`` shock can be summed analytically.  The remaining
one-dimensional backward recursion is

    f_n(v) = 1/2 cosh(c_n sqrt(1-rho^2) w(v))
             sum_{a=+-1} exp(c_n[d(v)+rho a w(v)]) f_{n+1}(V_a(v)),

with ``f_N=1``.  Hence ``E[exp(s Ybar)] = f_0(v0)`` for
``Ybar=N^{-1} sum_j R_j``.  The apparent ``4^N`` path problem has become a
one-dimensional deterministic functional recursion.

The production implementation represents ``f_n`` by cubic interpolation on a
uniform grid in ``sqrt(v)``.  Prices must be accompanied by grid, Fourier
cutoff, and quadrature refinement; no interpolation error is silently called
an analytic bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

from .heston_weak_euler import HestonWeakEulerSpec


@dataclass(frozen=True)
class TransformDiagnostics:
    variance_nodes: int
    interval_count: int
    maximum_interval_width: float
    clipped_evaluations: int
    maximum_clip_distance: float


def _variance_branch(v: float | np.ndarray, sign: int,
                     spec: HestonWeakEulerSpec) -> float | np.ndarray:
    h = spec.step
    return ((1.0 - spec.kappa * h) * v + spec.kappa * spec.theta * h
            + sign * spec.xi * np.sqrt(np.maximum(v, 0.0) * h))


def _branch_extrema(lo: float, hi: float, sign: int,
                    spec: HestonWeakEulerSpec) -> tuple[float, float]:
    """Exact extrema of one variance branch over a closed interval."""
    h = spec.step
    alpha = 1.0 - spec.kappa * h
    gamma = spec.xi * math.sqrt(h)
    candidates = [lo, hi]
    if lo <= 0.0 <= hi:
        candidates.append(0.0)
    if alpha != 0.0 and gamma != 0.0:
        root = -sign * gamma / (2.0 * alpha)
        if root >= 0.0:
            critical = root * root
            if lo <= critical <= hi:
                candidates.append(critical)
    values = [float(_variance_branch(v, sign, spec)) for v in candidates]
    return min(values), max(values)


def reachable_variance_intervals(
    spec: HestonWeakEulerSpec,
) -> list[tuple[float, float]]:
    """Certified interval enclosure of reachable variance at every date."""
    intervals = [(float(spec.v0), float(spec.v0))]
    lo = hi = float(spec.v0)
    for _ in range(spec.n_steps):
        down = _branch_extrema(lo, hi, -1, spec)
        up = _branch_extrema(lo, hi, 1, spec)
        next_lo = min(down[0], up[0])
        next_hi = max(down[1], up[1])
        tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(next_hi))
        if next_lo < -tolerance:
            raise ValueError(
                "weak-Euler variance is not nonnegative on its reachable enclosure"
            )
        lo = max(next_lo, 0.0)
        hi = max(next_hi, lo)
        intervals.append((lo, hi))
    return intervals


def _sqrt_grid(interval: tuple[float, float], nodes: int) -> np.ndarray:
    lo, hi = interval
    ulo, uhi = math.sqrt(max(lo, 0.0)), math.sqrt(max(hi, 0.0))
    if uhi == ulo:
        return np.array([ulo])
    return np.linspace(ulo, uhi, nodes)


def weak_euler_geometric_transform_many(
    s_values: Iterable[complex] | np.ndarray,
    spec: HestonWeakEulerSpec,
    *,
    variance_nodes: int = 512,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, TransformDiagnostics]:
    """Return ``E[exp(s Ybar)]`` for every complex ``s`` supplied.

    ``variance_nodes`` controls the deterministic interpolation error.  Use at
    least two node counts and report their price difference for production
    results.
    """
    if variance_nodes < 8:
        raise ValueError("variance_nodes must be at least 8")
    s = np.asarray(list(s_values), dtype=np.complex128)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("s_values must be a nonempty one-dimensional sequence")
    if not np.isfinite(s.real).all() or not np.isfinite(s.imag).all():
        raise ValueError("s_values must be finite")

    intervals = reachable_variance_intervals(spec)
    n_steps = spec.n_steps
    h = spec.step
    sqrt_h = math.sqrt(h)
    rho_bar = math.sqrt(max(1.0 - spec.rho * spec.rho, 0.0))

    next_grid = _sqrt_grid(intervals[-1], variance_nodes)
    next_values = np.ones((s.size, next_grid.size), dtype=np.complex128)
    clipped = 0
    maximum_clip = 0.0

    for n in range(n_steps - 1, -1, -1):
        current_grid = _sqrt_grid(intervals[n], variance_nodes)
        v = current_grid * current_grid
        shock_scale = current_grid * sqrt_h
        drift = (spec.rate - 0.5 * v) * h
        coefficient = ((n_steps - n) / n_steps) * s
        c = coefficient[:, None]

        if next_grid.size == 1:
            def evaluate_next(points: np.ndarray) -> np.ndarray:
                return np.broadcast_to(next_values[:, :1], (s.size, points.size))
        else:
            spline = CubicSpline(next_grid, next_values, axis=1)

            def evaluate_next(points: np.ndarray) -> np.ndarray:
                nonlocal clipped, maximum_clip
                bounded = np.clip(points, next_grid[0], next_grid[-1])
                distances = np.abs(points - bounded)
                clipped += int(np.count_nonzero(distances))
                if distances.size:
                    maximum_clip = max(maximum_clip, float(distances.max()))
                return spline(bounded)

        common = np.cosh(c * (rho_bar * shock_scale[None, :]))
        accumulated = np.zeros((s.size, current_grid.size), dtype=np.complex128)
        for sign in (-1, 1):
            v_next = np.asarray(_variance_branch(v, sign, spec), dtype=float)
            tolerance = 512.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(v_next))
            if np.any(v_next < -tolerance):
                raise ValueError("negative variance encountered in transform recursion")
            u_next = np.sqrt(np.maximum(v_next, 0.0))
            continuation = evaluate_next(u_next)
            exponent = c * (drift[None, :] + spec.rho * sign
                            * shock_scale[None, :])
            accumulated += np.exp(exponent) * continuation
        next_values = 0.5 * common * accumulated
        next_grid = current_grid

    answer = next_values[:, 0]
    diagnostics = TransformDiagnostics(
        variance_nodes=variance_nodes,
        interval_count=len(intervals),
        maximum_interval_width=max(hi - lo for lo, hi in intervals),
        clipped_evaluations=clipped,
        maximum_clip_distance=maximum_clip,
    )
    if return_diagnostics:
        return answer, diagnostics
    return answer


def weak_euler_geometric_transform(
    s: complex,
    spec: HestonWeakEulerSpec,
    *,
    variance_nodes: int = 512,
) -> complex:
    """Scalar wrapper for :func:`weak_euler_geometric_transform_many`."""
    return complex(weak_euler_geometric_transform_many(
        [s], spec, variance_nodes=variance_nodes)[0])


def weak_euler_geometric_asian_price(
    spec: HestonWeakEulerSpec,
    *,
    variance_nodes: int = 512,
    quadrature_order: int = 384,
    u_max: float = 200.0,
    damping: float = 1.5,
) -> dict[str, float | int]:
    """Price the geometric Asian payoff under the binary weak-Euler model.

    Fourier inversion is applied directly to the exponentially damped call
    payoff.  This gives an absolutely integrable ``O(u^-2)`` kernel even
    though the finite binary model has a discrete distribution whose
    characteristic function need not decay.  Gauss--Legendre quadrature is
    used on ``[0,u_max]``.  The returned parameters are diagnostics, not an
    error certificate; production callers must perform explicit refinement.
    """
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least 16")
    if not math.isfinite(u_max) or u_max <= 0.0:
        raise ValueError("u_max must be positive and finite")
    if not math.isfinite(damping) or damping <= 1.0:
        raise ValueError("damping must be finite and exceed one")

    x, weights = leggauss(quadrature_order)
    u = 0.5 * u_max * (x + 1.0)
    weights = 0.5 * u_max * weights
    arguments = np.concatenate((np.array([1.0 + 0.0j]), damping - 1j * u))
    transforms, diagnostics = weak_euler_geometric_transform_many(
        arguments, spec, variance_nodes=variance_nodes, return_diagnostics=True)
    moment = transforms[0]
    damped_transform = transforms[1:]
    log_moneyness = math.log(spec.strike / spec.s0)
    payoff_transform = (
        np.exp(-(damping - 1.0 - 1j * u) * log_moneyness)
        / ((damping - 1.0 - 1j * u) * (damping - 1j * u))
    )
    expected_geometric = spec.s0 * float(moment.real)
    call = (spec.discount * spec.s0 / math.pi
            * float(np.dot(weights, np.real(payoff_transform
                                             * damped_transform))))
    put = call - spec.discount * (expected_geometric - spec.strike)
    price = call if spec.option_type == "call" else put
    return {
        "price": price,
        "call_price": call,
        "put_price": put,
        "expected_geometric_average": expected_geometric,
        "variance_nodes": variance_nodes,
        "quadrature_order": quadrature_order,
        "u_max": u_max,
        "damping": damping,
        "clipped_evaluations": diagnostics.clipped_evaluations,
        "maximum_clip_distance": diagnostics.maximum_clip_distance,
        "maximum_variance_interval_width": diagnostics.maximum_interval_width,
    }
