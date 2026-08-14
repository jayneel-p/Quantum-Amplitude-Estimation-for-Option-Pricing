"""Discretely monitored geometric Asian options under the exact Heston model.

Derivation, independent of the inaccessible Kim, Kim, Kim, Wee (2016)
recursive method, but computing the same object.  The Heston model is
affine, so the joint conditional transform of the log return X_t =
ln(S_t/S_0) and variance v_t over one monitoring interval h is
exponentially affine,

    E[ exp(z X_{t+h} + B' v_{t+h}) | X_t, v_t ]
        = exp(z X_t + A_step(z, B'; h) + B_step(z, B'; h) v_t),

where B solves the Riccati equation B' = alpha + beta B + gamma B^2 with
alpha = (z^2 - z)/2, beta = rho xi z - kappa, gamma = xi^2/2, and
A' = r z + kappa theta B.  The substitution B = -f'/(gamma f) linearizes
the Riccati equation to f'' - beta f' + alpha gamma f = 0, giving the
closed form used here; the integral of B contributes
-(2 kappa theta / xi^2) ln(f(h)/f(0)) to A.

The transform of the geometric-average exponent Ybar = (1/N) sum_j X_{t_j}
follows by folding the fixings backward: starting from (z, A, B) = (0, 0, 0)
at t_N, add w s to z at each fixing (w = 1/N) and propagate one interval.
At t_0 the transform is E[e^{s Ybar}] = exp(A + B v_0) because X_0 = 0.
Working interval by interval keeps every complex logarithm near the
principal branch, which is the same numerical motivation as the published
recursive method.

Prices follow by Gil-Pelaez inversion on G = S_0 e^{Ybar}:

    call = e^{-rT} [ S_0 phi(-i) Pi_1 - K Pi_2 ],

with phi(u) = E[e^{i u Ybar}], k = ln(K/S_0),
Pi_2 = 1/2 + (1/pi) int_0^inf Re[e^{-iuk} phi(u) / (iu)] du and
Pi_1 the same integral with phi(u - i)/phi(-i).  The put follows from
put-call parity on G.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from scipy.integrate import quad


@dataclass(frozen=True)
class HestonParams:
    s0: float
    v0: float
    rate: float
    rho: float
    kappa: float
    theta: float
    xi: float

    def __post_init__(self) -> None:
        if self.s0 <= 0.0 or self.v0 < 0.0 or self.xi < 0.0 or self.kappa < 0.0:
            raise ValueError("invalid Heston parameters")
        if abs(self.rho) > 1.0:
            raise ValueError("|rho| must not exceed 1")


def _cexpm1(x: complex) -> complex:
    """Complex expm1; series for small arguments (cmath has no expm1)."""
    if abs(x) < 1e-4:
        return x * (1.0 + x / 2.0 * (1.0 + x / 3.0 * (1.0 + x / 4.0)))
    return cmath.exp(x) - 1.0


def _clog1p(w: complex) -> complex:
    """Complex log1p; series for small arguments (cmath has no log1p)."""
    if abs(w) < 1e-4:
        return w * (1.0 - w / 2.0 + w * w / 3.0 - w * w * w / 4.0)
    return cmath.log(1.0 + w)


def step_coefficients(z: complex, b_accumulated: complex, h: float,
                      params: HestonParams) -> tuple[complex, complex]:
    """One-interval affine propagation: returns (A_step, B_propagated).

    Let tau be time remaining to the interval's later endpoint.  The
    accumulated coefficient from later intervals is the initial condition
    B(0) = b_accumulated; the Riccati equation
    B' = alpha + beta B + gamma B^2 is solved forward in tau, and B(h) is
    the coefficient one monitoring date earlier.  A_step collects
    r z h + kappa theta int_0^h B dtau.
    """
    alpha = 0.5 * (z * z - z)
    beta = params.rho * params.xi * z - params.kappa
    gamma = 0.5 * params.xi * params.xi
    if gamma == 0.0:
        # deterministic variance: B' = alpha + beta B, linear case
        if beta == 0.0:
            b_h = b_accumulated + alpha * h
            integral = b_accumulated * h + 0.5 * alpha * h * h
        else:
            eb = cmath.exp(beta * h)
            b_h = (b_accumulated + alpha / beta) * eb - alpha / beta
            integral = ((b_accumulated + alpha / beta) * (eb - 1.0) / beta
                        - alpha * h / beta)
        a_step = params.rate * z * h + params.kappa * params.theta * integral
        return a_step, b_h
    # substitution B = -f'/(gamma f):  f'' - beta f' + alpha gamma f = 0,
    # roots lambda^2 - beta lambda + alpha gamma = 0.  The smaller root is
    # computed from the product of roots (= alpha gamma) to avoid the
    # catastrophic cancellation of (beta + d)/2 when |alpha gamma| << beta^2
    # (the Black-Scholes limit xi -> 0), which otherwise corrupts the
    # transform at the 1e-4 level.
    d = cmath.sqrt(beta * beta - 4.0 * alpha * gamma)
    if abs(beta - d) >= abs(beta + d):
        lam_big = 0.5 * (beta - d)
    else:
        lam_big = 0.5 * (beta + d)
    lam_small = (alpha * gamma / lam_big) if lam_big != 0 else 0.5 * (beta + d)
    # initial condition at tau = 0 fixes c1 (with c2 = 1):
    # B(0) = -(c1 lam_big + lam_small) / (gamma (c1 + 1)) = b_accumulated
    denominator = lam_big + gamma * b_accumulated
    if denominator == 0:
        raise ZeroDivisionError("degenerate Riccati initial condition")
    c1 = -(lam_small + gamma * b_accumulated) / denominator
    eb = cmath.exp(lam_big * h)
    es = cmath.exp(lam_small * h)
    f_h = c1 * eb + es
    f_0 = c1 + 1.0
    b_h = -(c1 * lam_big * eb + lam_small * es) / (gamma * f_h)
    # log(f_h / f_0) is O(gamma) when gamma is tiny but is multiplied by
    # 2 kappa theta / xi^2 = O(1/gamma); form the difference with expm1 and
    # take log1p so the small quantity never passes through a 1 + x
    # representation.
    delta = (c1 * _cexpm1(lam_big * h) + _cexpm1(lam_small * h)) / f_0
    a_step = (params.rate * z * h
              - (2.0 * params.kappa * params.theta / (params.xi * params.xi))
              * _clog1p(delta))
    return a_step, b_h


def geometric_average_transform(s: complex, params: HestonParams,
                                n_fixings: int, maturity: float) -> complex:
    """E[exp(s Ybar)] with Ybar the average of X at the N fixing dates."""
    h = maturity / n_fixings
    w = s / n_fixings
    z = 0.0 + 0.0j
    a_total = 0.0 + 0.0j
    b = 0.0 + 0.0j
    for _ in range(n_fixings):
        z = z + w
        a_step, b = step_coefficients(z, b, h, params)
        a_total = a_total + a_step
    return cmath.exp(a_total + b * params.v0)


def geometric_asian_price(params: HestonParams, strike: float,
                          maturity: float, n_fixings: int,
                          option_type: str = "call",
                          u_max: float = 200.0) -> dict[str, float]:
    """Discounted price of the discretely monitored geometric Asian option."""
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    log_moneyness = math.log(strike / params.s0)
    phi_minus_i = geometric_average_transform(1.0 + 0.0j, params,
                                              n_fixings, maturity)
    forward_g = params.s0 * phi_minus_i.real

    def integrand_pi2(u: float) -> float:
        phi = geometric_average_transform(1j * u, params, n_fixings, maturity)
        return (cmath.exp(-1j * u * log_moneyness) * phi / (1j * u)).real

    def integrand_pi1(u: float) -> float:
        phi = geometric_average_transform(1.0 + 1j * u, params,
                                          n_fixings, maturity)
        return (cmath.exp(-1j * u * log_moneyness) * phi
                / (1j * u * phi_minus_i)).real

    pi2_integral, pi2_err = quad(integrand_pi2, 1e-10, u_max, limit=400)
    pi1_integral, pi1_err = quad(integrand_pi1, 1e-10, u_max, limit=400)
    pi2 = 0.5 + pi2_integral / math.pi
    pi1 = 0.5 + pi1_integral / math.pi
    discount = math.exp(-params.rate * maturity)
    call = discount * (params.s0 * phi_minus_i.real * pi1 - strike * pi2)
    if option_type == "call":
        price = call
    else:
        price = call - discount * (forward_g - strike)
    return {
        "price": price,
        "call_price": call,
        "expected_geometric_average": forward_g,
        "pi1": pi1,
        "pi2": pi2,
        "quad_error_estimate": pi1_err + pi2_err,
    }
