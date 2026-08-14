"""Paper-faithful classical weak-Euler Heston model of Wang and Kan.

Implements the simplified weak Euler scheme of Wang and Kan, *Option pricing
under stochastic volatility on a quantum computer*, Quantum 8, 1504 (2024),
Eqs. (38)-(41) and (51)-(52): log return R and variance v evolve with two
independent binary shocks a_n, b_n in {-1,+1} per step,

    R_{n+1} = R_n + (r - v_n/2) h + sqrt(v_n) [rho a_n + sqrt(1-rho^2) b_n] sqrt(h)
    v_{n+1} = v_n + kappa (theta - v_n) h + xi sqrt(v_n) a_n sqrt(h),

with R_0 = 0, v_0 given, h = T/N.  Fixings are S_j = S0 exp(R_j) for
j = 1..N (S_0 excluded), matching the N-exponential, (N-1)-addition payoff
cost structure of their Eq. (120).  The arithmetic and geometric averages are

    A = (1/N) sum_j S_j,      G = S0 exp((1/N) sum_j R_j),

so A >= G pathwise by AM-GM.  Control-variate identities (call and put) are

    D_call = (A-K)+ - (G-K)+  >= 0
    D_put  = (K-G)+ - (K-A)+  >= 0.

`weak_euler_path_from_signs` is the scalar semantic reference; the vectorized
simulator is written independently and never calls it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class HestonWeakEulerSpec:
    s0: float
    v0: float
    rate: float
    rho: float
    kappa: float
    theta: float
    xi: float
    maturity: float
    n_steps: int
    strike: float
    option_type: Literal["call", "put"]

    def __post_init__(self) -> None:
        scalars = (self.s0, self.v0, self.rate, self.rho, self.kappa,
                   self.theta, self.xi, self.maturity, self.strike)
        if not all(math.isfinite(float(x)) for x in scalars):
            raise ValueError("all model parameters must be finite")
        if self.s0 <= 0.0 or self.maturity <= 0.0:
            raise ValueError("s0 and maturity must be positive")
        if self.v0 < 0.0 or self.theta < 0.0 or self.xi < 0.0 or self.kappa < 0.0:
            raise ValueError("v0, theta, xi, kappa must be nonnegative")
        if abs(self.rho) > 1.0:
            raise ValueError("|rho| must not exceed 1")
        if self.strike < 0.0:
            raise ValueError("strike must be nonnegative")
        if not isinstance(self.n_steps, int) or isinstance(self.n_steps, bool) or self.n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    @property
    def step(self) -> float:
        return self.maturity / self.n_steps

    @property
    def discount(self) -> float:
        return math.exp(-self.rate * self.maturity)


def variance_positivity_certificate(spec: HestonWeakEulerSpec) -> dict[str, object]:
    """Closed-form one-step certificate that v_{n+1} >= 0 for every v_n >= 0.

    Writing u = sqrt(v), the worst (down-shock) branch is
    f(u) = (1 - kappa h) u^2 - xi sqrt(h) u + kappa theta h, whose minimum
    over u >= 0 is kappa theta h - xi^2 h / (4 (1 - kappa h)) when
    kappa h < 1.  Nonnegative iff 4 kappa theta (1 - kappa h) >= xi^2.
    This covers every reachable v_n because v_0 >= 0 and the certificate is
    uniform in v_n >= 0.
    """
    h = spec.step
    kh = spec.kappa * h
    if kh >= 1.0:
        return {"certified": False, "reason": f"kappa*h = {kh} >= 1; certificate inapplicable"}
    lhs = 4.0 * spec.kappa * spec.theta * (1.0 - kh)
    rhs = spec.xi * spec.xi
    worst = spec.kappa * spec.theta * h - spec.xi * spec.xi * h / (4.0 * (1.0 - kh))
    return {
        "certified": bool(lhs >= rhs),
        "condition": "4*kappa*theta*(1-kappa*h) >= xi^2",
        "lhs": lhs,
        "rhs": rhs,
        "worst_case_next_variance": worst,
    }


def _payoffs(a: float, g: float, spec: HestonWeakEulerSpec) -> tuple[float, float, float]:
    """Return (raw payoff X, geometric payoff Y, residual D) undiscounted."""
    if spec.option_type == "call":
        x = max(a - spec.strike, 0.0)
        y = max(g - spec.strike, 0.0)
        return x, y, x - y
    x = max(spec.strike - a, 0.0)
    y = max(spec.strike - g, 0.0)
    return x, y, y - x


def weak_euler_path_from_signs(
    spec: HestonWeakEulerSpec,
    shock_signs: Sequence[tuple[int, int]],
) -> dict[str, object]:
    """Scalar semantic reference for one path given (a_n, b_n) signs."""
    if len(shock_signs) != spec.n_steps:
        raise ValueError("need exactly one (a, b) sign pair per step")
    h = spec.step
    sqrt_h = math.sqrt(h)
    rho_bar = math.sqrt(1.0 - spec.rho * spec.rho)
    r_path = [0.0]
    v_path = [spec.v0]
    r, v = 0.0, spec.v0
    min_v = v
    for a, b in shock_signs:
        if a not in (-1, 1) or b not in (-1, 1):
            raise ValueError("shock signs must be -1 or +1")
        if v < 0.0:
            raise ValueError(f"negative variance {v} before square root")
        sq = math.sqrt(v)
        r = r + (spec.rate - 0.5 * v) * h + sq * (spec.rho * a + rho_bar * b) * sqrt_h
        v = v + spec.kappa * (spec.theta - v) * h + spec.xi * sq * a * sqrt_h
        r_path.append(r)
        v_path.append(v)
        min_v = min(min_v, v)
    fixings = [spec.s0 * math.exp(x) for x in r_path[1:]]
    arithmetic = sum(fixings) / spec.n_steps
    geometric = spec.s0 * math.exp(sum(r_path[1:]) / spec.n_steps)
    x, y, d = _payoffs(arithmetic, geometric, spec)
    return {
        "log_returns": r_path,
        "variances": v_path,
        "fixings": fixings,
        "arithmetic_average": arithmetic,
        "geometric_average": geometric,
        "raw_payoff": x,
        "geometric_payoff": y,
        "residual": d,
        "minimum_variance": min_v,
    }


def enumerate_weak_euler(spec: HestonWeakEulerSpec, *, max_paths: int = 1 << 20) -> dict[str, object]:
    """Exhaustive enumeration of all 4**N sign paths (small N only)."""
    count = 4 ** spec.n_steps
    if count > max_paths:
        raise ValueError(f"{count:,} paths exceed max_paths={max_paths:,}")
    probability = 1.0 / count
    mass = 0.0
    e_x = e_y = e_d = 0.0
    min_gap = math.inf
    min_res = math.inf
    min_v = math.inf
    rows = []
    for signs in product(product((-1, 1), repeat=2), repeat=spec.n_steps):
        path = weak_euler_path_from_signs(spec, signs)
        mass += probability
        e_x += probability * path["raw_payoff"]
        e_y += probability * path["geometric_payoff"]
        e_d += probability * path["residual"]
        gap = path["arithmetic_average"] - path["geometric_average"]
        min_gap = min(min_gap, gap)
        min_res = min(min_res, path["residual"])
        min_v = min(min_v, path["minimum_variance"])
        rows.append({"signs": signs, **{k: path[k] for k in (
            "arithmetic_average", "geometric_average", "raw_payoff",
            "geometric_payoff", "residual")}})
    return {
        "path_count": count,
        "probability_mass": mass,
        "raw_payoff_expectation": e_x,
        "geometric_payoff_expectation": e_y,
        "residual_expectation": e_d,
        "minimum_average_gap": min_gap,
        "minimum_residual": min_res,
        "minimum_variance": min_v,
        "paths": rows,
    }


def simulate_weak_euler_payoffs(
    spec: HestonWeakEulerSpec,
    *,
    paths: int,
    seed: int,
    chunk_size: int = 65536,
    keep_samples: bool = True,
) -> dict[str, object]:
    """Vectorized chunked weak-Euler Monte Carlo, independent of the scalar
    reference.

    Accumulates payoff moments online.  When ``keep_samples`` is true the
    per-path raw payoff and residual arrays are returned for cap-grid and
    quantile analysis (8 bytes per path per array).
    """
    if paths < 1 or chunk_size < 1:
        raise ValueError("paths and chunk_size must be positive")
    h = spec.step
    sqrt_h = math.sqrt(h)
    rho_bar = math.sqrt(1.0 - spec.rho * spec.rho)
    rng = np.random.default_rng(seed)

    sums = {k: 0.0 for k in ("x", "y", "d", "xx", "yy", "dd", "xy")}
    min_gap = math.inf
    min_res = math.inf
    min_v = math.inf
    negative_variance = 0
    nonfinite = 0
    x_samples: list[np.ndarray] = []
    d_samples: list[np.ndarray] = []

    done = 0
    while done < paths:
        m = min(chunk_size, paths - done)
        signs = rng.integers(0, 2, size=(m, spec.n_steps, 2), dtype=np.int8)
        signs = 2 * signs - 1
        r = np.zeros(m)
        v = np.full(m, float(spec.v0))
        sum_exp_r = np.zeros(m)
        sum_r = np.zeros(m)
        for n in range(spec.n_steps):
            neg = v < 0.0
            if neg.any():
                negative_variance += int(neg.sum())
                raise FloatingPointError(
                    f"negative pre-sqrt variance encountered at step {n}; "
                    f"min v = {float(v.min())}"
                )
            sq = np.sqrt(v)
            a = signs[:, n, 0]
            b = signs[:, n, 1]
            r = r + (spec.rate - 0.5 * v) * h + sq * (spec.rho * a + rho_bar * b) * sqrt_h
            v = v + spec.kappa * (spec.theta - v) * h + spec.xi * sq * a * sqrt_h
            min_v = min(min_v, float(v.min()))
            sum_exp_r += np.exp(r)
            sum_r += r
        arithmetic = spec.s0 * sum_exp_r / spec.n_steps
        geometric = spec.s0 * np.exp(sum_r / spec.n_steps)
        nonfinite += int((~np.isfinite(arithmetic)).sum())
        if spec.option_type == "call":
            x = np.maximum(arithmetic - spec.strike, 0.0)
            y = np.maximum(geometric - spec.strike, 0.0)
            d = x - y
        else:
            x = np.maximum(spec.strike - arithmetic, 0.0)
            y = np.maximum(spec.strike - geometric, 0.0)
            d = y - x
        sums["x"] += float(x.sum())
        sums["y"] += float(y.sum())
        sums["d"] += float(d.sum())
        sums["xx"] += float(np.dot(x, x))
        sums["yy"] += float(np.dot(y, y))
        sums["dd"] += float(np.dot(d, d))
        sums["xy"] += float(np.dot(x, y))
        min_gap = min(min_gap, float((arithmetic - geometric).min()))
        min_res = min(min_res, float(d.min()))
        if keep_samples:
            x_samples.append(x)
            d_samples.append(d)
        done += m

    n = float(paths)
    mean_x, mean_y, mean_d = sums["x"] / n, sums["y"] / n, sums["d"] / n
    var_x = max(sums["xx"] / n - mean_x**2, 0.0) * n / (n - 1.0)
    var_y = max(sums["yy"] / n - mean_y**2, 0.0) * n / (n - 1.0)
    var_d = max(sums["dd"] / n - mean_d**2, 0.0) * n / (n - 1.0)
    cov_xy = (sums["xy"] / n - mean_x * mean_y) * n / (n - 1.0)
    corr = cov_xy / math.sqrt(var_x * var_y) if var_x > 0.0 and var_y > 0.0 else float("nan")
    disc = spec.discount
    result: dict[str, object] = {
        "paths": paths,
        "seed": seed,
        "chunk_size": chunk_size,
        "spec": {
            "s0": spec.s0, "v0": spec.v0, "rate": spec.rate, "rho": spec.rho,
            "kappa": spec.kappa, "theta": spec.theta, "xi": spec.xi,
            "maturity": spec.maturity, "n_steps": spec.n_steps,
            "strike": spec.strike, "option_type": spec.option_type,
        },
        "arithmetic_price_discounted": disc * mean_x,
        "arithmetic_price_se": disc * math.sqrt(var_x / n),
        "geometric_price_discounted": disc * mean_y,
        "geometric_price_se": disc * math.sqrt(var_y / n),
        "residual_mean_discounted": disc * mean_d,
        "residual_se": disc * math.sqrt(var_d / n),
        "payoff_variance": var_x,
        "geometric_variance": var_y,
        "residual_variance": var_d,
        "payoff_geometric_covariance": cov_xy,
        "payoff_geometric_correlation": corr,
        "minimum_average_gap": min_gap,
        "minimum_residual": min_res,
        "minimum_variance": min_v,
        "negative_variance_count": negative_variance,
        "nonfinite_count": nonfinite,
        "variance_certificate": variance_positivity_certificate(spec),
    }
    if keep_samples:
        x_all = np.concatenate(x_samples)
        d_all = np.concatenate(d_samples)
        qs = (0.5, 0.9, 0.99, 0.999, 1.0)
        result["raw_payoff_quantiles"] = {str(q): float(np.quantile(x_all, q)) for q in qs}
        result["residual_quantiles"] = {str(q): float(np.quantile(d_all, q)) for q in qs}
        result["_raw_payoff_samples"] = x_all
        result["_residual_samples"] = d_all
    return result
