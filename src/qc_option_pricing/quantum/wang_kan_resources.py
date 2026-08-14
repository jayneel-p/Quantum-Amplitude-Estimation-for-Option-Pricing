"""Wang-Kan weak-Euler resource model and control-variate variants.

Transcribes the primitive T-count formulas of Wang and Kan, *Option pricing
under stochastic volatility on a quantum computer*, Quantum 8, 1504 (2024),
Appendix A (their Table 8) and the module compositions of Section 4.5, and
adds the T-cost deltas of the unit-coefficient geometric control variate.

Citations by equation number of the published paper:
  Eq. (64)   iterative amplitude estimation query bound
  Eq. (113)  total T = Noracle [2 T(A) + T(Toffoli_{m+1})]
  Eq. (118)  U1, weak Euler
  Eq. (120)  U2, Asian option
  Eq. (124)  U3 = ARCSIN_SQRT + Usin
  Eqs. (128)-(158), (172)  primitive costs

The piecewise-polynomial parameters (M, d) for EXP and ARCSIN_SQRT at the
paper's epsilon values are not published.  `calibrate_ppoly` inverts the
published module totals over an (M, d) grid and reports every candidate;
this is a disclosed calibration, not a reproduction, and it does not affect
the control-variate deltas, which reuse the same calibrated EXP cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Primitive T-counts, Wang-Kan Table 8
# ----------------------------------------------------------------------

def t_toffoli(k: int) -> int:
    """k-qubit Toffoli via their default method [50]: 4k - 8 T, k >= 3."""
    if k < 3:
        return 0
    return 4 * k - 8


def t_add(n: int) -> int:
    return 4 * n - 4                       # Eq. (129) cost


def t_add_const(n: int) -> int:
    return 4 * n - 8                       # Eq. (131) cost; SUB_CONST equal


def t_c_add(n: int) -> int:
    return 8 * n - 4                       # Eqs. (133)-(134) cost


def t_comp_const(n: int) -> int:
    return 8 * n - 16                      # Eq. (136) cost


def t_mul_const(n: int, p: int) -> int:
    return 2 * n * n - 6 * n + 4 * p * n - 4 * p * p + 4 * p   # Eq. (144)


def t_sqrt(n: int) -> int:
    half = (n + 1) // 2
    return 8 * half * half + 32 * half - 8                     # Eq. (148)


def t_ppoly(n: int, p: int, m_pieces: int, degree: int) -> int:
    """Eq. (152)."""
    core = n * n - n + 2 * p * n - 2 * p * p + 2 * p - 1
    logm = max(math.ceil(math.log2(m_pieces)), 0) if m_pieces > 1 else 0
    return (8 * degree * core + 32 * m_pieces * (n - 2)
            + 16 * degree * m_pieces * (logm - 1))


def t_arcsin_sqrt(n: int, p: int, m_pieces: int, degree: int) -> int:
    """Eq. (157): 2 PPoly + 2 COMP_CONST + 2 SQRT + 4 c-SUB."""
    return (2 * t_ppoly(n, p, m_pieces, degree) + 2 * t_comp_const(n)
            + 2 * t_sqrt(n) + 4 * t_c_add(n))


def t_usin(n: int, eps_sin: float) -> float:
    return 4 * n + 3.3 * math.log2(2.0 / eps_sin)              # Eq. (172)


# ----------------------------------------------------------------------
# Module compositions, Section 4.5
# ----------------------------------------------------------------------

def t_u1_weak(n_steps: int, n: int, p: int) -> int:
    """Eq. (118): N [5 ADD + 2 ADD_CONST + 10 MUL_CONST + 2 SQRT]."""
    per_step = (5 * t_add(n) + 2 * t_add_const(n)
                + 10 * t_mul_const(n, p) + 2 * t_sqrt(n))
    return n_steps * per_step


def t_u2_asian(n_steps: int, n: int, p: int, t_exp: float) -> float:
    """Eq. (120): N EXP + (N-1) ADD + MUL_CONST + SUB_CONST + n Toffoli3."""
    return (n_steps * t_exp + (n_steps - 1) * t_add(n)
            + t_mul_const(n, p) + t_add_const(n) + n * t_toffoli(3))


def delta_u2_control_variate(n_steps: int, n: int, p: int, t_exp: float) -> float:
    """Added payoff arithmetic for the unit-coefficient geometric control.

    Reuses the N stored log-return registers: (N-1) additions for the sum of
    log returns, one MUL_CONST for the 1/N average, one further EXP for the
    geometric mean, one SUB_CONST for the strike hinge input, n Toffoli3 for
    the hinge conditional copy, and one ADD-width subtraction for the
    residual.  Identical operation counts for the call and put identities.
    """
    return ((n_steps - 1) * t_add(n) + t_mul_const(n, p) + t_exp
            + t_add_const(n) + n * t_toffoli(3) + t_add(n))


def t_u3(n: int, p: int, m_pieces: int, degree: int, eps_sin: float) -> float:
    return t_arcsin_sqrt(n, p, m_pieces, degree) + t_usin(n, eps_sin)  # Eq. (124)


def delta_threshold_encoder(n: int, threshold_bits: int) -> float:
    """Replace U3 with a uniform-threshold comparator (repository encoding).

    Costed as one comparator at the wider of the payoff and threshold widths
    plus the conditional copy; a deliberate over-allowance rather than a
    synthesized count.
    """
    width = max(n, threshold_bits)
    return t_comp_const(width) + t_add(width)


def t_q(t_a: float, total_qubits: int) -> float:
    """Eq. (113): Q = 2 T(A) + T(Toffoli over all qubits + 1)."""
    return 2.0 * t_a + t_toffoli(total_qubits + 1)


def n_oracle_queries(eps_amplitude: float, delta: float,
                     rounding: str = "nearest") -> int:
    """Eq. (64) IQAE query bound.

    At eps = 1e-3, delta = 0.1 the raw value is 7363.116.  The published
    count is 7363, which implies rounding to nearest (or truncation), not
    the upward rounding a bound strictly requires; ``rounding='ceil'``
    gives 7364.  The one-query difference is recorded, not hidden.
    """
    if not 0.0 < eps_amplitude < 1.0:
        raise ValueError("eps_amplitude must lie in (0, 1)")
    value = (1.4 / eps_amplitude) * math.log(
        (2.0 / delta) * math.log2(math.pi / (4.0 * eps_amplitude)))
    if rounding == "ceil":
        return math.ceil(value)
    if rounding == "nearest":
        return round(value)
    raise ValueError("rounding must be 'nearest' or 'ceil'")


def calibrate_ppoly(target_t: float, n: int, p: int,
                    max_m: int = 64, max_d: int = 8,
                    builder=t_ppoly) -> list[dict]:
    """All (M, d) whose cost is within 2% of a published module total."""
    rows = []
    for degree in range(1, max_d + 1):
        for m_pieces in range(1, max_m + 1):
            value = builder(n, p, m_pieces, degree)
            gap = abs(value - target_t) / target_t
            if gap <= 0.02:
                rows.append({"m_pieces": m_pieces, "degree": degree,
                             "t_count": value, "relative_gap": gap})
    return sorted(rows, key=lambda r: r["relative_gap"])


@dataclass(frozen=True)
class MatchedComparison:
    """Total-T comparison at one common dollar amplitude-estimation budget."""

    dollar_ae_budget: float
    delta: float
    discount: float
    range_raw: float
    range_cv: float
    t_q_raw: float
    t_q_cv: float

    def row(self) -> dict:
        eps_raw = self.dollar_ae_budget / (self.discount * self.range_raw)
        eps_cv = self.dollar_ae_budget / (self.discount * self.range_cv)
        nq_raw = n_oracle_queries(eps_raw, self.delta)
        nq_cv = n_oracle_queries(eps_cv, self.delta)
        total_raw = nq_raw * self.t_q_raw
        total_cv = nq_cv * self.t_q_cv
        return {
            "dollar_ae_budget": self.dollar_ae_budget,
            "delta": self.delta,
            "eps_amplitude_raw": eps_raw,
            "eps_amplitude_cv": eps_cv,
            "queries_raw": nq_raw,
            "queries_cv": nq_cv,
            "query_ratio": nq_raw / nq_cv,
            "t_q_raw": self.t_q_raw,
            "t_q_cv": self.t_q_cv,
            "t_q_overhead_cv": self.t_q_cv / self.t_q_raw,
            "total_t_raw": total_raw,
            "total_t_cv": total_cv,
            "total_t_ratio": total_raw / total_cv,
        }
