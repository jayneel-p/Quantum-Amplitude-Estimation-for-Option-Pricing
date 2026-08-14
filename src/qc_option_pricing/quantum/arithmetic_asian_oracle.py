"""Arithmetic Clifford+T Asian-control-variate oracle.

This module constructs a complete finite-precision ``A`` operator for a
uniform discrete-shock Asian call.  It intentionally does not evaluate an
exponential with a price-indexed QROM and does not use a bank of payoff
rotations.  Instead:

* each shock selects a classically precomputed fixed-point *increment factor*;
* reversible shift-and-add multiplication evolves arithmetic and geometric
  price registers;
* ripple-carry adders compute both call payoffs and their nonnegative residual;
* a uniform threshold register encodes ``residual / 2**m`` with a comparator.

The geometric control admits two implementations, selected by
``AsianGridSpec.geometric_leg``.  ``'per_date'`` evolves a second price
register with one shock-selected multiplication per date, mirroring the
arithmetic leg.  ``'collapsed'`` uses the fact that for a binary shock grid
``log G_N`` is affine in the shock bits: the weighted sum
``s = sum_d (N - d) b_d`` is accumulated into one ``ceil(log2(s_max))``-bit
register with one controlled constant addition per date, and ``exp`` of it
factorises over the bits of ``s``, so the geometric register needs only one
controlled constant multiplication per bit of ``s``.  Both legs round every
geometric operation downward, which is what makes the encoded residual
nonnegative on every path.  A third value, ``'none'``, omits the control
entirely: the arithmetic leg is evolved exactly as before and the same
uniform-threshold comparator encodes ``min(arithmetic payoff, cap)``
directly, with the cap taken from ``payoff_cap`` instead of
``residual_payoff_cap``.  It exists so the control-free baseline is a
buildable circuit rather than a ledger recount.

For the binary-shock construction every circuit operation is H, X, CX, or
Toffoli.  Resource estimates use the exact operation structure implemented
here and the explicit convention Toffoli = 7 T.  The construction is exact
for its directed-rounding finite model; it is not an exact representation of
continuous Black--Scholes GBM.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from statistics import NormalDist
from typing import Iterable, Sequence

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder

from qc_option_pricing.quantum.asian_oracle import AsianGridSpec


def _ceil_stable(value: float) -> int:
    tolerance = 32.0 * math.ulp(max(1.0, abs(value)))
    return int(math.ceil(value - tolerance))


def _floor_stable(value: float) -> int:
    tolerance = 32.0 * math.ulp(max(1.0, abs(value)))
    return int(math.floor(value + tolerance))


def _ceil_product(value: int, factor: int, fraction_bits: int) -> int:
    scale = 1 << fraction_bits
    return (value * factor + scale - 1) >> fraction_bits


def _floor_product(value: int, factor: int, fraction_bits: int) -> int:
    return (value * factor) >> fraction_bits


def _product_width(
    *,
    value_bits: int,
    fraction_bits: int,
    maximum_value: int,
    factors: Sequence[int],
) -> int:
    """Width of a product register holding ``value * factor + offset``."""

    largest = max(factors)
    maximum_product = maximum_value * largest + (1 << fraction_bits) - 1
    return max(
        value_bits + max(1, largest.bit_length()),
        maximum_product.bit_length(),
        fraction_bits + value_bits,
    )


def uniform_normal_midpoint_grid(
    shock_qubits: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a uniform, moment-normalized normal-quantile grid.

    Every point has probability ``2**(-shock_qubits)`` and can therefore be
    prepared exactly with Hadamards.  The representative points are normal
    midpoint quantiles, rescaled to have exactly zero empirical mean and unit
    empirical second moment.  This is a finite-grid approximation, not a
    Gaussian quadrature rule or a continuous-normal error certificate.
    """

    if isinstance(shock_qubits, bool) or not isinstance(shock_qubits, Integral) or shock_qubits < 1:
        raise ValueError("shock_qubits must be a positive integer")
    point_count = 1 << shock_qubits
    normal = NormalDist()
    points = tuple(
        normal.inv_cdf((index + 0.5) / point_count) for index in range(point_count)
    )
    scale = math.sqrt(math.fsum(point * point for point in points) / point_count)
    normalized = tuple(point / scale for point in points)
    return normalized, tuple(1.0 / point_count for _ in normalized)


@dataclass(frozen=True)
class ArithmeticAsianModel:
    """Integer model implemented by :func:`build_arithmetic_asian_oracle`.

    ``geometric_factors`` is populated only by the ``'per_date'`` geometric
    leg and ``geometric_chain_factors``, ``shock_weight_sum`` and
    ``shock_weight_bits`` only by the ``'collapsed'`` leg; each leg leaves the
    other's fields empty or zero.  ``product_bits`` is the width of the
    arithmetic leg's product registers and ``geometric_product_bits`` that of
    the geometric leg's.  The two coincide for ``'per_date'``, whose legs share
    one multiplier table range.  ``'none'`` has no geometric leg, so every
    geometric field is empty or zero, ``requested_residual_cap_numerator``
    holds the raw payoff cap from ``payoff_cap``, and the control fields of
    the geometric dynamic program stay zero.
    """

    spec: AsianGridSpec
    multiplier_fraction_bits: int
    factor_scale: int
    price_factors: tuple[int, ...]
    geometric_factors: tuple[tuple[int, ...], ...]
    geometric_chain_factors: tuple[int, ...]
    shock_weight_sum: int
    shock_weight_bits: int
    initial_price: int
    initial_geometric: int
    maximum_prices: tuple[int, ...]
    maximum_geometric_values: tuple[int, ...]
    maximum_total: int
    maximum_residual: int
    value_bits: int
    multiplier_bits: int
    product_bits: int
    geometric_product_bits: int
    total_bits: int
    residual_bits: int
    threshold_bits: int
    requested_residual_cap_numerator: int
    normalization_numerator: int
    geometric_payoff_count_numerator: int
    geometric_payoff_count_denominator: int
    geometric_dp_peak_states: int
    geometric_control_undiscounted_override: float | None
    geometric_control_standard_error_undiscounted: float | None

    @property
    def geometric_leg(self) -> str:
        return self.spec.geometric_leg

    @property
    def geometric_control_undiscounted(self) -> float:
        if self.spec.geometric_leg == "none":
            raise ValueError(
                "geometric_leg='none' encodes the raw arithmetic payoff and has "
                "no geometric control"
            )
        if self.geometric_control_undiscounted_override is not None:
            return self.geometric_control_undiscounted_override
        return self.geometric_payoff_count_numerator / (
            self.geometric_payoff_count_denominator * self.spec.price_scale
        )

    @property
    def normalization_dollars(self) -> float:
        return self.normalization_numerator / (
            self.spec.n_dates * self.spec.price_scale
        )


@dataclass(frozen=True)
class ArithmeticAsianReference:
    """Independent exhaustive reference for a tractable arithmetic model.

    Under ``geometric_leg='none'`` there is no control, so
    ``geometric_payoff_undiscounted`` is zero and the residual fields carry the
    arithmetic payoff itself: ``residual_payoff_undiscounted`` equals
    ``arithmetic_payoff_undiscounted`` and
    ``clipped_residual_payoff_undiscounted`` is its ``payoff_cap`` clip.
    """

    path_count: int
    probability_mass: float
    arithmetic_payoff_undiscounted: float
    geometric_payoff_undiscounted: float
    residual_payoff_undiscounted: float
    clipped_residual_payoff_undiscounted: float
    objective_probability: float
    minimum_residual_numerator: int
    maximum_residual_numerator: int


@dataclass(frozen=True)
class PrimitiveCounts:
    """Counts before replacing each exact Toffoli by Clifford+T."""

    h: int = 0
    x: int = 0
    z: int = 0
    cx: int = 0
    ccx: int = 0

    def __add__(self, other: "PrimitiveCounts") -> "PrimitiveCounts":
        return PrimitiveCounts(
            h=self.h + other.h,
            x=self.x + other.x,
            z=self.z + other.z,
            cx=self.cx + other.cx,
            ccx=self.ccx + other.ccx,
        )

    def scaled(self, multiplier: int) -> "PrimitiveCounts":
        return PrimitiveCounts(
            h=self.h * multiplier,
            x=self.x * multiplier,
            z=self.z * multiplier,
            cx=self.cx * multiplier,
            ccx=self.ccx * multiplier,
        )

    @property
    def t(self) -> int:
        return 7 * self.ccx

    @property
    def clifford_t_cx(self) -> int:
        # Standard exact 7-T Toffoli decomposition: six CX and two H.
        return self.cx + 6 * self.ccx

    @property
    def clifford_t_h(self) -> int:
        return self.h + 2 * self.ccx

    def as_dict(self) -> dict[str, int]:
        return {
            "h_before_toffoli_decomposition": self.h,
            "x": self.x,
            "z": self.z,
            "cx_before_toffoli_decomposition": self.cx,
            "toffoli": self.ccx,
            "t": self.t,
            "cx_after_standard_7t_toffoli_decomposition": self.clifford_t_cx,
            "h_after_standard_7t_toffoli_decomposition": self.clifford_t_h,
        }


@dataclass(frozen=True)
class ArithmeticAsianResourceEstimate:
    """Complete compositional logical and Clifford+T resource ledger."""

    a_qubits: int
    a_work_qubits: int
    reflection_clean_ancillas: int
    q_qubits_with_clean_reflection_ladder: int
    a_counts: PrimitiveCounts
    q_counts: PrimitiveCounts
    component_counts: dict[str, PrimitiveCounts]
    qrom_rows: int = 0
    arbitrary_rotations: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "a_qubits": self.a_qubits,
            "a_work_qubits": self.a_work_qubits,
            "reflection_clean_ancillas": self.reflection_clean_ancillas,
            "q_qubits_with_clean_reflection_ladder": self.q_qubits_with_clean_reflection_ladder,
            "a_counts": self.a_counts.as_dict(),
            "q_counts": self.q_counts.as_dict(),
            "components": {
                name: counts.as_dict() for name, counts in self.component_counts.items()
            },
            "qrom_rows": self.qrom_rows,
            "arbitrary_rotations": self.arbitrary_rotations,
            "synthesis_model": (
                "exact H/X/Z/CX/Toffoli network; each Toffoli replaced by the "
                "standard exact 7-T, 6-CX, 2-H decomposition"
            ),
        }


@dataclass(frozen=True)
class ArithmeticAsianOracle:
    """Executable ``A`` operator and its decoding convention."""

    circuit: QuantumCircuit
    model: ArithmeticAsianModel
    objective_qubit: int
    shock_qubits: tuple[int, ...]
    threshold_qubits: tuple[int, ...]
    work_qubits: tuple[int, ...]

    def post_process(self, objective_probability: float) -> float:
        if not 0.0 <= objective_probability <= 1.0:
            raise ValueError("objective probability must lie in [0, 1]")
        encoded = self.model.normalization_dollars * objective_probability
        spec = self.model.spec
        if spec.geometric_leg == "none":
            # The comparator already encodes the arithmetic payoff itself, so
            # there is no classical control to restore.
            undiscounted = encoded
        else:
            undiscounted = encoded + self.model.geometric_control_undiscounted
        return math.exp(-spec.rate * spec.maturity) * undiscounted


def build_arithmetic_asian_model(
    spec: AsianGridSpec,
    *,
    multiplier_fraction_bits: int = 12,
    geometric_control_undiscounted_override: float | None = None,
    geometric_control_standard_error_undiscounted: float | None = None,
) -> ArithmeticAsianModel:
    """Create the directed-rounding model used by the arithmetic circuit.

    The shock distribution must be uniform.  This restriction eliminates
    arbitrary state-preparation rotations; the paper's coarse 252-date binary
    Gauss--Hermite instance satisfies it exactly.  By default the finite
    geometric-control expectation is computed exactly with a dynamic program.
    For high-precision grids where that classical dynamic program is too large,
    a separately calibrated undiscounted control and its standard error may be
    supplied explicitly.  The override does not alter the circuit; it makes
    the classical restoration uncertainty an explicit part of the price error.

    Under ``geometric_leg='collapsed'`` the geometric value depends on the
    shocks only through the weighted sum ``sum_d (N - d) b_d``, so that
    dynamic program has at most ``N (N + 1) / 2 + 1`` states at any precision
    and the override is rarely needed.

    Under ``geometric_leg='none'`` there is no geometric control at all, so
    the dynamic program is skipped, an override is rejected, and the encoded
    quantity is ``min(arithmetic payoff, cap)`` with the cap taken from
    ``payoff_cap``.
    """

    if (
        isinstance(multiplier_fraction_bits, bool)
        or not isinstance(multiplier_fraction_bits, Integral)
        or multiplier_fraction_bits < 1
    ):
        raise ValueError("multiplier_fraction_bits must be a positive integer")
    n_states = len(spec.shock_points)
    expected_probability = 1.0 / n_states
    if any(
        not math.isclose(probability, expected_probability, rel_tol=0.0, abs_tol=1e-14)
        for probability in spec.shock_probabilities
    ):
        raise ValueError(
            "the arithmetic Clifford+T construction currently requires uniform shocks"
        )

    factor_scale = 1 << multiplier_fraction_bits
    dt = spec.maturity / spec.n_dates
    drift = spec.rate - 0.5 * spec.volatility**2
    diffusion = spec.volatility * math.sqrt(dt)

    price_factors = tuple(
        _ceil_stable(math.exp(drift * dt + diffusion * shock) * factor_scale)
        for shock in spec.shock_points
    )
    average_drift_time = dt * (spec.n_dates + 1) / 2.0
    initial_price = _ceil_stable(spec.s0 * spec.price_scale)
    collapsed = spec.geometric_leg == "collapsed"
    raw = spec.geometric_leg == "none"
    if raw:
        if spec.residual_payoff_cap is not None:
            raise ValueError(
                "geometric_leg='none' encodes the raw payoff; cap it with "
                "payoff_cap, not residual_payoff_cap"
            )
        if (
            geometric_control_undiscounted_override is not None
            or geometric_control_standard_error_undiscounted is not None
        ):
            raise ValueError(
                "geometric_leg='none' has no geometric control to calibrate"
            )
        shock_weight_sum = 0
        shock_weight_bits = 0
        initial_geometric = 0
        geometric_factors: tuple[tuple[int, ...], ...] = ()
        geometric_chain_factors: tuple[int, ...] = ()
        geometric_multipliers: tuple[int, ...] = ()
    elif collapsed:
        if spec.shock_qubits != 1:
            raise ValueError(
                "the collapsed geometric leg requires one binary shock qubit per date"
            )
        low, high = spec.shock_points
        if not high > low:
            raise ValueError(
                "the collapsed geometric leg requires an increasing binary shock grid"
            )
        shock_weight_sum = spec.n_dates * (spec.n_dates + 1) // 2
        shock_weight_bits = max(1, shock_weight_sum.bit_length())
        # log G_N is affine in the shock bits: the shock-dependent part is
        # (diffusion / N) * (low * W + (high - low) * s) with
        # s = sum_d (N - d) b_d, so the constant absorbs the drift and the
        # low * W offset and exp(alpha * s) factorises over the bits of s.
        chain_exponent = diffusion * (high - low) / spec.n_dates
        initial_geometric = _floor_stable(
            spec.s0
            * math.exp(
                drift * average_drift_time
                + diffusion * low * shock_weight_sum / spec.n_dates
            )
            * spec.price_scale
        )
        geometric_factors: tuple[tuple[int, ...], ...] = ()
        geometric_chain_factors = tuple(
            _floor_stable(math.exp(chain_exponent * (1 << bit)) * factor_scale)
            for bit in range(shock_weight_bits)
        )
        geometric_multipliers: tuple[int, ...] = geometric_chain_factors
    else:
        shock_weight_sum = 0
        shock_weight_bits = 0
        initial_geometric = _floor_stable(
            spec.s0 * math.exp(drift * average_drift_time) * spec.price_scale
        )
        geometric_factors = tuple(
            tuple(
                _floor_stable(
                    math.exp(diffusion * (spec.n_dates - date) * shock / spec.n_dates)
                    * factor_scale
                )
                for shock in spec.shock_points
            )
            for date in range(spec.n_dates)
        )
        geometric_chain_factors = ()
        geometric_multipliers = tuple(
            factor for row in geometric_factors for factor in row
        )
    all_factors = (*price_factors, *geometric_multipliers)
    if min(all_factors) <= 0:
        raise ValueError("multiplier precision rounds at least one positive factor to zero")
    if not raw and initial_geometric <= 0:
        # The collapsed leg's constant absorbs exp(-diffusion * W / N), so a
        # price scale that is fine for 'per_date' can still floor it to zero.
        raise ValueError(
            "price_scale is too coarse: the initial geometric value rounds to zero"
        )

    current_maximum = initial_price
    maximum_prices: list[int] = []
    for _ in range(spec.n_dates):
        current_maximum = max(
            _ceil_product(current_maximum, factor, multiplier_fraction_bits)
            for factor in price_factors
        )
        maximum_prices.append(current_maximum)

    maximum_geometric_values: list[int] = []
    chain_values: list[int] = []
    if collapsed:
        # Tabulate the chain on every reachable weighted sum, in increasing
        # order of the sum, so the per-step maxima are exact rather than the
        # unreachable all-ones bound.  Only the final step can overshoot the
        # largest reachable sum, so the concatenation index p + 2**bit is
        # always the position the extended entry belongs at.
        chain_values = [initial_geometric]
        for bit, factor in enumerate(geometric_chain_factors):
            if len(chain_values) != 1 << bit:
                raise AssertionError("weighted-sum enumeration lost a reachable state")
            chain_values = (
                chain_values
                + [
                    _floor_product(value, factor, multiplier_fraction_bits)
                    for value in chain_values
                ]
            )[: shock_weight_sum + 1]
            maximum_geometric_values.append(max(chain_values))
    elif not raw:
        geometric_maximum = initial_geometric
        for factors in geometric_factors:
            geometric_maximum = max(
                _floor_product(geometric_maximum, factor, multiplier_fraction_bits)
                for factor in factors
            )
            maximum_geometric_values.append(geometric_maximum)

    maximum_total = sum(maximum_prices)
    strike_sum = spec.n_dates * spec.strike_integer
    maximum_residual = max(0, maximum_total - strike_sum)
    if maximum_residual == 0:
        raise ValueError("the arithmetic payoff is identically zero on the configured range")

    maximum_value = max(
        initial_price,
        initial_geometric,
        *maximum_prices,
        *maximum_geometric_values,
    )
    value_bits = max(1, maximum_value.bit_length())
    multiplier_bits = max(1, max(all_factors).bit_length())
    if raw:
        # There is no geometric leg, so no geometric product registers exist.
        product_bits = _product_width(
            value_bits=value_bits,
            fraction_bits=multiplier_fraction_bits,
            maximum_value=maximum_value,
            factors=price_factors,
        )
        geometric_product_bits = 0
    elif collapsed:
        # The chain factors are exponentially larger than the per-date ones, so
        # the two legs size their product registers separately and the
        # arithmetic leg keeps exactly the width it has under 'per_date'.
        product_bits = _product_width(
            value_bits=value_bits,
            fraction_bits=multiplier_fraction_bits,
            maximum_value=maximum_value,
            factors=price_factors,
        )
        geometric_product_bits = _product_width(
            value_bits=value_bits,
            fraction_bits=multiplier_fraction_bits,
            maximum_value=maximum_value,
            factors=geometric_chain_factors,
        )
    else:
        product_bits = _product_width(
            value_bits=value_bits,
            fraction_bits=multiplier_fraction_bits,
            maximum_value=maximum_value,
            factors=all_factors,
        )
        geometric_product_bits = product_bits
    occupied_product_widths = (
        (product_bits,) if raw else (product_bits, geometric_product_bits)
    )
    if min(occupied_product_widths) < multiplier_fraction_bits + value_bits:
        raise AssertionError("product register does not expose a complete output slice")
    total_bits = max(1, maximum_total.bit_length())
    residual_bits = max(1, maximum_residual.bit_length())
    if raw:
        # The raw oracle clips the arithmetic payoff itself, so its cap comes
        # from payoff_cap and is generally much larger than a residual cap;
        # the threshold register is sized from it, not reused from a residual
        # configuration.
        if spec.payoff_cap is None:
            requested_residual_cap = maximum_residual
        else:
            requested_residual_cap = int(
                round(spec.payoff_cap * spec.n_dates * spec.price_scale)
            )
            if requested_residual_cap < 1:
                raise ValueError("payoff_cap is below one encoded payoff unit")
            if requested_residual_cap > maximum_residual:
                raise ValueError(
                    "payoff_cap exceeds the maximum encoded arithmetic payoff"
                )
    elif spec.residual_payoff_cap is None:
        requested_residual_cap = maximum_residual
    else:
        requested_residual_cap = int(
            round(spec.residual_payoff_cap * spec.n_dates * spec.price_scale)
        )
        if requested_residual_cap < 1:
            raise ValueError("residual_payoff_cap is below one encoded residual unit")
    threshold_bits = max(1, (requested_residual_cap - 1).bit_length())
    normalization_numerator = 1 << threshold_bits

    if raw:
        # There is no geometric control, so there is no dynamic program to
        # run and nothing for post-processing to restore.
        payoff_count_numerator = 0
        payoff_count_denominator = 0
        peak_states = 0
    elif geometric_control_undiscounted_override is not None:
        if not math.isfinite(geometric_control_undiscounted_override) or geometric_control_undiscounted_override < 0.0:
            raise ValueError("geometric-control override must be finite and nonnegative")
        if geometric_control_standard_error_undiscounted is None:
            raise ValueError("an external geometric control requires its standard error")
        if (
            not math.isfinite(geometric_control_standard_error_undiscounted)
            or geometric_control_standard_error_undiscounted < 0.0
        ):
            raise ValueError("geometric-control standard error must be finite and nonnegative")
        payoff_count_numerator = 0
        payoff_count_denominator = 0
        peak_states = 0
    else:
        if geometric_control_standard_error_undiscounted is not None:
            raise ValueError("a geometric-control standard error requires an override")
        # Exact classical preprocessing for the finite geometric control.  Counts
        # remain integers, so no stochastic or floating-point probability error is
        # introduced.
        payoff_count_denominator = n_states**spec.n_dates
        if collapsed:
            # The collapsed leg is a function of the weighted shock sum alone,
            # so the dynamic program runs over that sum: at most
            # shock_weight_sum + 1 states rather than 2**value_bits.
            weight_counts = [0] * (shock_weight_sum + 1)
            weight_counts[0] = 1
            reachable = 1
            for date in range(spec.n_dates):
                weight = spec.n_dates - date
                for value in range(reachable - 1, -1, -1):
                    count = weight_counts[value]
                    if count:
                        weight_counts[value + weight] += count
                reachable += weight
            peak_states = reachable
            if reachable != shock_weight_sum + 1:
                raise AssertionError("weighted-sum dynamic program lost a reachable state")
            payoff_count_numerator = sum(
                count * max(value - spec.strike_integer, 0)
                for count, value in zip(weight_counts, chain_values)
            )
            if sum(weight_counts) != payoff_count_denominator:
                raise AssertionError("geometric-control dynamic program lost probability mass")
        else:
            # At most 2**value_bits states can survive any level.
            distribution: Counter[int] = Counter({initial_geometric: 1})
            peak_states = 1
            for factors in geometric_factors:
                updated: Counter[int] = Counter()
                for value, count in distribution.items():
                    for factor in factors:
                        updated[_floor_product(value, factor, multiplier_fraction_bits)] += count
                distribution = updated
                peak_states = max(peak_states, len(distribution))
            payoff_count_numerator = sum(
                count * max(value - spec.strike_integer, 0)
                for value, count in distribution.items()
            )
            if sum(distribution.values()) != payoff_count_denominator:
                raise AssertionError("geometric-control dynamic program lost probability mass")

    return ArithmeticAsianModel(
        spec=spec,
        multiplier_fraction_bits=multiplier_fraction_bits,
        factor_scale=factor_scale,
        price_factors=price_factors,
        geometric_factors=geometric_factors,
        geometric_chain_factors=geometric_chain_factors,
        shock_weight_sum=shock_weight_sum,
        shock_weight_bits=shock_weight_bits,
        initial_price=initial_price,
        initial_geometric=initial_geometric,
        maximum_prices=tuple(maximum_prices),
        maximum_geometric_values=tuple(maximum_geometric_values),
        maximum_total=maximum_total,
        maximum_residual=maximum_residual,
        value_bits=value_bits,
        multiplier_bits=multiplier_bits,
        product_bits=product_bits,
        geometric_product_bits=geometric_product_bits,
        total_bits=total_bits,
        residual_bits=residual_bits,
        threshold_bits=threshold_bits,
        requested_residual_cap_numerator=requested_residual_cap,
        normalization_numerator=normalization_numerator,
        geometric_payoff_count_numerator=payoff_count_numerator,
        geometric_payoff_count_denominator=payoff_count_denominator,
        geometric_dp_peak_states=peak_states,
        geometric_control_undiscounted_override=geometric_control_undiscounted_override,
        geometric_control_standard_error_undiscounted=(
            geometric_control_standard_error_undiscounted
        ),
    )


def _weighted_shock_sum(model: ArithmeticAsianModel, digits: Sequence[int]) -> int:
    """Value the collapsed leg's weighted-sum register holds, ``sum (N-d) b_d``."""

    return sum(
        (model.spec.n_dates - date) * digit for date, digit in enumerate(digits)
    )


def _path_values(model: ArithmeticAsianModel, digits: Sequence[int]) -> tuple[int, int, int, int]:
    spec = model.spec
    leg = spec.geometric_leg
    price = model.initial_price
    total = 0
    geometric = model.initial_geometric
    for date, digit in enumerate(digits):
        price = _ceil_product(
            price, model.price_factors[digit], model.multiplier_fraction_bits
        )
        total += price
        if leg == "per_date":
            geometric = _floor_product(
                geometric,
                model.geometric_factors[date][digit],
                model.multiplier_fraction_bits,
            )
    if leg == "collapsed":
        weighted = _weighted_shock_sum(model, digits)
        for bit, factor in enumerate(model.geometric_chain_factors):
            if (weighted >> bit) & 1:
                geometric = _floor_product(
                    geometric, factor, model.multiplier_fraction_bits
                )
    arithmetic_payoff = max(total - spec.n_dates * spec.strike_integer, 0)
    # Under 'none' the initial geometric value is zero and never evolves, so
    # the geometric payoff is zero and the residual IS the arithmetic payoff.
    geometric_payoff = max(geometric - spec.strike_integer, 0)
    residual = arithmetic_payoff - spec.n_dates * geometric_payoff
    if residual < 0:
        raise AssertionError("directed rounding failed to preserve pathwise AM--GM")
    return arithmetic_payoff, geometric_payoff, residual, total


def enumerate_arithmetic_asian(model: ArithmeticAsianModel) -> ArithmeticAsianReference:
    """Independently enumerate every shock path for a small model."""

    spec = model.spec
    n_states = len(spec.shock_points)
    path_count = n_states**spec.n_dates
    if path_count > 1_000_000:
        raise ValueError("exhaustive enumeration is limited to one million paths")
    arithmetic_sum = 0
    geometric_sum = 0
    residual_sum = 0
    clipped_residual_sum = 0
    minimum_residual = math.inf
    maximum_residual = 0
    for digits in itertools.product(range(n_states), repeat=spec.n_dates):
        arithmetic, geometric, residual, _ = _path_values(model, digits)
        arithmetic_sum += arithmetic
        geometric_sum += geometric
        residual_sum += residual
        clipped_residual_sum += min(
            residual, model.requested_residual_cap_numerator
        )
        minimum_residual = min(minimum_residual, residual)
        maximum_residual = max(maximum_residual, residual)
        if arithmetic != spec.n_dates * geometric + residual:
            raise AssertionError("encoded control-variate identity failed")
    denominator = path_count * spec.n_dates * spec.price_scale
    return ArithmeticAsianReference(
        path_count=path_count,
        probability_mass=1.0,
        arithmetic_payoff_undiscounted=arithmetic_sum / denominator,
        geometric_payoff_undiscounted=(spec.n_dates * geometric_sum) / denominator,
        residual_payoff_undiscounted=residual_sum / denominator,
        clipped_residual_payoff_undiscounted=clipped_residual_sum / denominator,
        objective_probability=clipped_residual_sum
        / (path_count * model.normalization_numerator),
        minimum_residual_numerator=int(minimum_residual),
        maximum_residual_numerator=maximum_residual,
    )


@lru_cache(maxsize=None)
def _cdkm_primitive_operations(width: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    circuit = CDKMRippleCarryAdder(width, kind="fixed").decompose(reps=2)
    operations: list[tuple[str, tuple[int, ...]]] = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in {"cx", "ccx"}:
            raise AssertionError(f"unexpected CDKM primitive {name}")
        operations.append(
            (
                name,
                tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
            )
        )
    return tuple(operations)


def _append_controlled_adder(
    circuit: QuantumCircuit,
    enable,
    addend: Sequence,
    target: Sequence,
    helper,
    c3x_temporary,
) -> None:
    """Append an exact controlled CDKM adder using only Toffoli gates.

    A controlled CX becomes one Toffoli.  A controlled Toffoli becomes a
    three-Toffoli C3X decomposition using ``c3x_temporary``, which is returned
    to zero after each primitive.
    """

    width = len(addend)
    if len(target) != width:
        raise ValueError("controlled-adder registers must have equal widths")
    mapping = [*addend, *target, helper]
    for name, indices in _cdkm_primitive_operations(width):
        qubits = [mapping[index] for index in indices]
        if name == "cx":
            circuit.ccx(enable, qubits[0], qubits[1])
        else:
            circuit.ccx(enable, qubits[0], c3x_temporary)
            circuit.ccx(c3x_temporary, qubits[1], qubits[2])
            circuit.ccx(enable, qubits[0], c3x_temporary)


def _append_equality_flag(
    circuit: QuantumCircuit,
    controls: Sequence,
    state: int,
    flag,
    work: Sequence,
) -> None:
    width = len(controls)
    flipped = [controls[bit] for bit in range(width) if not ((state >> bit) & 1)]
    if flipped:
        circuit.x(flipped)
    if width == 1:
        circuit.cx(controls[0], flag)
    elif width == 2:
        circuit.ccx(controls[0], controls[1], flag)
    else:
        if len(work) < width - 2:
            raise ValueError("insufficient equality work qubits")
        circuit.ccx(controls[0], controls[1], work[0])
        for bit in range(2, width - 1):
            circuit.ccx(work[bit - 2], controls[bit], work[bit - 1])
        circuit.ccx(work[width - 3], controls[-1], flag)
        for bit in reversed(range(2, width - 1)):
            circuit.ccx(work[bit - 2], controls[bit], work[bit - 1])
        circuit.ccx(controls[0], controls[1], work[0])
    if flipped:
        circuit.x(flipped)


def build_selected_fixed_multiplier_gate(
    *,
    value_bits: int,
    shock_bits: int,
    product_bits: int,
    fraction_bits: int,
    factors: Sequence[int],
    rounding: str,
):
    """Return ``|x,z,0> -> |x,z,x*C_z+offset>`` as an arithmetic gate."""

    if len(factors) != 1 << shock_bits:
        raise ValueError("one multiplier is required for every shock basis state")
    if rounding not in {"ceil", "floor"}:
        raise ValueError("rounding must be 'ceil' or 'floor'")
    if min(factors) <= 0:
        raise ValueError("multipliers must be positive")
    if max(factor.bit_length() + value_bits for factor in factors) > product_bits + 1:
        raise ValueError("product register is too small")

    x = QuantumRegister(value_bits, "x")
    shock = QuantumRegister(shock_bits, "z")
    product = QuantumRegister(product_bits, "product")
    constant = QuantumRegister(product_bits, "constant")
    helper = QuantumRegister(1, "helper")
    c3temp = QuantumRegister(1, "c3temp")
    equality = QuantumRegister(1, "equality")
    equality_work = (
        QuantumRegister(shock_bits - 2, "equality_work") if shock_bits > 2 else None
    )
    term = QuantumRegister(1, "term")
    registers = [x, shock, product, constant, helper, c3temp, equality]
    if equality_work is not None:
        registers.append(equality_work)
    registers.append(term)
    circuit = QuantumCircuit(*registers, name=f"mul_{rounding}")

    if rounding == "ceil":
        circuit.x(list(product[:fraction_bits]))
    work = [] if equality_work is None else list(equality_work)
    for state, factor in enumerate(factors):
        _append_equality_flag(circuit, shock, state, equality[0], work)
        for bit, input_qubit in enumerate(x):
            shifted = factor << bit
            if shifted >= 1 << product_bits:
                raise ValueError("shifted multiplier does not fit product register")
            constant_qubits = [
                constant[position]
                for position in range(product_bits)
                if (shifted >> position) & 1
            ]
            if constant_qubits:
                circuit.x(constant_qubits)
            circuit.ccx(equality[0], input_qubit, term[0])
            _append_controlled_adder(
                circuit,
                term[0],
                constant,
                product,
                helper[0],
                c3temp[0],
            )
            circuit.ccx(equality[0], input_qubit, term[0])
            if constant_qubits:
                circuit.x(constant_qubits)
        _append_equality_flag(circuit, shock, state, equality[0], work)
    return circuit.to_gate(label=f"mul-{rounding}")


def _append_positive_part(
    circuit: QuantumCircuit,
    input_qubits: Sequence,
    output_qubits: Sequence,
    subtract: int,
    scratch: Sequence,
    constant: Sequence,
    helper,
) -> None:
    width = len(scratch)
    if len(constant) < width or len(output_qubits) < len(input_qubits):
        raise ValueError("positive-part workspace is too narrow")
    for source, target in zip(input_qubits, scratch):
        circuit.cx(source, target)
    encoded_constant = (-subtract) % (1 << width)
    loaded = [constant[bit] for bit in range(width) if (encoded_constant >> bit) & 1]
    if loaded:
        circuit.x(loaded)
    adder = CDKMRippleCarryAdder(width, kind="fixed").to_gate()
    circuit.append(adder, [*constant[:width], *scratch, helper])
    sign = scratch[-1]
    circuit.x(sign)
    for source, target in zip(scratch[: len(input_qubits)], output_qubits):
        circuit.ccx(sign, source, target)
    circuit.x(sign)
    circuit.append(adder.inverse(), [*constant[:width], *scratch, helper])
    if loaded:
        circuit.x(loaded)
    for source, target in zip(input_qubits, scratch):
        circuit.cx(source, target)


def _append_constant_multiplier(
    circuit: QuantumCircuit,
    input_qubits: Sequence,
    multiplier: int,
    target: Sequence,
    constant: Sequence,
    helper,
    c3temp,
) -> None:
    width = len(target)
    for bit, enable in enumerate(input_qubits):
        shifted = multiplier << bit
        if shifted >= 1 << width:
            # The corresponding high input bit is certified zero by the model.
            continue
        loaded = [constant[position] for position in range(width) if (shifted >> position) & 1]
        if loaded:
            circuit.x(loaded)
        _append_controlled_adder(
            circuit, enable, constant[:width], target, helper, c3temp
        )
        if loaded:
            circuit.x(loaded)


def _append_quantum_less_than(
    circuit: QuantumCircuit,
    threshold: Sequence,
    residual: Sequence,
    flag,
    scratch: Sequence,
    zero_high,
    helper,
) -> None:
    """XOR ``[threshold < residual]`` into ``flag`` and clean scratch."""

    width = len(residual) + 1
    if len(scratch) < width or len(threshold) > len(residual):
        raise ValueError("quantum comparator registers have inconsistent widths")
    for source, target in zip(threshold, scratch):
        circuit.cx(source, target)
    addend = [*residual, zero_high]
    adder = CDKMRippleCarryAdder(width, kind="fixed").to_gate()
    circuit.append(adder.inverse(), [*addend, *scratch[:width], helper])
    circuit.cx(scratch[len(residual)], flag)
    circuit.append(adder, [*addend, *scratch[:width], helper])
    for source, target in zip(threshold, scratch):
        circuit.cx(source, target)


def _append_constant_less_than(
    circuit: QuantumCircuit,
    threshold: Sequence,
    cap: int,
    flag,
    scratch: Sequence,
    constant: Sequence,
    helper,
) -> None:
    """XOR ``[threshold < cap]`` into ``flag`` and clean scratch."""

    width = len(threshold) + 1
    if not 0 < cap <= 1 << len(threshold):
        raise ValueError("cap must lie in the threshold register's represented range")
    for source, target in zip(threshold, scratch):
        circuit.cx(source, target)
    encoded = (-cap) % (1 << width)
    loaded = [constant[bit] for bit in range(width) if (encoded >> bit) & 1]
    if loaded:
        circuit.x(loaded)
    adder = CDKMRippleCarryAdder(width, kind="fixed").to_gate()
    circuit.append(adder, [*constant[:width], *scratch[:width], helper])
    circuit.cx(scratch[len(threshold)], flag)
    circuit.append(adder.inverse(), [*constant[:width], *scratch[:width], helper])
    if loaded:
        circuit.x(loaded)
    for source, target in zip(threshold, scratch):
        circuit.cx(source, target)


def _append_threshold_encoder(
    circuit: QuantumCircuit,
    threshold: Sequence,
    residual: Sequence,
    objective,
    scratch: Sequence,
    constant: Sequence,
    zero_high,
    helper,
    residual_flag,
    cap_flag,
    cap_numerator: int,
) -> None:
    """Encode ``min(residual, cap) / 2**len(threshold)`` exactly."""

    normalization = 1 << len(threshold)
    if cap_numerator >= normalization:
        _append_quantum_less_than(
            circuit,
            threshold,
            residual,
            objective,
            scratch,
            zero_high,
            helper,
        )
        return
    _append_quantum_less_than(
        circuit,
        threshold,
        residual,
        residual_flag,
        scratch,
        zero_high,
        helper,
    )
    _append_constant_less_than(
        circuit,
        threshold,
        cap_numerator,
        cap_flag,
        scratch,
        constant,
        helper,
    )
    circuit.ccx(residual_flag, cap_flag, objective)
    _append_constant_less_than(
        circuit,
        threshold,
        cap_numerator,
        cap_flag,
        scratch,
        constant,
        helper,
    )
    _append_quantum_less_than(
        circuit,
        threshold,
        residual,
        residual_flag,
        scratch,
        zero_high,
        helper,
    )


def build_arithmetic_asian_oracle(
    spec_or_model: AsianGridSpec | ArithmeticAsianModel,
    *,
    multiplier_fraction_bits: int = 12,
) -> ArithmeticAsianOracle:
    """Construct an executable arithmetic QCV ``A`` operator.

    This builder is intended for exact small-width verification.  Use
    :func:`estimate_arithmetic_asian_resources` for the 252-date construction;
    it applies the identical module counts without materialising millions of
    gates.
    """

    model = (
        spec_or_model
        if isinstance(spec_or_model, ArithmeticAsianModel)
        else build_arithmetic_asian_model(
            spec_or_model, multiplier_fraction_bits=multiplier_fraction_bits
        )
    )
    spec = model.spec
    n = spec.n_dates
    q = spec.shock_qubits
    v = model.value_bits
    w = model.product_bits
    g = model.geometric_product_bits
    t = model.total_bits
    m = model.threshold_bits
    collapsed = spec.geometric_leg == "collapsed"
    raw = spec.geometric_leg == "none"
    chain_steps = len(model.geometric_chain_factors) if collapsed else n
    constant_bits = max(w, g, t + 1, m + 1)

    shock = QuantumRegister(n * q, "shock")
    threshold = QuantumRegister(m, "threshold")
    objective = QuantumRegister(1, "objective")
    price0 = QuantumRegister(v, "price0")
    geometric0 = None if raw else QuantumRegister(v, "geometric0")
    price_products = QuantumRegister(n * w, "price_products")
    shock_weight = (
        QuantumRegister(model.shock_weight_bits, "shock_weight") if collapsed else None
    )
    geometric_products = (
        None if raw else QuantumRegister(chain_steps * g, "geometric_products")
    )
    total = QuantumRegister(t, "total")
    arithmetic_payoff = QuantumRegister(t, "arithmetic_payoff")
    geometric_payoff = None if raw else QuantumRegister(t, "geometric_payoff")
    scaled_geometric_payoff = (
        None if raw else QuantumRegister(t, "scaled_geometric_payoff")
    )
    residual = None if raw else QuantumRegister(t, "residual")
    scratch = QuantumRegister(t + 1, "scratch")
    pad = QuantumRegister(t - v, "pad") if t > v else None
    constant = QuantumRegister(constant_bits, "constant")
    helper = QuantumRegister(1, "helper")
    c3temp = QuantumRegister(1, "c3temp")
    equality = QuantumRegister(1, "equality")
    equality_work = QuantumRegister(q - 2, "equality_work") if q > 2 else None
    term = QuantumRegister(1, "term")
    residual_flag = QuantumRegister(1, "residual_flag")
    cap_flag = QuantumRegister(1, "cap_flag")

    work_registers = [
        price0,
        *([] if geometric0 is None else [geometric0]),
        price_products,
        *([] if shock_weight is None else [shock_weight]),
        *([] if geometric_products is None else [geometric_products]),
        total,
        arithmetic_payoff,
        *([] if geometric_payoff is None else [geometric_payoff]),
        *([] if scaled_geometric_payoff is None else [scaled_geometric_payoff]),
        *([] if residual is None else [residual]),
        scratch,
    ]
    if pad is not None:
        work_registers.append(pad)
    work_registers.extend([constant, helper, c3temp, equality])
    if equality_work is not None:
        work_registers.append(equality_work)
    work_registers.extend([term, residual_flag, cap_flag])

    circuit = QuantumCircuit(
        shock, threshold, objective, *work_registers, name="ArithmeticAsianQCV-A"
    )
    compute = QuantumCircuit(shock, *work_registers, name="compute_residual")

    for bit in range(v):
        if (model.initial_price >> bit) & 1:
            compute.x(price0[bit])
        if not raw and (model.initial_geometric >> bit) & 1:
            compute.x(geometric0[bit])

    pad_qubits: list = [] if pad is None else list(pad)
    multiplier_work = [helper[0], c3temp[0], equality[0]]
    if equality_work is not None:
        multiplier_work.extend(equality_work)
    multiplier_work.append(term[0])

    current_price: Sequence = list(price0)
    total_adder = CDKMRippleCarryAdder(t, kind="fixed").to_gate()
    for date in range(n):
        product = list(price_products[date * w : (date + 1) * w])
        shock_slice = list(shock[date * q : (date + 1) * q])
        multiplier = build_selected_fixed_multiplier_gate(
            value_bits=v,
            shock_bits=q,
            product_bits=w,
            fraction_bits=model.multiplier_fraction_bits,
            factors=model.price_factors,
            rounding="ceil",
        )
        compute.append(
            multiplier,
            [
                *current_price,
                *shock_slice,
                *product,
                *constant[:w],
                *multiplier_work,
            ],
        )
        current_price = product[
            model.multiplier_fraction_bits : model.multiplier_fraction_bits + v
        ]
        compute.append(
            total_adder,
            [*current_price, *pad_qubits, *total, helper[0]],
        )

    current_geometric: Sequence = [] if raw else list(geometric0)
    if raw:
        # 'none' has no geometric leg: nothing to accumulate or exponentiate.
        pass
    elif collapsed:
        # One controlled constant addition per date builds sum_d (N - d) b_d,
        # then one controlled constant multiplication per bit of that sum
        # exponentiates it.  The 'off' branch multiplies by factor_scale, whose
        # product slice returns the value unchanged.
        for date in range(n):
            _append_constant_multiplier(
                compute,
                [shock[date]],
                n - date,
                shock_weight,
                constant,
                helper[0],
                c3temp[0],
            )
        chain_work = [helper[0], c3temp[0], equality[0], term[0]]
        for bit, factor in enumerate(model.geometric_chain_factors):
            product = list(geometric_products[bit * g : (bit + 1) * g])
            multiplier = build_selected_fixed_multiplier_gate(
                value_bits=v,
                shock_bits=1,
                product_bits=g,
                fraction_bits=model.multiplier_fraction_bits,
                factors=(model.factor_scale, factor),
                rounding="floor",
            )
            compute.append(
                multiplier,
                [
                    *current_geometric,
                    shock_weight[bit],
                    *product,
                    *constant[:g],
                    *chain_work,
                ],
            )
            current_geometric = product[
                model.multiplier_fraction_bits : model.multiplier_fraction_bits + v
            ]
    else:
        for date in range(n):
            product = list(geometric_products[date * g : (date + 1) * g])
            shock_slice = list(shock[date * q : (date + 1) * q])
            multiplier = build_selected_fixed_multiplier_gate(
                value_bits=v,
                shock_bits=q,
                product_bits=g,
                fraction_bits=model.multiplier_fraction_bits,
                factors=model.geometric_factors[date],
                rounding="floor",
            )
            compute.append(
                multiplier,
                [
                    *current_geometric,
                    *shock_slice,
                    *product,
                    *constant[:g],
                    *multiplier_work,
                ],
            )
            current_geometric = product[
                model.multiplier_fraction_bits : model.multiplier_fraction_bits + v
            ]

    _append_positive_part(
        compute,
        total,
        arithmetic_payoff,
        n * spec.strike_integer,
        scratch,
        constant,
        helper[0],
    )
    if not raw:
        _append_positive_part(
            compute,
            current_geometric,
            geometric_payoff,
            spec.strike_integer,
            scratch,
            constant,
            helper[0],
        )
        _append_constant_multiplier(
            compute,
            geometric_payoff[:v],
            n,
            scaled_geometric_payoff,
            constant,
            helper[0],
            c3temp[0],
        )
        compute.cx(arithmetic_payoff, residual)
        residual_subtractor = CDKMRippleCarryAdder(t, kind="fixed").to_gate().inverse()
        compute.append(
            residual_subtractor,
            [*scaled_geometric_payoff, *residual, helper[0]],
        )

    circuit.h(shock)
    circuit.h(threshold)
    compute_gate = compute.to_gate(
        label="compute-payoff" if raw else "compute-residual"
    )
    compute_qubits = [*shock, *(qubit for register in work_registers for qubit in register)]
    circuit.append(compute_gate, compute_qubits)
    # Under 'none' the residual register does not exist and the comparator
    # reads the arithmetic payoff itself.
    _append_threshold_encoder(
        circuit,
        threshold,
        arithmetic_payoff if raw else residual,
        objective[0],
        scratch,
        constant,
        constant[t],
        helper[0],
        residual_flag[0],
        cap_flag[0],
        model.requested_residual_cap_numerator,
    )
    circuit.append(compute_gate.inverse(), compute_qubits)

    shock_indices = tuple(circuit.find_bit(qubit).index for qubit in shock)
    threshold_indices = tuple(circuit.find_bit(qubit).index for qubit in threshold)
    work_indices = tuple(
        circuit.find_bit(qubit).index
        for register in work_registers
        for qubit in register
    )
    return ArithmeticAsianOracle(
        circuit=circuit,
        model=model,
        objective_qubit=circuit.find_bit(objective[0]).index,
        shock_qubits=shock_indices,
        threshold_qubits=threshold_indices,
        work_qubits=work_indices,
    )


def _base_adder_counts(width: int) -> PrimitiveCounts:
    return PrimitiveCounts(cx=4 * width, ccx=2 * width)


def _controlled_adder_counts(width: int) -> PrimitiveCounts:
    return PrimitiveCounts(ccx=10 * width)


def _equality_counts(width: int, state: int) -> PrimitiveCounts:
    zeros = width - state.bit_count()
    if width == 1:
        return PrimitiveCounts(x=2 * zeros, cx=1)
    return PrimitiveCounts(
        x=2 * zeros,
        ccx=1 if width == 2 else 2 * width - 3,
    )


def _selected_multiplier_counts(
    model: ArithmeticAsianModel,
    factors: Sequence[int],
    rounding: str,
    *,
    shock_bits: int | None = None,
    product_bits: int | None = None,
) -> PrimitiveCounts:
    counts = PrimitiveCounts(
        x=model.multiplier_fraction_bits if rounding == "ceil" else 0
    )
    q = model.spec.shock_qubits if shock_bits is None else shock_bits
    width = model.product_bits if product_bits is None else product_bits
    for state, factor in enumerate(factors):
        counts += _equality_counts(q, state).scaled(2)
        for bit in range(model.value_bits):
            shifted = factor << bit
            counts += PrimitiveCounts(x=2 * shifted.bit_count(), ccx=2)
            counts += _controlled_adder_counts(width)
    return counts


def _positive_part_counts(
    *, input_bits: int, output_bits: int, arithmetic_bits: int, subtract: int
) -> PrimitiveCounts:
    width = arithmetic_bits + 1
    encoded = (-subtract) % (1 << width)
    return PrimitiveCounts(
        x=2 * encoded.bit_count() + 2,
        cx=2 * input_bits + 8 * width,
        ccx=4 * width + output_bits,
    )


def _constant_multiplier_counts(
    *, input_bits: int, target_bits: int, multiplier: int
) -> PrimitiveCounts:
    counts = PrimitiveCounts()
    for bit in range(input_bits):
        shifted = multiplier << bit
        if shifted >= 1 << target_bits:
            continue
        counts += PrimitiveCounts(x=2 * shifted.bit_count())
        counts += _controlled_adder_counts(target_bits)
    return counts


def _a_qubit_counts(model: ArithmeticAsianModel) -> tuple[int, int]:
    spec = model.spec
    n = spec.n_dates
    q = spec.shock_qubits
    v = model.value_bits
    w = model.product_bits
    g = model.geometric_product_bits
    t = model.total_bits
    m = model.threshold_bits
    collapsed = spec.geometric_leg == "collapsed"
    raw = spec.geometric_leg == "none"
    # The collapsed leg trades n product registers for one product register per
    # bit of the weighted shock sum, plus that sum's own register.  'none'
    # drops every control register: geometric0 (one of the two value-width
    # registers) and the geometric_payoff, scaled_geometric_payoff and
    # residual registers (three of the five total-width ones); its
    # geometric_product_bits and shock_weight_bits are zero.
    chain_steps = len(model.geometric_chain_factors) if collapsed else n
    constant_bits = max(w, g, t + 1, m + 1)
    value_registers = 1 if raw else 2
    total_width_registers = 2 if raw else 5
    work = (
        value_registers * v
        + n * w
        + chain_steps * g
        + model.shock_weight_bits
        + total_width_registers * t
        + (t + 1)
        + max(0, t - v)
        + constant_bits
        + 6
        + max(0, q - 2)
    )
    active = n * q + m + 1 + work
    return active, work


def estimate_arithmetic_asian_resources(
    model: ArithmeticAsianModel,
) -> ArithmeticAsianResourceEstimate:
    """Count a complete ``A`` and Grover iterate ``Q`` compositionally."""

    spec = model.spec
    n = spec.n_dates
    q = spec.shock_qubits
    v = model.value_bits
    t = model.total_bits
    m = model.threshold_bits

    price_multipliers = _selected_multiplier_counts(
        model, model.price_factors, "ceil"
    ).scaled(n)
    collapsed = spec.geometric_leg == "collapsed"
    raw = spec.geometric_leg == "none"
    geometric_weighted_sum = PrimitiveCounts()
    geometric_chain = PrimitiveCounts()
    geometric_multipliers = PrimitiveCounts()
    if collapsed:
        for date in range(n):
            geometric_weighted_sum += _constant_multiplier_counts(
                input_bits=1,
                target_bits=model.shock_weight_bits,
                multiplier=n - date,
            )
        for factor in model.geometric_chain_factors:
            geometric_chain += _selected_multiplier_counts(
                model,
                (model.factor_scale, factor),
                "floor",
                shock_bits=1,
                product_bits=model.geometric_product_bits,
            )
    elif not raw:
        for factors in model.geometric_factors:
            geometric_multipliers += _selected_multiplier_counts(
                model, factors, "floor", product_bits=model.geometric_product_bits
            )
    price_sum = _base_adder_counts(t).scaled(n)
    arithmetic_positive = _positive_part_counts(
        input_bits=t,
        output_bits=t,
        arithmetic_bits=t,
        subtract=n * spec.strike_integer,
    )
    if raw:
        # No geometric payoff, no scaling by n, no residual subtraction: the
        # comparator reads the arithmetic payoff register directly.
        geometric_positive = PrimitiveCounts()
        geometric_scaling = PrimitiveCounts()
        residual_subtraction = PrimitiveCounts()
    else:
        geometric_positive = _positive_part_counts(
            input_bits=v,
            output_bits=v,
            arithmetic_bits=t,
            subtract=spec.strike_integer,
        )
        geometric_scaling = _constant_multiplier_counts(
            input_bits=v, target_bits=t, multiplier=n
        )
        residual_subtraction = PrimitiveCounts(cx=t) + _base_adder_counts(t)
    initialization = PrimitiveCounts(
        x=model.initial_price.bit_count() + model.initial_geometric.bit_count()
    )
    compute = (
        initialization
        + price_multipliers
        + price_sum
        + geometric_weighted_sum
        + geometric_chain
        + geometric_multipliers
        + arithmetic_positive
        + geometric_positive
        + geometric_scaling
        + residual_subtraction
    )
    residual_comparator = PrimitiveCounts(
        cx=2 * m + 8 * (t + 1) + 1,
        ccx=4 * (t + 1),
    )
    if model.requested_residual_cap_numerator < model.normalization_numerator:
        cap_width = m + 1
        encoded_cap = (-model.requested_residual_cap_numerator) % (1 << cap_width)
        cap_comparator = PrimitiveCounts(
            x=2 * encoded_cap.bit_count(),
            cx=2 * m + 8 * cap_width + 1,
            ccx=4 * cap_width,
        )
        threshold = (
            residual_comparator.scaled(2)
            + cap_comparator.scaled(2)
            + PrimitiveCounts(ccx=1)
        )
    else:
        threshold = residual_comparator
    state_preparation = PrimitiveCounts(h=n * q + m)
    a_counts = state_preparation + compute.scaled(2) + threshold

    a_qubits, work_qubits = _a_qubit_counts(model)
    if a_qubits < 3:
        reflection = PrimitiveCounts(x=2 * a_qubits, h=2, cx=1)
        reflection_ancillas = 0
    else:
        reflection = PrimitiveCounts(
            x=2 * a_qubits,
            h=2,
            z=1,
            ccx=2 * a_qubits - 5,
        )
        reflection_ancillas = a_qubits - 3
    q_counts = a_counts.scaled(2) + reflection

    components = {
        "state_preparation_in_A": state_preparation,
        "price_selected_multipliers_in_compute": price_multipliers,
        "arithmetic_price_sum_in_compute": price_sum,
    }
    if collapsed:
        components["geometric_weighted_sum_in_compute"] = geometric_weighted_sum
        components["geometric_exponential_chain_in_compute"] = geometric_chain
    elif not raw:
        components["geometric_selected_multipliers_in_compute"] = geometric_multipliers
    components["arithmetic_positive_part_in_compute"] = arithmetic_positive
    if not raw:
        components["geometric_positive_part_in_compute"] = geometric_positive
        components["geometric_payoff_scaling_in_compute"] = geometric_scaling
        components["residual_subtraction_in_compute"] = residual_subtraction
    components.update(
        {
            "initialization_in_compute": initialization,
            "compute_plus_uncompute_in_A": compute.scaled(2),
            "uniform_threshold_encoder_in_A": threshold,
            "zero_and_objective_reflections_in_Q": reflection,
        }
    )
    return ArithmeticAsianResourceEstimate(
        a_qubits=a_qubits,
        a_work_qubits=work_qubits,
        reflection_clean_ancillas=reflection_ancillas,
        q_qubits_with_clean_reflection_ladder=a_qubits + reflection_ancillas,
        a_counts=a_counts,
        q_counts=q_counts,
        component_counts=components,
    )


def primitive_counts_from_circuit(oracle: ArithmeticAsianOracle) -> PrimitiveCounts:
    """Transpile a tractable executable oracle to the declared primitive basis."""

    primitive = transpile(
        oracle.circuit,
        basis_gates=["h", "x", "cx", "ccx"],
        optimization_level=0,
    )
    operations = primitive.count_ops()
    unexpected = set(operations) - {"h", "x", "cx", "ccx", "barrier"}
    if unexpected:
        raise AssertionError(f"non-arithmetic gates remain: {sorted(unexpected)}")
    return PrimitiveCounts(
        h=int(operations.get("h", 0)),
        x=int(operations.get("x", 0)),
        cx=int(operations.get("cx", 0)),
        ccx=int(operations.get("ccx", 0)),
    )


def arithmetic_objective_probability_from_mps(
    oracle: ArithmeticAsianOracle,
) -> tuple[float, float]:
    """Return objective probability and expected nonzero work Hamming weight.

    The expected Hamming weight upper-bounds the probability that any work
    qubit is nonzero, while requiring only one MPS expectation-value snapshot.
    """

    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qiskit-aer is required for MPS validation") from exc
    circuit = transpile(
        oracle.circuit,
        basis_gates=["h", "x", "cx", "ccx"],
        optimization_level=0,
    )
    from qiskit.quantum_info import SparsePauliOp

    hamming_terms = [("I", [], len(oracle.work_qubits) / 2.0)] + [
        ("Z", [position], -0.5) for position in range(len(oracle.work_qubits))
    ]
    hamming = SparsePauliOp.from_sparse_list(
        hamming_terms, num_qubits=len(oracle.work_qubits)
    )
    circuit.save_expectation_value(
        hamming, list(oracle.work_qubits), label="work_hamming_weight"
    )
    circuit.save_probabilities_dict([oracle.objective_qubit], label="objective")
    result = AerSimulator(method="matrix_product_state").run(circuit, shots=None).result()
    if not result.success:
        raise RuntimeError(f"MPS simulation failed: {result.status}")
    data = result.data(0)
    probability = float(data["objective"].get(1, 0.0))
    leakage = float(data["work_hamming_weight"])
    return probability, max(0.0, leakage)


def arithmetic_roundtrip_leakage_from_mps(oracle: ArithmeticAsianOracle) -> float:
    """Return an upper bound on nonzero probability after ``A A^-1``."""

    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qiskit-aer is required for MPS validation") from exc
    circuit = oracle.circuit.compose(oracle.circuit.inverse())
    circuit = transpile(
        circuit,
        basis_gates=["h", "x", "cx", "ccx"],
        optimization_level=0,
    )
    from qiskit.quantum_info import SparsePauliOp

    hamming_terms = [("I", [], circuit.num_qubits / 2.0)] + [
        ("Z", [position], -0.5) for position in range(circuit.num_qubits)
    ]
    hamming = SparsePauliOp.from_sparse_list(
        hamming_terms, num_qubits=circuit.num_qubits
    )
    circuit.save_expectation_value(
        hamming, list(range(circuit.num_qubits)), label="hamming_weight"
    )
    result = AerSimulator(method="matrix_product_state").run(circuit, shots=None).result()
    if not result.success:
        raise RuntimeError(f"MPS simulation failed: {result.status}")
    return max(0.0, float(result.data(0)["hamming_weight"]))


def iter_reachable_path_values(
    model: ArithmeticAsianModel,
) -> Iterable[tuple[tuple[int, ...], tuple[int, int, int, int]]]:
    """Expose the independent path recurrence for verification scripts."""

    n_states = len(model.spec.shock_points)
    path_count = n_states**model.spec.n_dates
    if path_count > 1_000_000:
        raise ValueError("path iterator is limited to one million paths")
    for digits in itertools.product(range(n_states), repeat=model.spec.n_dates):
        yield digits, _path_values(model, digits)
