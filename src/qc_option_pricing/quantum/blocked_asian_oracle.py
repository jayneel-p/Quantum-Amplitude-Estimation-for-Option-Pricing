"""General equal-block geometric-control oracle for arithmetic Asian calls.

The existing :mod:`telescoping_asian_ladder` module is the frozen two-level
reference implementation used by the paper.  This module provides a separate
single-amplitude oracle for

``H - C_k``,

where ``H`` is the arithmetic-Asian call payoff and ``C_k`` is the call on the
mean of ``k`` equal contiguous-block geometric averages.  The arithmetic path
leg is unchanged as ``k`` varies.  Each added block contributes one collapsed
weighted-sum exponential chain, a term in the block sum, and its registers.

The circuit currently supports a uniform binary shock grid.  Its drift,
volatility, initial price, strike, maturity, date count, fixed-point scales,
and block count are supplied by the caller.  It is an exact reversible oracle
for that directed-rounding finite model, not an exact encoding of continuous
Black--Scholes dynamics.  All circuit operations are H, X, CX, or Toffoli;
logical T counts use the convention Toffoli = 7 T.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING, localcontext
from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder
from scipy.stats import norm, qmc

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    ArithmeticAsianResourceEstimate,
    PrimitiveCounts,
    _append_constant_multiplier,
    _append_positive_part,
    _append_threshold_encoder,
    _base_adder_counts,
    _ceil_product,
    _ceil_stable,
    _constant_multiplier_counts,
    _controlled_adder_counts,
    _equality_counts,
    _positive_part_counts,
    _product_width,
    build_selected_fixed_multiplier_gate,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec
from qc_option_pricing.quantum.telescoping_asian_ladder import (
    BlockGeometricModel,
    ControlPartitionModel,
    _compile_block,
    _encoded_block_value,
)


@dataclass(frozen=True)
class BlockRoundingCertificate:
    """Exhaustive directed-rounding check for one weighted-sum chain."""

    block_index: int
    start_date: int
    stop_date: int
    weighted_sum_values: int
    worst_upward_error_units: float
    worst_downward_error_units: float
    rounds_down_everywhere: bool


@dataclass(frozen=True)
class BlockedAsianSharedModel:
    """Integer dynamics and one ``k``-block control shared by the oracle."""

    spec: AsianGridSpec
    block_count: int
    multiplier_fraction_bits: int
    factor_scale: int
    price_factors: tuple[int, ...]
    initial_price: int
    maximum_prices: tuple[int, ...]
    price_product_bits: int
    value_bits: int
    common_bits: int
    maximum_total: int
    arithmetic_payoff_maximum: int
    partition: ControlPartitionModel
    rounding_certificates: tuple[BlockRoundingCertificate, ...]

    @property
    def dates_per_block(self) -> int:
        return self.spec.n_dates // self.block_count


@dataclass(frozen=True)
class BlockedAsianModel:
    """Compiled model for the nonnegative residual ``H - C_k``."""

    shared: BlockedAsianSharedModel
    requested_cap_numerator: int
    normalization_numerator: int
    threshold_bits: int
    maximum_residual_bound_numerator: int

    @property
    def spec(self) -> AsianGridSpec:
        return self.shared.spec

    @property
    def block_count(self) -> int:
        return self.shared.block_count

    @property
    def normalization_dollars(self) -> float:
        return self.normalization_numerator / (
            self.spec.n_dates * self.spec.price_scale
        )

    @property
    def requested_cap_dollars(self) -> float:
        return self.requested_cap_numerator / (
            self.spec.n_dates * self.spec.price_scale
        )


@dataclass(frozen=True)
class BlockedAsianPathValues:
    """Common-denominator target, control, and residual on one shock path."""

    target: int
    control: int
    residual: int


@dataclass(frozen=True)
class BlockedAsianReference:
    """Exhaustive finite-grid reference for a tractable instance."""

    path_count: int
    target_undiscounted: float
    control_undiscounted: float
    residual_undiscounted: float
    clipped_residual_undiscounted: float
    objective_probability: float
    minimum_residual_numerator: int
    maximum_residual_numerator: int


@dataclass(frozen=True)
class BlockedAsianSample:
    """Vectorized samples of the exact finite-grid integer payoffs."""

    target: np.ndarray
    control: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class BlockControlPriceEstimate:
    """Randomized-QMC estimate of the continuous ``k``-block control price."""

    block_count: int
    replicates: int
    points_per_replicate: int
    seed: int
    undiscounted_mean: float
    discounted_mean: float
    discounted_standard_error: float
    discounted_ci95_low: float
    discounted_ci95_high: float
    replicate_discounted_prices: tuple[float, ...]


@dataclass(frozen=True)
class EncodedBlockControlPriceEstimate:
    """Scrambled-Sobol estimate of the finite binary-grid control price."""

    block_count: int
    replicates: int
    paths_per_replicate: int
    seed: int
    undiscounted_mean: float
    discounted_mean: float
    discounted_standard_error: float
    discounted_ci95_low: float
    discounted_ci95_high: float
    replicate_discounted_prices: tuple[float, ...]
    method: str
    continuous_reference_discounted: float | None
    continuous_reference_standard_error: float | None
    correction_discounted: float | None
    correction_standard_error: float | None


@dataclass(frozen=True)
class BlockedAsianOracle:
    """Executable uniform-threshold ``A`` operator for ``H - C_k``."""

    circuit: QuantumCircuit
    model: BlockedAsianModel
    objective_qubit: int
    shock_qubits: tuple[int, ...]
    threshold_qubits: tuple[int, ...]
    work_qubits: tuple[int, ...]

    def decode_residual(self, objective_probability: float) -> float:
        """Return the discounted residual price represented by the amplitude."""

        if not 0.0 <= objective_probability <= 1.0:
            raise ValueError("objective probability must lie in [0, 1]")
        spec = self.model.spec
        return (
            math.exp(-spec.rate * spec.maturity)
            * self.model.normalization_dollars
            * objective_probability
        )

    def post_process(
        self,
        objective_probability: float,
        *,
        control_undiscounted: float,
    ) -> float:
        """Restore a supplied control expectation and discount the total."""

        if not math.isfinite(control_undiscounted) or control_undiscounted < 0.0:
            raise ValueError("control_undiscounted must be finite and nonnegative")
        spec = self.model.spec
        residual = self.model.normalization_dollars * objective_probability
        return math.exp(-spec.rate * spec.maturity) * (
            control_undiscounted + residual
        )


def _encoded_block_values_vectorized(
    block: BlockGeometricModel,
    weighted_sum: np.ndarray,
    fraction_bits: int,
) -> np.ndarray:
    values = np.full(weighted_sum.size, block.initial_geometric, dtype=np.int64)
    for bit, factor in enumerate(block.chain_factors):
        selected = ((weighted_sum >> bit) & 1).astype(bool)
        values = np.where(
            selected,
            (values * np.int64(factor)) >> fraction_bits,
            values,
        )
    return values


def sample_blocked_payoffs(
    shared: BlockedAsianSharedModel,
    *,
    paths: int,
    seed: int,
    chunk: int = 250_000,
) -> BlockedAsianSample:
    """Sample the oracle's exact integer recurrences without building a circuit."""

    if isinstance(paths, bool) or not isinstance(paths, Integral) or paths < 2:
        raise ValueError("paths must be an integer of at least two")
    if isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    largest_factor = max(shared.price_factors)
    if shared.maximum_prices[-1] * largest_factor >= np.iinfo(np.int64).max:
        raise OverflowError("sampling would overflow int64 price products")
    spec = shared.spec
    factor_scale = np.int64(1 << shared.multiplier_fraction_bits)
    price_factors = np.asarray(shared.price_factors, dtype=np.int64)
    targets = np.empty(paths, dtype=np.int64)
    controls = np.empty(paths, dtype=np.int64)
    residuals = np.empty(paths, dtype=np.int64)
    rng = np.random.default_rng(seed)
    done = 0
    while done < paths:
        size = min(chunk, paths - done)
        price = np.full(size, shared.initial_price, dtype=np.int64)
        total = np.zeros(size, dtype=np.int64)
        weighted = [
            np.zeros(size, dtype=np.int64) for _ in shared.partition.blocks
        ]
        for date in range(spec.n_dates):
            digits = rng.integers(0, 2, size=size, dtype=np.int8)
            price = (
                price * np.take(price_factors, digits) + factor_scale - 1
            ) >> shared.multiplier_fraction_bits
            total += price
            digits64 = digits.astype(np.int64)
            for accumulator, block in zip(weighted, shared.partition.blocks):
                weight = block.shock_weights[date]
                if weight:
                    accumulator += digits64 * np.int64(weight)
        block_sum = np.zeros(size, dtype=np.int64)
        for block, accumulator in zip(shared.partition.blocks, weighted):
            block_sum += _encoded_block_values_vectorized(
                block,
                accumulator,
                shared.multiplier_fraction_bits,
            )
        target = np.maximum(total - spec.n_dates * spec.strike_integer, 0)
        control = (spec.n_dates // shared.block_count) * np.maximum(
            block_sum - shared.block_count * spec.strike_integer,
            0,
        )
        residual = target - control
        if int(residual.min()) < 0:
            raise AssertionError(
                "the sampled blocked control exceeded the arithmetic payoff"
            )
        targets[done : done + size] = target
        controls[done : done + size] = control
        residuals[done : done + size] = residual
        done += size
    return BlockedAsianSample(target=targets, control=controls, residual=residuals)


def sample_blocked_residuals(
    shared: BlockedAsianSharedModel,
    *,
    paths: int,
    seed: int,
    chunk: int = 250_000,
) -> np.ndarray:
    """Return only the residual arm from :func:`sample_blocked_payoffs`."""

    return sample_blocked_payoffs(
        shared,
        paths=paths,
        seed=seed,
        chunk=chunk,
    ).residual


def black_scholes_binary_spec(
    *,
    n_dates: int,
    s0: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    price_scale: int = 16_384,
    shock_points: tuple[float, float] = (-1.0, 1.0),
) -> AsianGridSpec:
    """Create the finite binary-shock Black--Scholes specification used here."""

    return AsianGridSpec(
        n_dates=n_dates,
        shock_points=shock_points,
        shock_probabilities=(0.5, 0.5),
        s0=s0,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        shock_scale=1,
        price_scale=price_scale,
        geometric_leg="collapsed",
    )


def _validate_inputs(
    spec: AsianGridSpec,
    block_count: int,
    multiplier_fraction_bits: int,
) -> None:
    if spec.shock_qubits != 1 or len(spec.shock_points) != 2:
        raise ValueError("the blocked oracle requires one binary shock qubit per date")
    if not all(
        math.isclose(probability, 0.5, rel_tol=0.0, abs_tol=1e-14)
        for probability in spec.shock_probabilities
    ):
        raise ValueError("the blocked oracle requires uniform binary shocks")
    if not spec.shock_points[1] > spec.shock_points[0]:
        raise ValueError("binary shock points must be strictly increasing")
    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, Integral)
        or block_count < 1
        or block_count > spec.n_dates
    ):
        raise ValueError("block_count must be an integer in [1, n_dates]")
    if spec.n_dates % block_count:
        raise ValueError("block_count must divide n_dates for equal blocks")
    if (
        isinstance(multiplier_fraction_bits, bool)
        or not isinstance(multiplier_fraction_bits, Integral)
        or multiplier_fraction_bits < 1
    ):
        raise ValueError("multiplier_fraction_bits must be a positive integer")


def _certify_block_rounding(
    *,
    spec: AsianGridSpec,
    block: BlockGeometricModel,
    multiplier_fraction_bits: int,
) -> BlockRoundingCertificate:
    """Compare every reachable weighted sum with an 80-digit reference."""

    n = spec.n_dates
    m = block.stop_date - block.start_date
    low, high = spec.shock_points
    worst_upward = Decimal("-Infinity")
    worst_downward = Decimal(0)
    with localcontext() as context:
        context.prec = 80
        d = lambda value: Decimal(str(value))
        dt = d(spec.maturity) / Decimal(n)
        drift = d(spec.rate) - d(spec.volatility) ** 2 / Decimal(2)
        diffusion = d(spec.volatility) * dt.sqrt()
        average_fixing_index = Decimal(
            block.start_date + block.stop_date + 1
        ) / Decimal(2)
        for weighted_sum in range(block.shock_weight_sum + 1):
            exponent = (
                drift * dt * average_fixing_index
                + diffusion
                / Decimal(m)
                * (
                    d(low) * Decimal(block.shock_weight_sum)
                    + d(high - low) * Decimal(weighted_sum)
                )
            )
            exact = d(spec.s0) * exponent.exp() * Decimal(spec.price_scale)
            encoded = Decimal(
                _encoded_block_value(
                    block, weighted_sum, multiplier_fraction_bits
                )
            )
            worst_upward = max(worst_upward, encoded - exact)
            worst_downward = max(worst_downward, exact - encoded)
    return BlockRoundingCertificate(
        block_index=block.block_index,
        start_date=block.start_date,
        stop_date=block.stop_date,
        weighted_sum_values=block.shock_weight_sum + 1,
        worst_upward_error_units=float(worst_upward),
        worst_downward_error_units=float(worst_downward),
        rounds_down_everywhere=bool(worst_upward <= 0),
    )


@lru_cache(maxsize=None)
def _build_shared_model(
    spec: AsianGridSpec,
    block_count: int,
    multiplier_fraction_bits: int,
) -> BlockedAsianSharedModel:
    _validate_inputs(spec, block_count, multiplier_fraction_bits)
    n = spec.n_dates
    factor_scale = 1 << multiplier_fraction_bits
    dt = spec.maturity / n
    drift = spec.rate - 0.5 * spec.volatility**2
    diffusion = spec.volatility * math.sqrt(dt)
    price_factors = tuple(
        _ceil_stable(math.exp(drift * dt + diffusion * shock) * factor_scale)
        for shock in spec.shock_points
    )
    if min(price_factors) <= 0:
        raise ValueError("multiplier precision rounds a price factor to zero")
    initial_price = _ceil_stable(spec.s0 * spec.price_scale)
    current_maximum = initial_price
    maximum_prices: list[int] = []
    for _ in range(n):
        current_maximum = max(
            _ceil_product(current_maximum, factor, multiplier_fraction_bits)
            for factor in price_factors
        )
        maximum_prices.append(current_maximum)
    maximum_total = sum(maximum_prices)
    arithmetic_maximum = max(0, maximum_total - n * spec.strike_integer)
    if arithmetic_maximum == 0:
        raise ValueError("the arithmetic payoff is identically zero")

    provisional_blocks = tuple(
        _compile_block(
            spec=spec,
            multiplier_fraction_bits=multiplier_fraction_bits,
            block_count=block_count,
            block_index=block_index,
        )
        for block_index in range(block_count)
    )
    certificates = tuple(
        _certify_block_rounding(
            spec=spec,
            block=block,
            multiplier_fraction_bits=multiplier_fraction_bits,
        )
        for block in provisional_blocks
    )
    if not all(certificate.rounds_down_everywhere for certificate in certificates):
        raise ValueError("a block exponential chain does not round downward everywhere")
    provisional_blocks = tuple(
        replace(
            block,
            rounding_error_bound_units=max(
                0,
                int(
                    Decimal(str(certificate.worst_downward_error_units)).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                ),
            ),
        )
        for block, certificate in zip(provisional_blocks, certificates)
    )

    maximum_value = max(
        initial_price,
        *maximum_prices,
        *(block.maximum_geometric for block in provisional_blocks),
    )
    value_bits = max(1, maximum_value.bit_length())
    price_product_bits = _product_width(
        value_bits=value_bits,
        fraction_bits=multiplier_fraction_bits,
        maximum_value=maximum_value,
        factors=price_factors,
    )
    blocks = tuple(
        replace(
            block,
            product_bits=_product_width(
                value_bits=value_bits,
                fraction_bits=multiplier_fraction_bits,
                maximum_value=maximum_value,
                factors=block.chain_factors,
            ),
        )
        for block in provisional_blocks
    )
    maximum_block_sum = sum(block.maximum_geometric for block in blocks)
    maximum_control = (n // block_count) * max(
        maximum_block_sum - block_count * spec.strike_integer,
        0,
    )
    common_bits = max(
        1,
        max(
            maximum_total,
            arithmetic_maximum,
            maximum_block_sum,
            maximum_control,
        ).bit_length(),
    )
    partition = ControlPartitionModel(
        block_count=block_count,
        blocks=blocks,
        strike_adjustment_units=0,
        maximum_block_sum=maximum_block_sum,
        maximum_common_numerator=maximum_control,
    )
    return BlockedAsianSharedModel(
        spec=spec,
        block_count=block_count,
        multiplier_fraction_bits=multiplier_fraction_bits,
        factor_scale=factor_scale,
        price_factors=price_factors,
        initial_price=initial_price,
        maximum_prices=tuple(maximum_prices),
        price_product_bits=price_product_bits,
        value_bits=value_bits,
        common_bits=common_bits,
        maximum_total=maximum_total,
        arithmetic_payoff_maximum=arithmetic_maximum,
        partition=partition,
        rounding_certificates=certificates,
    )


def build_blocked_asian_model(
    spec: AsianGridSpec,
    block_count: int,
    *,
    multiplier_fraction_bits: int = 12,
    residual_cap_dollars: float | None = None,
) -> BlockedAsianModel:
    """Compile a general ``H - C_k`` model from a finite BS specification."""

    shared = _build_shared_model(spec, block_count, multiplier_fraction_bits)
    maximum_bound = shared.arithmetic_payoff_maximum
    if residual_cap_dollars is None:
        requested_cap = maximum_bound
    else:
        if not math.isfinite(residual_cap_dollars) or residual_cap_dollars <= 0.0:
            raise ValueError("residual_cap_dollars must be finite and positive")
        requested_cap = int(
            round(residual_cap_dollars * spec.n_dates * spec.price_scale)
        )
        if requested_cap < 1:
            raise ValueError("residual cap is below one encoded unit")
    threshold_bits = max(1, (requested_cap - 1).bit_length())
    return BlockedAsianModel(
        shared=shared,
        requested_cap_numerator=requested_cap,
        normalization_numerator=1 << threshold_bits,
        threshold_bits=threshold_bits,
        maximum_residual_bound_numerator=maximum_bound,
    )


def build_black_scholes_blocked_model(
    *,
    n_dates: int,
    block_count: int,
    s0: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    price_scale: int = 16_384,
    multiplier_fraction_bits: int = 30,
    residual_cap_dollars: float | None = None,
    shock_points: tuple[float, float] = (-1.0, 1.0),
) -> BlockedAsianModel:
    """Convenience interface accepting Black--Scholes and encoding inputs."""

    spec = black_scholes_binary_spec(
        n_dates=n_dates,
        s0=s0,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        price_scale=price_scale,
        shock_points=shock_points,
    )
    return build_blocked_asian_model(
        spec,
        block_count,
        multiplier_fraction_bits=multiplier_fraction_bits,
        residual_cap_dollars=residual_cap_dollars,
    )


def blocked_asian_path_values(
    shared: BlockedAsianSharedModel,
    digits: Sequence[int],
) -> BlockedAsianPathValues:
    """Evaluate the same directed-rounding recurrences as the circuit."""

    spec = shared.spec
    if len(digits) != spec.n_dates or any(digit not in (0, 1) for digit in digits):
        raise ValueError("digits must contain one binary shock per fixing date")
    price = shared.initial_price
    total = 0
    for digit in digits:
        price = _ceil_product(
            price,
            shared.price_factors[digit],
            shared.multiplier_fraction_bits,
        )
        total += price
    target = max(total - spec.n_dates * spec.strike_integer, 0)
    block_sum = 0
    for block in shared.partition.blocks:
        weighted_sum = sum(
            weight * digit for weight, digit in zip(block.shock_weights, digits)
        )
        block_sum += _encoded_block_value(
            block, weighted_sum, shared.multiplier_fraction_bits
        )
    control = (spec.n_dates // shared.block_count) * max(
        block_sum - shared.block_count * spec.strike_integer,
        0,
    )
    residual = target - control
    if residual < 0:
        raise AssertionError("the encoded blocked control exceeded the arithmetic payoff")
    return BlockedAsianPathValues(target=target, control=control, residual=residual)


def blocked_asian_payoffs_from_digits(
    shared: BlockedAsianSharedModel,
    digits: np.ndarray,
) -> BlockedAsianSample:
    """Vectorize the exact integer recurrences over an explicit bit matrix."""

    array = np.asarray(digits)
    if array.ndim != 2 or array.shape[1] != shared.spec.n_dates:
        raise ValueError("digits must have shape (paths, n_dates)")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("digits must contain only zero and one")
    array = array.astype(np.int64, copy=False)
    paths = array.shape[0]
    if paths < 1:
        raise ValueError("digits must contain at least one path")
    spec = shared.spec
    factor_scale = np.int64(1 << shared.multiplier_fraction_bits)
    factors = np.asarray(shared.price_factors, dtype=np.int64)
    price = np.full(paths, shared.initial_price, dtype=np.int64)
    total = np.zeros(paths, dtype=np.int64)
    for date in range(spec.n_dates):
        price = (
            price * np.take(factors, array[:, date]) + factor_scale - 1
        ) >> shared.multiplier_fraction_bits
        total += price
    block_sum = np.zeros(paths, dtype=np.int64)
    for block in shared.partition.blocks:
        weights = np.asarray(block.shock_weights, dtype=np.int64)
        weighted_sum = array @ weights
        block_sum += _encoded_block_values_vectorized(
            block,
            weighted_sum,
            shared.multiplier_fraction_bits,
        )
    target = np.maximum(total - spec.n_dates * spec.strike_integer, 0)
    control = (spec.n_dates // shared.block_count) * np.maximum(
        block_sum - shared.block_count * spec.strike_integer,
        0,
    )
    residual = target - control
    if int(residual.min()) < 0:
        raise AssertionError("the blocked control exceeded the arithmetic payoff")
    return BlockedAsianSample(target=target, control=control, residual=residual)


def enumerate_blocked_asian(model: BlockedAsianModel) -> BlockedAsianReference:
    """Exhaustively enumerate a small finite-grid residual instance."""

    path_count = 1 << model.spec.n_dates
    if path_count > 1_000_000:
        raise ValueError("blocked-oracle enumeration is limited to one million paths")
    target_total = 0
    control_total = 0
    residual_total = 0
    clipped_total = 0
    minimum = math.inf
    maximum = 0
    for digits in itertools.product(range(2), repeat=model.spec.n_dates):
        values = blocked_asian_path_values(model.shared, digits)
        target_total += values.target
        control_total += values.control
        residual_total += values.residual
        clipped_total += min(values.residual, model.requested_cap_numerator)
        minimum = min(minimum, values.residual)
        maximum = max(maximum, values.residual)
    denominator = path_count * model.spec.n_dates * model.spec.price_scale
    return BlockedAsianReference(
        path_count=path_count,
        target_undiscounted=target_total / denominator,
        control_undiscounted=control_total / denominator,
        residual_undiscounted=residual_total / denominator,
        clipped_residual_undiscounted=clipped_total / denominator,
        objective_probability=clipped_total
        / (path_count * model.normalization_numerator),
        minimum_residual_numerator=int(minimum),
        maximum_residual_numerator=maximum,
    )


def _selected_multiplier_counts(
    *,
    value_bits: int,
    shock_bits: int,
    product_bits: int,
    fraction_bits: int,
    factors: Sequence[int],
    rounding: str,
) -> PrimitiveCounts:
    counts = PrimitiveCounts(x=fraction_bits if rounding == "ceil" else 0)
    for state, factor in enumerate(factors):
        counts += _equality_counts(shock_bits, state).scaled(2)
        for bit in range(value_bits):
            shifted = factor << bit
            if shifted >= 1 << product_bits:
                raise ValueError("shifted multiplier does not fit product register")
            counts += PrimitiveCounts(x=2 * shifted.bit_count(), ccx=2)
            counts += _controlled_adder_counts(product_bits)
    return counts


def _control_partition_counts(
    shared: BlockedAsianSharedModel,
) -> tuple[PrimitiveCounts, dict[str, PrimitiveCounts]]:
    partition = shared.partition
    weighted = PrimitiveCounts()
    chains = PrimitiveCounts()
    for block in partition.blocks:
        for weight in block.shock_weights:
            if weight:
                weighted += _constant_multiplier_counts(
                    input_bits=1,
                    target_bits=block.shock_weight_bits,
                    multiplier=weight,
                )
        for factor in block.chain_factors:
            chains += _selected_multiplier_counts(
                value_bits=shared.value_bits,
                shock_bits=1,
                product_bits=block.product_bits,
                fraction_bits=shared.multiplier_fraction_bits,
                factors=(shared.factor_scale, factor),
                rounding="floor",
            )
    summation = _base_adder_counts(shared.common_bits).scaled(partition.block_count)
    positive = _positive_part_counts(
        input_bits=shared.common_bits,
        output_bits=shared.common_bits,
        arithmetic_bits=shared.common_bits,
        subtract=partition.block_count * shared.spec.strike_integer,
    )
    scaling = _constant_multiplier_counts(
        input_bits=shared.common_bits,
        target_bits=shared.common_bits,
        multiplier=shared.spec.n_dates // partition.block_count,
    )
    prefix = f"p{partition.block_count}"
    components = {
        f"{prefix}_weighted_sums_in_compute": weighted,
        f"{prefix}_exponential_chains_in_compute": chains,
        f"{prefix}_block_sum_in_compute": summation,
        f"{prefix}_positive_part_in_compute": positive,
        f"{prefix}_common_scaling_in_compute": scaling,
    }
    return weighted + chains + summation + positive + scaling, components


def _a_qubits(model: BlockedAsianModel) -> tuple[int, int]:
    shared = model.shared
    spec = shared.spec
    n = spec.n_dates
    v = shared.value_bits
    c = shared.common_bits
    work = v + n * shared.price_product_bits + 2 * c
    for block in shared.partition.blocks:
        work += (
            v
            + block.shock_weight_bits
            + len(block.chain_factors) * block.product_bits
        )
    work += 3 * c
    max_product_bits = max(
        shared.price_product_bits,
        *(block.product_bits for block in shared.partition.blocks),
    )
    constant_bits = max(max_product_bits, c + 1, model.threshold_bits + 1)
    work += (
        c
        + (c + 1)
        + max(0, c - v)
        + constant_bits
        + 6
        + max(0, spec.shock_qubits - 2)
    )
    active = n * spec.shock_qubits + model.threshold_bits + 1 + work
    return active, work


def estimate_blocked_asian_resources(
    model: BlockedAsianModel,
) -> ArithmeticAsianResourceEstimate:
    """Count one general blocked ``A`` and one Grover iterate compositionally."""

    shared = model.shared
    spec = shared.spec
    n = spec.n_dates
    c = shared.common_bits
    m = model.threshold_bits
    initialization = PrimitiveCounts(x=shared.initial_price.bit_count())
    initialization += PrimitiveCounts(
        x=sum(block.initial_geometric.bit_count() for block in shared.partition.blocks)
    )
    price_multipliers = _selected_multiplier_counts(
        value_bits=shared.value_bits,
        shock_bits=spec.shock_qubits,
        product_bits=shared.price_product_bits,
        fraction_bits=shared.multiplier_fraction_bits,
        factors=shared.price_factors,
        rounding="ceil",
    ).scaled(n)
    price_sum = _base_adder_counts(c).scaled(n)
    arithmetic_positive = _positive_part_counts(
        input_bits=c,
        output_bits=c,
        arithmetic_bits=c,
        subtract=n * spec.strike_integer,
    )
    control, control_components = _control_partition_counts(shared)
    subtraction = PrimitiveCounts(cx=c) + _base_adder_counts(c)
    compute = (
        initialization
        + price_multipliers
        + price_sum
        + arithmetic_positive
        + control
        + subtraction
    )
    components = {
        "price_selected_multipliers_in_compute": price_multipliers,
        "arithmetic_price_sum_in_compute": price_sum,
        "arithmetic_positive_part_in_compute": arithmetic_positive,
        "initialization_in_compute": initialization,
        "residual_subtraction_in_compute": subtraction,
        **control_components,
    }
    residual_comparator = PrimitiveCounts(
        cx=2 * m + 8 * (c + 1) + 1,
        ccx=4 * (c + 1),
    )
    if model.requested_cap_numerator < model.normalization_numerator:
        cap_width = m + 1
        encoded_cap = (-model.requested_cap_numerator) % (1 << cap_width)
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
    state_preparation = PrimitiveCounts(h=n * spec.shock_qubits + m)
    a_counts = state_preparation + compute.scaled(2) + threshold
    a_qubits, work_qubits = _a_qubits(model)
    reflection = PrimitiveCounts(
        x=2 * a_qubits,
        h=2,
        z=1,
        ccx=2 * a_qubits - 5,
    )
    q_counts = a_counts.scaled(2) + reflection
    components.update(
        {
            "state_preparation_in_A": state_preparation,
            "compute_plus_uncompute_in_A": compute.scaled(2),
            "uniform_threshold_encoder_in_A": threshold,
            "zero_and_objective_reflections_in_Q": reflection,
        }
    )
    return ArithmeticAsianResourceEstimate(
        a_qubits=a_qubits,
        a_work_qubits=work_qubits,
        reflection_clean_ancillas=a_qubits - 3,
        q_qubits_with_clean_reflection_ladder=2 * a_qubits - 3,
        a_counts=a_counts,
        q_counts=q_counts,
        component_counts=components,
    )


def build_blocked_asian_oracle(model: BlockedAsianModel) -> BlockedAsianOracle:
    """Build an executable H/X/CX/Toffoli oracle for ``H - C_k``."""

    shared = model.shared
    spec = shared.spec
    partition = shared.partition
    n = spec.n_dates
    q = spec.shock_qubits
    v = shared.value_bits
    c = shared.common_bits
    m = model.threshold_bits
    max_product_bits = max(
        shared.price_product_bits,
        *(block.product_bits for block in partition.blocks),
    )
    constant_bits = max(max_product_bits, c + 1, m + 1)

    shock = QuantumRegister(n * q, "shock")
    threshold = QuantumRegister(m, "threshold")
    objective = QuantumRegister(1, "objective")
    price0 = QuantumRegister(v, "price0")
    price_products = QuantumRegister(n * shared.price_product_bits, "price_products")
    total = QuantumRegister(c, "total")
    arithmetic_payoff = QuantumRegister(c, "arithmetic_payoff")
    block_registers = []
    for block in partition.blocks:
        block_registers.append(
            {
                "initial": QuantumRegister(v, f"b{block.block_index}_g0"),
                "weight": QuantumRegister(
                    block.shock_weight_bits, f"b{block.block_index}_weight"
                ),
                "products": QuantumRegister(
                    len(block.chain_factors) * block.product_bits,
                    f"b{block.block_index}_products",
                ),
            }
        )
    control_sum = QuantumRegister(c, "control_sum")
    control_payoff = QuantumRegister(c, "control_payoff")
    control_scaled = QuantumRegister(c, "control_scaled")
    residual = QuantumRegister(c, "residual")
    scratch = QuantumRegister(c + 1, "scratch")
    pad = QuantumRegister(c - v, "pad") if c > v else None
    constant = QuantumRegister(constant_bits, "constant")
    helper = QuantumRegister(1, "helper")
    c3temp = QuantumRegister(1, "c3temp")
    equality = QuantumRegister(1, "equality")
    equality_work = QuantumRegister(q - 2, "equality_work") if q > 2 else None
    term = QuantumRegister(1, "term")
    residual_flag = QuantumRegister(1, "residual_flag")
    cap_flag = QuantumRegister(1, "cap_flag")

    work_registers: list[QuantumRegister] = [
        price0,
        price_products,
        total,
        arithmetic_payoff,
    ]
    for registers in block_registers:
        work_registers.extend(
            [registers["initial"], registers["weight"], registers["products"]]
        )
    work_registers.extend(
        [control_sum, control_payoff, control_scaled, residual, scratch]
    )
    if pad is not None:
        work_registers.append(pad)
    work_registers.extend([constant, helper, c3temp, equality])
    if equality_work is not None:
        work_registers.append(equality_work)
    work_registers.extend([term, residual_flag, cap_flag])

    circuit = QuantumCircuit(
        shock,
        threshold,
        objective,
        *work_registers,
        name=f"AsianBlockedK{partition.block_count}-A",
    )
    compute = QuantumCircuit(shock, *work_registers, name="compute-blocked-residual")
    for bit in range(v):
        if (shared.initial_price >> bit) & 1:
            compute.x(price0[bit])
    for block, registers in zip(partition.blocks, block_registers):
        for bit in range(v):
            if (block.initial_geometric >> bit) & 1:
                compute.x(registers["initial"][bit])

    pad_qubits = [] if pad is None else list(pad)
    multiplier_work = [helper[0], c3temp[0], equality[0]]
    if equality_work is not None:
        multiplier_work.extend(equality_work)
    multiplier_work.append(term[0])

    current_price: Sequence = list(price0)
    total_adder = CDKMRippleCarryAdder(c, kind="fixed").to_gate()
    for date in range(n):
        width = shared.price_product_bits
        product = list(price_products[date * width : (date + 1) * width])
        shock_slice = list(shock[date * q : (date + 1) * q])
        multiplier = build_selected_fixed_multiplier_gate(
            value_bits=v,
            shock_bits=q,
            product_bits=width,
            fraction_bits=shared.multiplier_fraction_bits,
            factors=shared.price_factors,
            rounding="ceil",
        )
        compute.append(
            multiplier,
            [
                *current_price,
                *shock_slice,
                *product,
                *constant[:width],
                *multiplier_work,
            ],
        )
        current_price = product[
            shared.multiplier_fraction_bits : shared.multiplier_fraction_bits + v
        ]
        compute.append(
            total_adder,
            [*current_price, *pad_qubits, *total, helper[0]],
        )
    _append_positive_part(
        compute,
        total,
        arithmetic_payoff,
        n * spec.strike_integer,
        scratch,
        constant,
        helper[0],
    )

    control_sum_adder = CDKMRippleCarryAdder(c, kind="fixed").to_gate()
    for block, registers in zip(partition.blocks, block_registers):
        weight_register = registers["weight"]
        for date, weight in enumerate(block.shock_weights):
            if weight:
                _append_constant_multiplier(
                    compute,
                    [shock[date]],
                    weight,
                    weight_register,
                    constant,
                    helper[0],
                    c3temp[0],
                )
        current_geometric: Sequence = list(registers["initial"])
        products = registers["products"]
        chain_work = [helper[0], c3temp[0], equality[0], term[0]]
        for bit, factor in enumerate(block.chain_factors):
            width = block.product_bits
            product = list(products[bit * width : (bit + 1) * width])
            multiplier = build_selected_fixed_multiplier_gate(
                value_bits=v,
                shock_bits=1,
                product_bits=width,
                fraction_bits=shared.multiplier_fraction_bits,
                factors=(shared.factor_scale, factor),
                rounding="floor",
            )
            compute.append(
                multiplier,
                [
                    *current_geometric,
                    weight_register[bit],
                    *product,
                    *constant[:width],
                    *chain_work,
                ],
            )
            current_geometric = product[
                shared.multiplier_fraction_bits : shared.multiplier_fraction_bits + v
            ]
        compute.append(
            control_sum_adder,
            [*current_geometric, *pad_qubits, *control_sum, helper[0]],
        )
    _append_positive_part(
        compute,
        control_sum,
        control_payoff,
        partition.block_count * spec.strike_integer,
        scratch,
        constant,
        helper[0],
    )
    _append_constant_multiplier(
        compute,
        control_payoff,
        n // partition.block_count,
        control_scaled,
        constant,
        helper[0],
        c3temp[0],
    )
    compute.cx(arithmetic_payoff, residual)
    subtractor = CDKMRippleCarryAdder(c, kind="fixed").to_gate().inverse()
    compute.append(subtractor, [*control_scaled, *residual, helper[0]])

    circuit.h(shock)
    circuit.h(threshold)
    compute_gate = compute.to_gate(label="compute-blocked-residual")
    compute_qubits = [
        *shock,
        *(qubit for register in work_registers for qubit in register),
    ]
    circuit.append(compute_gate, compute_qubits)
    _append_threshold_encoder(
        circuit,
        threshold,
        residual,
        objective[0],
        scratch,
        constant,
        constant[c],
        helper[0],
        residual_flag[0],
        cap_flag[0],
        model.requested_cap_numerator,
    )
    circuit.append(compute_gate.inverse(), compute_qubits)
    return BlockedAsianOracle(
        circuit=circuit,
        model=model,
        objective_qubit=circuit.find_bit(objective[0]).index,
        shock_qubits=tuple(circuit.find_bit(qubit).index for qubit in shock),
        threshold_qubits=tuple(circuit.find_bit(qubit).index for qubit in threshold),
        work_qubits=tuple(
            circuit.find_bit(qubit).index
            for register in work_registers
            for qubit in register
        ),
    )


def primitive_counts_from_blocked_asian_circuit(
    oracle: BlockedAsianOracle,
) -> PrimitiveCounts:
    """Transpile a materialized circuit to the logical counting basis."""

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


def _block_log_gaussian_parameters(
    spec: AsianGridSpec,
    block_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the continuous-BS mean vector and covariance of ``log G_j``."""

    if spec.n_dates % block_count:
        raise ValueError("block_count must divide n_dates")
    m = spec.n_dates // block_count
    times = np.arange(1, spec.n_dates + 1, dtype=float) * (
        spec.maturity / spec.n_dates
    )
    blocks = [times[index * m : (index + 1) * m] for index in range(block_count)]
    drift = spec.rate - 0.5 * spec.volatility**2
    means = np.asarray(
        [math.log(spec.s0) + drift * float(block.mean()) for block in blocks]
    )
    covariance = np.empty((block_count, block_count), dtype=float)
    for row, first in enumerate(blocks):
        for column, second in enumerate(blocks):
            covariance[row, column] = spec.volatility**2 * float(
                np.minimum(first[:, None], second[None, :]).mean()
            )
    return means, covariance


def estimate_encoded_block_control_price_rqmc(
    shared: BlockedAsianSharedModel,
    *,
    log2_paths: int = 14,
    replicates: int = 16,
    seed: int = 20_260_902,
    continuous_reference: BlockControlPriceEstimate | None = None,
) -> EncodedBlockControlPriceEstimate:
    """Estimate the exact finite-grid control with scrambled Sobol' bit paths.

    The uncertainty is the standard error across independent scrambles.  This
    is an empirical randomized-QMC error estimate, not a deterministic bound.
    """

    if (
        isinstance(log2_paths, bool)
        or not isinstance(log2_paths, Integral)
        or log2_paths < 1
    ):
        raise ValueError("log2_paths must be a positive integer")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, Integral)
        or replicates < 2
    ):
        raise ValueError("replicates must be an integer of at least two")
    spec = shared.spec
    denominator = spec.n_dates * spec.price_scale
    discount = math.exp(-spec.rate * spec.maturity)
    prices = []
    corrections = []
    dt = spec.maturity / spec.n_dates
    drift_step = (spec.rate - 0.5 * spec.volatility**2) * dt
    diffusion_step = spec.volatility * math.sqrt(dt)
    dates_per_block = spec.n_dates // shared.block_count
    tiny = np.finfo(float).eps
    for replicate in range(replicates):
        sampler = qmc.Sobol(
            d=spec.n_dates,
            scramble=True,
            seed=seed + replicate,
        )
        uniforms = sampler.random_base2(log2_paths)
        digits = (uniforms >= 0.5).astype(np.int8)
        payoffs = blocked_asian_payoffs_from_digits(shared, digits)
        encoded_discounted = discount * float(payoffs.control.mean()) / denominator
        if continuous_reference is None:
            prices.append(encoded_discounted)
        else:
            normals = norm.ppf(np.clip(uniforms, tiny, 1.0 - tiny))
            log_prices = math.log(spec.s0) + np.cumsum(
                drift_step + diffusion_step * normals,
                axis=1,
            )
            block_logs = log_prices.reshape(
                log_prices.shape[0],
                shared.block_count,
                dates_per_block,
            ).mean(axis=2)
            continuous_control = np.exp(block_logs).mean(axis=1)
            continuous_payoff = np.maximum(continuous_control - spec.strike, 0.0)
            continuous_discounted = discount * float(continuous_payoff.mean())
            corrections.append(encoded_discounted - continuous_discounted)
    if continuous_reference is None:
        values = np.asarray(prices)
        estimate = float(values.mean())
        standard_error = float(values.std(ddof=1) / math.sqrt(replicates))
        method = "direct scrambled Sobol' over binary shock paths"
        reference_mean = None
        reference_standard_error = None
        correction_mean = None
        correction_standard_error = None
    else:
        correction_values = np.asarray(corrections)
        correction_mean = float(correction_values.mean())
        correction_standard_error = float(
            correction_values.std(ddof=1) / math.sqrt(replicates)
        )
        reference_mean = continuous_reference.discounted_mean
        reference_standard_error = continuous_reference.discounted_standard_error
        estimate = reference_mean + correction_mean
        standard_error = math.hypot(
            reference_standard_error,
            correction_standard_error,
        )
        values = reference_mean + correction_values
        method = (
            "coupled encoded-minus-continuous correction plus an independent "
            "k-dimensional continuous-control reference"
        )
    return EncodedBlockControlPriceEstimate(
        block_count=shared.block_count,
        replicates=replicates,
        paths_per_replicate=1 << log2_paths,
        seed=seed,
        undiscounted_mean=estimate / discount,
        discounted_mean=estimate,
        discounted_standard_error=standard_error,
        discounted_ci95_low=estimate - 1.96 * standard_error,
        discounted_ci95_high=estimate + 1.96 * standard_error,
        replicate_discounted_prices=tuple(float(value) for value in values),
        method=method,
        continuous_reference_discounted=reference_mean,
        continuous_reference_standard_error=reference_standard_error,
        correction_discounted=correction_mean,
        correction_standard_error=correction_standard_error,
    )


def estimate_block_control_price_rqmc(
    spec: AsianGridSpec,
    block_count: int,
    *,
    log2_points: int = 16,
    replicates: int = 16,
    seed: int = 20_260_802,
) -> BlockControlPriceEstimate:
    """Price ``C_k`` by independently scrambled Sobol' Gaussian integration.

    The reported uncertainty is the standard error across independent
    randomizations.  It is an empirical numerical error estimate, not a
    deterministic quadrature certificate.
    """

    _validate_inputs(spec, block_count, 1)
    if (
        isinstance(log2_points, bool)
        or not isinstance(log2_points, Integral)
        or log2_points < 1
    ):
        raise ValueError("log2_points must be a positive integer")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, Integral)
        or replicates < 2
    ):
        raise ValueError("replicates must be an integer of at least two")
    discount = math.exp(-spec.rate * spec.maturity)
    means, covariance = _block_log_gaussian_parameters(spec, block_count)
    if spec.volatility == 0.0:
        deterministic = discount * max(
            float(np.exp(means).mean()) - spec.strike,
            0.0,
        )
        values = np.full(replicates, deterministic)
    else:
        cholesky = np.linalg.cholesky(covariance)
        prices = []
        tiny = np.finfo(float).eps
        for replicate in range(replicates):
            sampler = qmc.Sobol(d=block_count, scramble=True, seed=seed + replicate)
            uniforms = sampler.random_base2(log2_points)
            normals = norm.ppf(np.clip(uniforms, tiny, 1.0 - tiny))
            log_geometric = means + normals @ cholesky.T
            block_average = np.exp(log_geometric).mean(axis=1)
            payoff = np.maximum(block_average - spec.strike, 0.0)
            prices.append(discount * float(payoff.mean()))
        values = np.asarray(prices)
    estimate = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(replicates))
    return BlockControlPriceEstimate(
        block_count=block_count,
        replicates=replicates,
        points_per_replicate=1 << log2_points,
        seed=seed,
        undiscounted_mean=estimate / discount,
        discounted_mean=estimate,
        discounted_standard_error=standard_error,
        discounted_ci95_low=estimate - 1.96 * standard_error,
        discounted_ci95_high=estimate + 1.96 * standard_error,
        replicate_discounted_prices=tuple(float(value) for value in values),
    )
