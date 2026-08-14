"""Finite-precision two-level telescoping control ladder for Asian calls.

This module implements the smallest nontrivial ladder proposed in the paper,

``C0 + (C1 - C0) + (H - C1) = H``,

where ``H`` is the arithmetic-Asian call payoff, ``C1`` is the call on the
average of two equal-block geometric means, and ``C0`` is a full-period
geometric-Asian call.  Each increment is exposed as its own executable
uniform-threshold ``A`` operator so that amplitude estimation may allocate a
separate error budget to each level.

Independent fixed-point exponential chains do not automatically preserve the
continuous ordering ``C0 <= C1``.  The production parameters exhibit rare
rounding reversals.  To keep the encoded increments nonnegative, the coarsest
control uses a certified strike shift ``delta``.  The shift is computed from
exhaustive one-dimensional rounding-error bounds for the two fine block
geometric means.  The shifted coarsest control remains analytically tractable,
and the exact finite-grid identity becomes

``H = C0(K + delta) + D1 + D2``.

All circuit operations are H, X, CX, or Toffoli.  As in
``arithmetic_asian_oracle``, reported T counts apply the explicit convention
Toffoli = 7 T; they are not a physical-resource or runtime estimate.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING, localcontext
from functools import lru_cache
from numbers import Integral
from typing import Iterable, Literal, Sequence

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder

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
    _floor_product,
    _floor_stable,
    _positive_part_counts,
    _product_width,
    build_selected_fixed_multiplier_gate,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec


K2Increment = Literal[
    "coarse_to_blocked",
    "blocked_to_target",
    "coarse_to_target",
]


@dataclass(frozen=True)
class BlockGeometricModel:
    """One collapsed geometric-average leg for a contiguous date block."""

    block_count: int
    block_index: int
    start_date: int
    stop_date: int
    shock_weights: tuple[int, ...]
    shock_weight_sum: int
    shock_weight_bits: int
    initial_geometric: int
    chain_factors: tuple[int, ...]
    maximum_geometric: int
    rounding_error_bound_units: int
    product_bits: int = 0


@dataclass(frozen=True)
class ControlPartitionModel:
    """Encoded call control formed from equal contiguous blocks."""

    block_count: int
    blocks: tuple[BlockGeometricModel, ...]
    strike_adjustment_units: int
    maximum_block_sum: int
    maximum_common_numerator: int


@dataclass(frozen=True)
class K2LadderSharedModel:
    """Arithmetic and control data shared by all three ladder oracles."""

    spec: AsianGridSpec
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
    partitions: tuple[ControlPartitionModel, ...]
    coarse_strike_adjustment_units: int
    coarsest_control_undiscounted: float

    @property
    def partition_map(self) -> dict[int, ControlPartitionModel]:
        return {partition.block_count: partition for partition in self.partitions}

    @property
    def coarse_strike_adjustment_dollars(self) -> float:
        return self.coarse_strike_adjustment_units / self.spec.price_scale


@dataclass(frozen=True)
class K2LadderIncrementModel:
    """One executable increment of the certified two-block ladder."""

    shared: K2LadderSharedModel
    increment: K2Increment
    requested_cap_numerator: int
    normalization_numerator: int
    threshold_bits: int
    maximum_increment_bound_numerator: int

    @property
    def spec(self) -> AsianGridSpec:
        return self.shared.spec

    @property
    def normalization_dollars(self) -> float:
        return self.normalization_numerator / (
            self.spec.n_dates * self.spec.price_scale
        )


@dataclass(frozen=True)
class K2LadderPathValues:
    """Common-denominator integer payoffs on one shock path."""

    target: int
    coarse_control: int
    blocked_control: int
    coarse_to_blocked: int
    blocked_to_target: int

    @property
    def coarse_to_target(self) -> int:
        return self.target - self.coarse_control


@dataclass(frozen=True)
class K2LadderReference:
    """Exact enumeration of a tractable finite ladder increment."""

    path_count: int
    increment_undiscounted: float
    clipped_increment_undiscounted: float
    objective_probability: float
    minimum_increment_numerator: int
    maximum_increment_numerator: int


@dataclass(frozen=True)
class K2LadderOracle:
    """Executable ``A`` operator for one ladder increment."""

    circuit: QuantumCircuit
    model: K2LadderIncrementModel
    objective_qubit: int
    shock_qubits: tuple[int, ...]
    threshold_qubits: tuple[int, ...]
    work_qubits: tuple[int, ...]

    def post_process(self, objective_probability: float) -> float:
        """Decode a discounted increment, excluding the coarsest control."""

        if not 0.0 <= objective_probability <= 1.0:
            raise ValueError("objective probability must lie in [0, 1]")
        undiscounted = self.model.normalization_dollars * objective_probability
        spec = self.model.spec
        return math.exp(-spec.rate * spec.maturity) * undiscounted


def _validate_base_spec(spec: AsianGridSpec) -> None:
    if spec.shock_qubits != 1:
        raise ValueError("the k=2 ladder currently requires one binary shock qubit per date")
    if spec.n_dates % 2:
        raise ValueError("the k=2 ladder requires an even number of fixing dates")
    if not math.isclose(spec.shock_probabilities[0], 0.5, abs_tol=1e-14) or not math.isclose(
        spec.shock_probabilities[1], 0.5, abs_tol=1e-14
    ):
        raise ValueError("the k=2 ladder currently requires uniform binary shocks")
    if not spec.shock_points[1] > spec.shock_points[0]:
        raise ValueError("binary shock points must be strictly increasing")


def _block_weights(n_dates: int, start: int, stop: int) -> tuple[int, ...]:
    """Return unnormalised shock coefficients for dates ``[start, stop)``."""

    return tuple(
        max(0, stop - max(start, shock_date))
        for shock_date in range(n_dates)
    )


def _encoded_block_value(
    block: BlockGeometricModel,
    weighted_sum: int,
    fraction_bits: int,
) -> int:
    value = block.initial_geometric
    for bit, factor in enumerate(block.chain_factors):
        if (weighted_sum >> bit) & 1:
            value = _floor_product(value, factor, fraction_bits)
    return value


def _block_error_bound_units(
    *,
    spec: AsianGridSpec,
    block: BlockGeometricModel,
    multiplier_fraction_bits: int,
) -> int:
    """Certify a price-grid bound on downward exponential-chain error.

    The block value depends only on one integer weighted sum.  Inspecting every
    value from zero through its maximum is therefore exhaustive even though
    the underlying path space has ``2**N`` states.
    """

    n = spec.n_dates
    m = block.stop_date - block.start_date
    low, high = spec.shock_points
    with localcontext() as context:
        context.prec = 80
        d = lambda value: Decimal(str(value))
        dt = d(spec.maturity) / Decimal(n)
        drift = d(spec.rate) - d(spec.volatility) ** 2 / Decimal(2)
        diffusion = d(spec.volatility) * dt.sqrt()
        average_fixing_index = Decimal(
            block.start_date + block.stop_date + 1
        ) / Decimal(2)
        maximum_error = Decimal(0)
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
            exact_price_units = d(spec.s0) * exponent.exp() * Decimal(spec.price_scale)
            encoded = _encoded_block_value(
                block, weighted_sum, multiplier_fraction_bits
            )
            maximum_error = max(maximum_error, exact_price_units - Decimal(encoded))
    return max(
        0,
        int(maximum_error.to_integral_value(rounding=ROUND_CEILING)),
    )


def _compile_block(
    *,
    spec: AsianGridSpec,
    multiplier_fraction_bits: int,
    block_count: int,
    block_index: int,
) -> BlockGeometricModel:
    n = spec.n_dates
    m = n // block_count
    start = block_index * m
    stop = start + m
    weights = _block_weights(n, start, stop)
    weight_sum = sum(weights)
    weight_bits = max(1, weight_sum.bit_length())
    dt = spec.maturity / n
    drift = spec.rate - 0.5 * spec.volatility**2
    diffusion = spec.volatility * math.sqrt(dt)
    low, high = spec.shock_points
    average_fixing_index = (start + stop + 1) / 2.0
    factor_scale = 1 << multiplier_fraction_bits
    initial = _floor_stable(
        spec.s0
        * math.exp(
            drift * dt * average_fixing_index
            + diffusion * low * weight_sum / m
        )
        * spec.price_scale
    )
    factors = tuple(
        _floor_stable(
            math.exp(diffusion * (high - low) * (1 << bit) / m)
            * factor_scale
        )
        for bit in range(weight_bits)
    )
    if initial <= 0 or min(factors) <= 0:
        raise ValueError("fixed-point precision rounds a block-geometric value to zero")
    provisional = BlockGeometricModel(
        block_count=block_count,
        block_index=block_index,
        start_date=start,
        stop_date=stop,
        shock_weights=weights,
        shock_weight_sum=weight_sum,
        shock_weight_bits=weight_bits,
        initial_geometric=initial,
        chain_factors=factors,
        maximum_geometric=0,
        rounding_error_bound_units=0,
    )
    maximum = _encoded_block_value(provisional, weight_sum, multiplier_fraction_bits)
    provisional = replace(provisional, maximum_geometric=maximum)
    error_bound = _block_error_bound_units(
        spec=spec,
        block=provisional,
        multiplier_fraction_bits=multiplier_fraction_bits,
    )
    return replace(provisional, rounding_error_bound_units=error_bound)


def _exact_coarsest_control(
    *,
    spec: AsianGridSpec,
    block: BlockGeometricModel,
    multiplier_fraction_bits: int,
    strike_adjustment_units: int,
) -> float:
    """Exact finite-grid expectation of the shifted coarsest control."""

    counts = [0] * (block.shock_weight_sum + 1)
    counts[0] = 1
    reachable = 1
    for weight in block.shock_weights:
        if weight == 0:
            continue
        for value in range(reachable - 1, -1, -1):
            count = counts[value]
            if count:
                counts[value + weight] += count
        reachable += weight
    if sum(counts) != 1 << spec.n_dates:
        raise AssertionError("coarsest-control dynamic program lost probability mass")
    strike = spec.strike_integer + strike_adjustment_units
    numerator = sum(
        count
        * max(
            _encoded_block_value(block, weighted_sum, multiplier_fraction_bits)
            - strike,
            0,
        )
        for weighted_sum, count in enumerate(counts)
    )
    return numerator / ((1 << spec.n_dates) * spec.price_scale)


@lru_cache(maxsize=None)
def _build_k2_shared_model(
    spec: AsianGridSpec,
    multiplier_fraction_bits: int,
) -> K2LadderSharedModel:
    _validate_base_spec(spec)
    if (
        isinstance(multiplier_fraction_bits, bool)
        or not isinstance(multiplier_fraction_bits, Integral)
        or multiplier_fraction_bits < 1
    ):
        raise ValueError("multiplier_fraction_bits must be a positive integer")

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

    raw_blocks = {
        block_count: tuple(
            _compile_block(
                spec=spec,
                multiplier_fraction_bits=multiplier_fraction_bits,
                block_count=block_count,
                block_index=block_index,
            )
            for block_index in range(block_count)
        )
        for block_count in (1, 2)
    }
    # If g0, g1, g2 denote encoded geometric averages, the exact AM--GM
    # ordering and the certified downward-error bounds imply
    #   2*g0 - g1 - g2 <= error_1 + error_2.
    # Raising the coarsest strike by ceil((error_1 + error_2)/2) therefore
    # restores the encoded call-payoff ordering pathwise.
    fine_error_sum = sum(
        block.rounding_error_bound_units for block in raw_blocks[2]
    )
    coarse_shift = (fine_error_sum + 1) // 2

    maximum_value = max(
        initial_price,
        *maximum_prices,
        *(block.maximum_geometric for blocks in raw_blocks.values() for block in blocks),
    )
    value_bits = max(1, maximum_value.bit_length())
    price_product_bits = _product_width(
        value_bits=value_bits,
        fraction_bits=multiplier_fraction_bits,
        maximum_value=maximum_value,
        factors=price_factors,
    )
    compiled_blocks = {
        block_count: tuple(
            replace(
                block,
                product_bits=_product_width(
                    value_bits=value_bits,
                    fraction_bits=multiplier_fraction_bits,
                    maximum_value=maximum_value,
                    factors=block.chain_factors,
                ),
            )
            for block in blocks
        )
        for block_count, blocks in raw_blocks.items()
    }

    provisional_partitions: list[ControlPartitionModel] = []
    common_candidates = [maximum_total, arithmetic_maximum]
    for block_count in (1, 2):
        adjustment = coarse_shift if block_count == 1 else 0
        maximum_sum = sum(
            block.maximum_geometric for block in compiled_blocks[block_count]
        )
        payoff = max(
            maximum_sum
            - block_count * (spec.strike_integer + adjustment),
            0,
        )
        common_numerator = (n // block_count) * payoff
        common_candidates.extend([maximum_sum, common_numerator])
        provisional_partitions.append(
            ControlPartitionModel(
                block_count=block_count,
                blocks=compiled_blocks[block_count],
                strike_adjustment_units=adjustment,
                maximum_block_sum=maximum_sum,
                maximum_common_numerator=common_numerator,
            )
        )
    common_bits = max(1, max(common_candidates).bit_length())
    coarsest = _exact_coarsest_control(
        spec=spec,
        block=compiled_blocks[1][0],
        multiplier_fraction_bits=multiplier_fraction_bits,
        strike_adjustment_units=coarse_shift,
    )
    return K2LadderSharedModel(
        spec=spec,
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
        partitions=tuple(provisional_partitions),
        coarse_strike_adjustment_units=coarse_shift,
        coarsest_control_undiscounted=coarsest,
    )


def build_k2_ladder_model(
    spec: AsianGridSpec,
    increment: K2Increment,
    *,
    multiplier_fraction_bits: int = 12,
    increment_cap_dollars: float | None = None,
) -> K2LadderIncrementModel:
    """Compile one of the two ladder increments or its direct comparator."""

    if increment not in {
        "coarse_to_blocked",
        "blocked_to_target",
        "coarse_to_target",
    }:
        raise ValueError("unknown k=2 ladder increment")
    shared = _build_k2_shared_model(spec, multiplier_fraction_bits)
    partitions = shared.partition_map
    if increment == "coarse_to_blocked":
        maximum_bound = partitions[2].maximum_common_numerator
    else:
        maximum_bound = shared.arithmetic_payoff_maximum
    if increment_cap_dollars is None:
        requested_cap = maximum_bound
    else:
        if not math.isfinite(increment_cap_dollars) or increment_cap_dollars <= 0.0:
            raise ValueError("increment_cap_dollars must be finite and positive")
        requested_cap = int(
            round(increment_cap_dollars * spec.n_dates * spec.price_scale)
        )
        if requested_cap < 1:
            raise ValueError("increment cap is below one encoded unit")
    threshold_bits = max(1, (requested_cap - 1).bit_length())
    return K2LadderIncrementModel(
        shared=shared,
        increment=increment,
        requested_cap_numerator=requested_cap,
        normalization_numerator=1 << threshold_bits,
        threshold_bits=threshold_bits,
        maximum_increment_bound_numerator=maximum_bound,
    )


def _partition_common_numerator(
    shared: K2LadderSharedModel,
    partition: ControlPartitionModel,
    digits: Sequence[int],
) -> int:
    block_sum = 0
    for block in partition.blocks:
        weighted = sum(
            weight * digit for weight, digit in zip(block.shock_weights, digits)
        )
        block_sum += _encoded_block_value(
            block, weighted, shared.multiplier_fraction_bits
        )
    payoff = max(
        block_sum
        - partition.block_count
        * (shared.spec.strike_integer + partition.strike_adjustment_units),
        0,
    )
    return (shared.spec.n_dates // partition.block_count) * payoff


def k2_ladder_path_values(
    shared: K2LadderSharedModel,
    digits: Sequence[int],
) -> K2LadderPathValues:
    """Evaluate both ladder increments on one path using integer recurrences."""

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
    partitions = shared.partition_map
    coarse = _partition_common_numerator(shared, partitions[1], digits)
    blocked = _partition_common_numerator(shared, partitions[2], digits)
    d1 = blocked - coarse
    d2 = target - blocked
    if d1 < 0:
        raise AssertionError("certified strike shift failed to order C0 and C1")
    if d2 < 0:
        raise AssertionError("blocked control exceeded the arithmetic payoff")
    if target != coarse + d1 + d2:
        raise AssertionError("finite-grid ladder failed to telescope")
    return K2LadderPathValues(
        target=target,
        coarse_control=coarse,
        blocked_control=blocked,
        coarse_to_blocked=d1,
        blocked_to_target=d2,
    )


def _increment_numerator(
    model: K2LadderIncrementModel,
    values: K2LadderPathValues,
) -> int:
    if model.increment == "coarse_to_blocked":
        return values.coarse_to_blocked
    if model.increment == "blocked_to_target":
        return values.blocked_to_target
    return values.coarse_to_target


def enumerate_k2_ladder_increment(
    model: K2LadderIncrementModel,
) -> K2LadderReference:
    """Exhaustively enumerate a tractable ladder increment."""

    path_count = 1 << model.spec.n_dates
    if path_count > 1_000_000:
        raise ValueError("k=2 ladder enumeration is limited to one million paths")
    total = 0
    clipped = 0
    minimum = math.inf
    maximum = 0
    for digits in itertools.product(range(2), repeat=model.spec.n_dates):
        value = _increment_numerator(
            model, k2_ladder_path_values(model.shared, digits)
        )
        total += value
        clipped += min(value, model.requested_cap_numerator)
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    denominator = path_count * model.spec.n_dates * model.spec.price_scale
    return K2LadderReference(
        path_count=path_count,
        increment_undiscounted=total / denominator,
        clipped_increment_undiscounted=clipped / denominator,
        objective_probability=clipped
        / (path_count * model.normalization_numerator),
        minimum_increment_numerator=int(minimum),
        maximum_increment_numerator=maximum,
    )


def iter_k2_ladder_path_values(
    shared: K2LadderSharedModel,
) -> Iterable[tuple[tuple[int, ...], K2LadderPathValues]]:
    path_count = 1 << shared.spec.n_dates
    if path_count > 1_000_000:
        raise ValueError("k=2 ladder path iteration is limited to one million paths")
    for digits in itertools.product(range(2), repeat=shared.spec.n_dates):
        yield digits, k2_ladder_path_values(shared, digits)


def _required_partitions(increment: K2Increment) -> tuple[int, ...]:
    if increment == "coarse_to_blocked":
        return (1, 2)
    if increment == "blocked_to_target":
        return (2,)
    return (1,)


def _uses_arithmetic(increment: K2Increment) -> bool:
    return increment in {"blocked_to_target", "coarse_to_target"}


def build_k2_ladder_oracle(
    model: K2LadderIncrementModel,
) -> K2LadderOracle:
    """Build an executable H/X/CX/Toffoli ``A`` for one ladder increment."""

    shared = model.shared
    spec = shared.spec
    n = spec.n_dates
    q = spec.shock_qubits
    v = shared.value_bits
    c = shared.common_bits
    m = model.threshold_bits
    uses_arithmetic = _uses_arithmetic(model.increment)
    partitions = shared.partition_map
    required = _required_partitions(model.increment)
    max_product_bits = max(
        [shared.price_product_bits if uses_arithmetic else 0]
        + [
            block.product_bits
            for block_count in required
            for block in partitions[block_count].blocks
        ]
    )
    constant_bits = max(max_product_bits, c + 1, m + 1)

    shock = QuantumRegister(n * q, "shock")
    threshold = QuantumRegister(m, "threshold")
    objective = QuantumRegister(1, "objective")
    price0 = QuantumRegister(v, "price0") if uses_arithmetic else None
    price_products = (
        QuantumRegister(n * shared.price_product_bits, "price_products")
        if uses_arithmetic
        else None
    )
    total = QuantumRegister(c, "total") if uses_arithmetic else None
    arithmetic_payoff = (
        QuantumRegister(c, "arithmetic_payoff") if uses_arithmetic else None
    )

    partition_registers: dict[int, dict[str, object]] = {}
    for block_count in required:
        block_registers = []
        for block in partitions[block_count].blocks:
            block_registers.append(
                {
                    "initial": QuantumRegister(
                        v, f"p{block_count}b{block.block_index}_g0"
                    ),
                    "weight": QuantumRegister(
                        block.shock_weight_bits,
                        f"p{block_count}b{block.block_index}_weight",
                    ),
                    "products": QuantumRegister(
                        len(block.chain_factors) * block.product_bits,
                        f"p{block_count}b{block.block_index}_products",
                    ),
                }
            )
        partition_registers[block_count] = {
            "blocks": block_registers,
            "sum": QuantumRegister(c, f"p{block_count}_sum"),
            "payoff": QuantumRegister(c, f"p{block_count}_payoff"),
            "scaled": QuantumRegister(c, f"p{block_count}_scaled"),
        }

    residual = QuantumRegister(c, "increment")
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

    work_registers: list[QuantumRegister] = []
    for register in (price0, price_products, total, arithmetic_payoff):
        if register is not None:
            work_registers.append(register)
    for block_count in required:
        registers = partition_registers[block_count]
        for block_register in registers["blocks"]:
            work_registers.extend(
                [
                    block_register["initial"],
                    block_register["weight"],
                    block_register["products"],
                ]
            )
        work_registers.extend(
            [registers["sum"], registers["payoff"], registers["scaled"]]
        )
    work_registers.extend([residual, scratch])
    if pad is not None:
        work_registers.append(pad)
    work_registers.extend([constant, helper, c3temp, equality])
    if equality_work is not None:
        work_registers.append(equality_work)
    work_registers.extend([term, residual_flag, cap_flag])

    circuit = QuantumCircuit(
        shock, threshold, objective, *work_registers, name="AsianK2Ladder-A"
    )
    compute = QuantumCircuit(shock, *work_registers, name="compute-ladder-increment")

    if uses_arithmetic:
        for bit in range(v):
            if (shared.initial_price >> bit) & 1:
                compute.x(price0[bit])
    for block_count in required:
        for block, block_register in zip(
            partitions[block_count].blocks,
            partition_registers[block_count]["blocks"],
        ):
            for bit in range(v):
                if (block.initial_geometric >> bit) & 1:
                    compute.x(block_register["initial"][bit])

    pad_qubits = [] if pad is None else list(pad)
    multiplier_work = [helper[0], c3temp[0], equality[0]]
    if equality_work is not None:
        multiplier_work.extend(equality_work)
    multiplier_work.append(term[0])

    if uses_arithmetic:
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

    control_values: dict[int, QuantumRegister] = {}
    control_sum_adder = CDKMRippleCarryAdder(c, kind="fixed").to_gate()
    for block_count in required:
        partition = partitions[block_count]
        registers = partition_registers[block_count]
        current_geometric_values: list[Sequence] = []
        for block, block_register in zip(partition.blocks, registers["blocks"]):
            weight_register = block_register["weight"]
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
            current_geometric: Sequence = list(block_register["initial"])
            products = block_register["products"]
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
            current_geometric_values.append(current_geometric)
        for current_geometric in current_geometric_values:
            compute.append(
                control_sum_adder,
                [
                    *current_geometric,
                    *pad_qubits,
                    *registers["sum"],
                    helper[0],
                ],
            )
        _append_positive_part(
            compute,
            registers["sum"],
            registers["payoff"],
            block_count
            * (spec.strike_integer + partition.strike_adjustment_units),
            scratch,
            constant,
            helper[0],
        )
        _append_constant_multiplier(
            compute,
            registers["payoff"],
            n // block_count,
            registers["scaled"],
            constant,
            helper[0],
            c3temp[0],
        )
        control_values[block_count] = registers["scaled"]

    if model.increment == "coarse_to_blocked":
        minuend = control_values[2]
        subtrahend = control_values[1]
    elif model.increment == "blocked_to_target":
        minuend = arithmetic_payoff
        subtrahend = control_values[2]
    else:
        minuend = arithmetic_payoff
        subtrahend = control_values[1]
    compute.cx(minuend, residual)
    subtractor = CDKMRippleCarryAdder(c, kind="fixed").to_gate().inverse()
    compute.append(subtractor, [*subtrahend, *residual, helper[0]])

    circuit.h(shock)
    circuit.h(threshold)
    compute_gate = compute.to_gate(label="compute-ladder-increment")
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

    return K2LadderOracle(
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
    shared: K2LadderSharedModel,
    partition: ControlPartitionModel,
) -> tuple[PrimitiveCounts, dict[str, PrimitiveCounts]]:
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
        subtract=partition.block_count
        * (shared.spec.strike_integer + partition.strike_adjustment_units),
    )
    scaling = _constant_multiplier_counts(
        input_bits=shared.common_bits,
        target_bits=shared.common_bits,
        multiplier=shared.spec.n_dates // partition.block_count,
    )
    components = {
        f"p{partition.block_count}_weighted_sums_in_compute": weighted,
        f"p{partition.block_count}_exponential_chains_in_compute": chains,
        f"p{partition.block_count}_block_sum_in_compute": summation,
        f"p{partition.block_count}_positive_part_in_compute": positive,
        f"p{partition.block_count}_common_scaling_in_compute": scaling,
    }
    return weighted + chains + summation + positive + scaling, components


def _k2_a_qubits(model: K2LadderIncrementModel) -> tuple[int, int]:
    shared = model.shared
    spec = shared.spec
    n = spec.n_dates
    v = shared.value_bits
    c = shared.common_bits
    uses_arithmetic = _uses_arithmetic(model.increment)
    required = _required_partitions(model.increment)
    partitions = shared.partition_map
    work = 0
    if uses_arithmetic:
        work += v + n * shared.price_product_bits + 2 * c
    for block_count in required:
        for block in partitions[block_count].blocks:
            work += (
                v
                + block.shock_weight_bits
                + len(block.chain_factors) * block.product_bits
            )
        work += 3 * c
    max_product_bits = max(
        [shared.price_product_bits if uses_arithmetic else 0]
        + [
            block.product_bits
            for block_count in required
            for block in partitions[block_count].blocks
        ]
    )
    constant_bits = max(max_product_bits, c + 1, model.threshold_bits + 1)
    work += (
        c  # residual
        + (c + 1)  # scratch
        + max(0, c - v)  # shared zero padding
        + constant_bits
        + 6
        + max(0, spec.shock_qubits - 2)
    )
    active = n * spec.shock_qubits + model.threshold_bits + 1 + work
    return active, work


def estimate_k2_ladder_resources(
    model: K2LadderIncrementModel,
) -> ArithmeticAsianResourceEstimate:
    """Count one ladder ``A`` and Grover iterate compositionally."""

    shared = model.shared
    spec = shared.spec
    n = spec.n_dates
    c = shared.common_bits
    m = model.threshold_bits
    uses_arithmetic = _uses_arithmetic(model.increment)
    required = _required_partitions(model.increment)
    partitions = shared.partition_map

    components: dict[str, PrimitiveCounts] = {}
    compute = PrimitiveCounts()
    initialization = PrimitiveCounts()
    if uses_arithmetic:
        initialization += PrimitiveCounts(x=shared.initial_price.bit_count())
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
        compute += price_multipliers + price_sum + arithmetic_positive
        components.update(
            {
                "price_selected_multipliers_in_compute": price_multipliers,
                "arithmetic_price_sum_in_compute": price_sum,
                "arithmetic_positive_part_in_compute": arithmetic_positive,
            }
        )
    for block_count in required:
        partition = partitions[block_count]
        initialization += PrimitiveCounts(
            x=sum(block.initial_geometric.bit_count() for block in partition.blocks)
        )
        control_counts, control_components = _control_partition_counts(
            shared, partition
        )
        compute += control_counts
        components.update(control_components)
    subtraction = PrimitiveCounts(cx=c) + _base_adder_counts(c)
    compute += initialization + subtraction
    components["initialization_in_compute"] = initialization
    components["increment_subtraction_in_compute"] = subtraction

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
    a_qubits, work_qubits = _k2_a_qubits(model)
    reflection = PrimitiveCounts(
        x=2 * a_qubits,
        h=2,
        z=1,
        ccx=2 * a_qubits - 5,
    )
    reflection_ancillas = a_qubits - 3
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
        reflection_clean_ancillas=reflection_ancillas,
        q_qubits_with_clean_reflection_ladder=a_qubits + reflection_ancillas,
        a_counts=a_counts,
        q_counts=q_counts,
        component_counts=components,
    )


def primitive_counts_from_k2_ladder_circuit(
    oracle: K2LadderOracle,
) -> PrimitiveCounts:
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


def k2_ladder_price_from_probabilities(
    coarse_to_blocked: K2LadderOracle,
    coarse_to_blocked_probability: float,
    blocked_to_target: K2LadderOracle,
    blocked_to_target_probability: float,
) -> float:
    """Combine the shifted analytic control and the two decoded increments."""

    first = coarse_to_blocked.model
    second = blocked_to_target.model
    if first.increment != "coarse_to_blocked" or second.increment != "blocked_to_target":
        raise ValueError("oracles must be supplied in ladder order")
    if first.shared != second.shared:
        raise ValueError("ladder increments do not share the same finite model")
    for probability in (coarse_to_blocked_probability, blocked_to_target_probability):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("objective probabilities must lie in [0, 1]")
    shared = first.shared
    undiscounted = (
        shared.coarsest_control_undiscounted
        + first.normalization_dollars * coarse_to_blocked_probability
        + second.normalization_dollars * blocked_to_target_probability
    )
    return math.exp(-shared.spec.rate * shared.spec.maturity) * undiscounted

