"""Reversible finite-grid arithmetic-Asian option oracles.

This module implements a complete *encoded* Asian path oracle.  It does not
prepare one amplitude per precomputed path.  Instead it prepares every shock
register independently, evolves a signed cumulative shock, evaluates each GBM
fixing through a reversible lookup, and accumulates the arithmetic sum.

Two objective encodings are provided:

``raw``
    Encodes the (possibly explicitly capped) arithmetic-call payoff.

``qcv``
    Encodes the nonnegative pathwise residual between the arithmetic call and
    a geometric-Asian control.  Arithmetic prices are rounded upward and the
    geometric average downward.  Consequently discrete AM--GM guarantees that
    the encoded residual is nonnegative on every reachable path.

The price exponential is implemented as a one-dimensional QROM truth table.
For validation, the final residual can either use the original rectangular
QROM or a factorized construction: separate one-dimensional arithmetic- and
geometric-payoff lookups followed by a reversible modular subtraction.  The
latter removes the product-domain residual table, but it is still a reference
oracle rather than a complete fault-tolerant arithmetic implementation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate, ModularAdderGate, RYGate, StatePreparation


OracleKind = Literal["raw", "qcv"]
ResidualMethod = Literal["rectangular_qrom", "factorized_arithmetic"]


def gauss_hermite_normal_grid(shock_qubits: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a ``2**shock_qubits`` Gauss--Hermite grid for N(0, 1).

    The returned rule integrates standard-normal polynomial moments through
    degree ``2 * 2**shock_qubits - 1`` before the later integer shock
    quantization.  It includes the normal tails without imposing a hard cutoff.
    """
    if shock_qubits < 1:
        raise ValueError("shock_qubits must be positive")
    nodes, weights = np.polynomial.hermite.hermgauss(1 << shock_qubits)
    points = np.sqrt(2.0) * nodes
    probabilities = weights / math.sqrt(math.pi)
    probabilities /= probabilities.sum()
    return tuple(float(value) for value in points), tuple(float(value) for value in probabilities)


def _bits_for_unsigned(maximum: int) -> int:
    if maximum < 0:
        raise ValueError("an unsigned maximum cannot be negative")
    return max(1, int(maximum).bit_length())


def _bits_for_signed(minimum: int, maximum: int) -> int:
    """Smallest two's-complement width containing [minimum, maximum]."""
    if minimum > maximum:
        raise ValueError("minimum exceeds maximum")
    width = 1
    while minimum < -(1 << (width - 1)) or maximum > (1 << (width - 1)) - 1:
        width += 1
    return width


def _encode_signed(value: int, width: int) -> int:
    if value < -(1 << (width - 1)) or value > (1 << (width - 1)) - 1:
        raise ValueError(f"{value} does not fit in {width} signed bits")
    return value % (1 << width)


def _ceil_fixed(value: float, scale: int) -> int:
    # The tolerance prevents an exactly representable integer from being
    # rounded upward because of a one-ulp transcendental error.
    return int(math.ceil(value * scale - 32.0 * np.finfo(float).eps * max(1.0, abs(value * scale))))


def _floor_fixed(value: float, scale: int) -> int:
    return int(math.floor(value * scale + 32.0 * np.finfo(float).eps * max(1.0, abs(value * scale))))


@dataclass(frozen=True)
class AsianGridSpec:
    """Definition of the finite stochastic model encoded by the oracle.

    ``shock_points`` are standardized normal approximations and
    ``shock_probabilities`` are their probabilities.  The number of points
    must be a power of two.  Each point is rounded to ``shock_scale`` before it
    enters the reversible dynamics.  ``price_scale`` is units per currency
    unit and the strike must be exactly representable on this grid.

    ``geometric_leg`` selects how the arithmetic Clifford+T construction in
    :mod:`qc_option_pricing.quantum.arithmetic_asian_oracle` evaluates the
    geometric control.  ``'per_date'`` evolves a second price register with one
    shock-selected multiplication per date.  ``'collapsed'`` accumulates the
    weighted shock sum into one small register and then applies one controlled
    constant multiplication per bit of that register.  ``'none'`` builds the
    raw arithmetic oracle with no control variate at all: only the arithmetic
    leg is evolved and the threshold comparator encodes the arithmetic payoff
    directly, capped by ``payoff_cap`` rather than ``residual_payoff_cap``.
    The QROM oracles in this module ignore the field.
    """

    n_dates: int
    shock_points: tuple[float, ...]
    shock_probabilities: tuple[float, ...]
    s0: float
    strike: float
    rate: float
    volatility: float
    maturity: float
    shock_scale: int = 8
    price_scale: int = 4
    payoff_cap: float | None = None
    residual_payoff_cap: float | None = None
    geometric_leg: str = "per_date"

    def __post_init__(self) -> None:
        if isinstance(self.n_dates, bool) or not isinstance(self.n_dates, Integral) or self.n_dates < 1:
            raise ValueError("n_dates must be positive")
        scalar_parameters = {
            "s0": self.s0,
            "strike": self.strike,
            "rate": self.rate,
            "volatility": self.volatility,
            "maturity": self.maturity,
        }
        if not all(math.isfinite(float(value)) for value in scalar_parameters.values()):
            raise ValueError("model parameters must be finite")
        if self.s0 <= 0.0 or self.strike < 0.0:
            raise ValueError("s0 must be positive and strike nonnegative")
        if self.volatility < 0.0 or self.maturity <= 0.0:
            raise ValueError("volatility must be nonnegative and maturity positive")
        if any(
            isinstance(scale, bool) or not isinstance(scale, Integral) or scale < 1
            for scale in (self.shock_scale, self.price_scale)
        ):
            raise ValueError("fixed-point scales must be positive integers")
        if len(self.shock_points) < 2 or len(self.shock_points) != len(self.shock_probabilities):
            raise ValueError("shock points and probabilities must have equal length >= 2")
        if len(self.shock_points) & (len(self.shock_points) - 1):
            raise ValueError("the number of shock points must be a power of two")
        probabilities = np.asarray(self.shock_probabilities, dtype=float)
        if np.any(probabilities < 0.0) or not np.isfinite(probabilities).all():
            raise ValueError("shock probabilities must be finite and nonnegative")
        if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("shock probabilities must sum to one")
        if not np.isfinite(np.asarray(self.shock_points, dtype=float)).all():
            raise ValueError("shock points must be finite")
        if self.payoff_cap is not None and (
            not math.isfinite(float(self.payoff_cap)) or self.payoff_cap <= 0.0
        ):
            raise ValueError("payoff_cap must be positive")
        if self.residual_payoff_cap is not None and (
            not math.isfinite(float(self.residual_payoff_cap))
            or self.residual_payoff_cap <= 0.0
        ):
            raise ValueError("residual_payoff_cap must be positive")
        if self.geometric_leg not in ("per_date", "collapsed", "none"):
            raise ValueError("geometric_leg must be 'per_date', 'collapsed', or 'none'")
        strike_scaled = self.strike * self.price_scale
        if not math.isclose(strike_scaled, round(strike_scaled), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("strike * price_scale must be an integer")

    @property
    def shock_qubits(self) -> int:
        return (len(self.shock_points) - 1).bit_length()

    @property
    def shock_integers(self) -> tuple[int, ...]:
        return tuple(int(round(z * self.shock_scale)) for z in self.shock_points)

    @property
    def strike_integer(self) -> int:
        return int(round(self.strike * self.price_scale))


@dataclass(frozen=True)
class AsianOracleModel:
    """All integer tables and register widths for an :class:`AsianGridSpec`."""

    spec: AsianGridSpec
    prefix_bits: int
    total_bits: int
    weighted_bits: int
    residual_bits: int
    prefix_min: int
    prefix_max: int
    weighted_min: int
    weighted_max: int
    maximum_total: int
    raw_cap_numerator: int
    residual_cap_numerator: int
    price_tables: tuple[Mapping[int, int], ...]
    geometric_table: Mapping[int, int]

    @property
    def raw_cap_dollars(self) -> float:
        return self.raw_cap_numerator / (self.spec.n_dates * self.spec.price_scale)

    @property
    def residual_cap_dollars(self) -> float:
        return self.residual_cap_numerator / (self.spec.n_dates * self.spec.price_scale)


@dataclass(frozen=True)
class AsianOracleCircuit:
    """Circuit plus the exact decoding convention for its objective amplitude."""

    circuit: QuantumCircuit
    model: AsianOracleModel
    kind: OracleKind
    residual_method: ResidualMethod | None
    objective_qubit: int
    shock_qubits: tuple[int, ...]
    work_qubits: tuple[int, ...]
    lookup_rows: int
    lookup_bit_toggles: int
    controlled_rotations: int
    geometric_control_undiscounted: float

    @property
    def cap_dollars(self) -> float:
        if self.kind == "raw":
            return self.model.raw_cap_dollars
        return self.model.residual_cap_dollars

    def post_process(self, objective_probability: float) -> float:
        """Convert objective-one probability to a discounted option price."""
        if not 0.0 <= objective_probability <= 1.0:
            raise ValueError("objective probability must lie in [0, 1]")
        residual = self.cap_dollars * objective_probability
        undiscounted = residual
        if self.kind == "qcv":
            undiscounted += self.geometric_control_undiscounted
        return math.exp(-self.model.spec.rate * self.model.spec.maturity) * undiscounted


@dataclass(frozen=True)
class EncodedAsianReference:
    """Independent exhaustive result for a small encoded model."""

    path_count: int
    probability_mass: float
    arithmetic_payoff_undiscounted: float
    geometric_payoff_undiscounted: float
    residual_payoff_undiscounted: float
    raw_objective_probability: float
    qcv_objective_probability: float
    clipped_raw_payoff_undiscounted: float
    clipped_residual_payoff_undiscounted: float
    minimum_residual_numerator: int


@dataclass(frozen=True)
class AsianOracleResourceEstimate:
    """Logical resource structure before gate synthesis."""

    kind: OracleKind
    residual_method: ResidualMethod | None
    n_dates: int
    shock_qubits: int
    work_qubits: int
    total_qubits: int
    modular_adders: int
    lookup_rows: int
    lookup_bit_toggles: int
    controlled_rotations: int
    notes: str


def build_asian_model(spec: AsianGridSpec) -> AsianOracleModel:
    """Compile a finite GBM grid into integer lookup tables and widths."""
    shock_ints = spec.shock_integers
    z_min, z_max = min(shock_ints), max(shock_ints)
    if z_min == z_max:
        raise ValueError("shock quantization collapsed all shock points to one integer")

    prefix_min = min(0, spec.n_dates * z_min)
    prefix_max = max(0, spec.n_dates * z_max)
    prefix_bits = _bits_for_signed(prefix_min, prefix_max)
    dt = spec.maturity / spec.n_dates
    drift = spec.rate - 0.5 * spec.volatility**2
    diffusion = spec.volatility * math.sqrt(dt) / spec.shock_scale

    price_tables: list[dict[int, int]] = []
    maximum_total = 0
    for fixing in range(1, spec.n_dates + 1):
        lower = min(0, fixing * z_min)
        upper = max(0, fixing * z_max)
        table: dict[int, int] = {}
        fixing_maximum = 0
        for cumulative_shock in range(lower, upper + 1):
            stock = spec.s0 * math.exp(drift * fixing * dt + diffusion * cumulative_shock)
            encoded = _ceil_fixed(stock, spec.price_scale)
            table[_encode_signed(cumulative_shock, prefix_bits)] = encoded
            fixing_maximum = max(fixing_maximum, encoded)
        price_tables.append(table)
        maximum_total += fixing_maximum

    total_bits = _bits_for_unsigned(maximum_total)
    strike_sum = spec.n_dates * spec.strike_integer
    largest_raw = max(0, maximum_total - strike_sum)
    if largest_raw == 0:
        raise ValueError("the encoded payoff is identically zero on the configured range")

    if spec.payoff_cap is None:
        raw_cap = largest_raw
    else:
        raw_cap = int(round(spec.payoff_cap * spec.n_dates * spec.price_scale))
        if raw_cap < 1:
            raise ValueError("payoff_cap is below one encoded payoff unit")

    # The log-geometric average depends on the weighted innovation sum
    # sum_j (N-j+1) z_j.
    weight_sum = spec.n_dates * (spec.n_dates + 1) // 2
    weighted_min = min(0, weight_sum * z_min)
    weighted_max = max(0, weight_sum * z_max)
    weighted_bits = _bits_for_signed(weighted_min, weighted_max)
    average_drift_time = dt * (spec.n_dates + 1) / 2.0
    geometric_diffusion = spec.volatility * math.sqrt(dt) / (spec.n_dates * spec.shock_scale)
    geometric_table: dict[int, int] = {}
    minimum_geometric = math.inf
    for weighted_shock in range(weighted_min, weighted_max + 1):
        geometric = spec.s0 * math.exp(
            drift * average_drift_time + geometric_diffusion * weighted_shock
        )
        encoded = _floor_fixed(geometric, spec.price_scale)
        geometric_table[_encode_signed(weighted_shock, weighted_bits)] = encoded
        minimum_geometric = min(minimum_geometric, encoded)

    if spec.residual_payoff_cap is None:
        # The residual cannot exceed the raw payoff, so the raw cap is a valid
        # default -- but it forfeits the QCV's range advantage.  Pass
        # residual_payoff_cap (see residual_cap_from_bias_budget) to claim it.
        residual_cap = raw_cap
    else:
        residual_cap = int(round(spec.residual_payoff_cap * spec.n_dates * spec.price_scale))
        if residual_cap < 1:
            raise ValueError("residual_payoff_cap is below one encoded payoff unit")
    residual_bits = _bits_for_unsigned(largest_raw)

    return AsianOracleModel(
        spec=spec,
        prefix_bits=prefix_bits,
        total_bits=total_bits,
        weighted_bits=weighted_bits,
        residual_bits=residual_bits,
        prefix_min=prefix_min,
        prefix_max=prefix_max,
        weighted_min=weighted_min,
        weighted_max=weighted_max,
        maximum_total=maximum_total,
        raw_cap_numerator=raw_cap,
        residual_cap_numerator=residual_cap,
        price_tables=tuple(price_tables),
        geometric_table=geometric_table,
    )


def _condition_on_state(
    circuit: QuantumCircuit,
    controls: Sequence,
    state: int,
) -> list:
    """Turn a computational-basis equality into an all-one control."""
    flipped = []
    for bit, qubit in enumerate(controls):
        if ((state >> bit) & 1) == 0:
            circuit.x(qubit)
            flipped.append(qubit)
    return flipped


def _restore_condition(circuit: QuantumCircuit, flipped: Iterable) -> None:
    for qubit in reversed(tuple(flipped)):
        circuit.x(qubit)


def _xor_lookup(
    circuit: QuantumCircuit,
    controls: Sequence,
    target: Sequence,
    table: Mapping[int, int],
) -> tuple[int, int]:
    """Apply |x>|y> -> |x>|y xor table[x]> and return row/toggle counts."""
    rows = 0
    toggles = 0
    target_limit = 1 << len(target)
    for state, value in sorted(table.items()):
        if value < 0 or value >= target_limit:
            raise ValueError(f"lookup value {value} does not fit in {len(target)} bits")
        if value == 0:
            continue
        rows += 1
        flipped = _condition_on_state(circuit, controls, state)
        for bit, qubit in enumerate(target):
            if (value >> bit) & 1:
                circuit.append(MCXGate(len(controls)), [*controls, qubit])
                toggles += 1
        _restore_condition(circuit, flipped)
    return rows, toggles


def _controlled_payoff_rotation(
    circuit: QuantumCircuit,
    controls: Sequence,
    objective,
    numerators: Mapping[int, int],
    cap: int,
) -> int:
    """Encode min(numerator/cap, 1) as objective-one probability."""
    rotations = 0
    for state, numerator in sorted(numerators.items()):
        if numerator <= 0:
            continue
        probability = min(float(numerator) / cap, 1.0)
        theta = 2.0 * math.asin(math.sqrt(probability))
        flipped = _condition_on_state(circuit, controls, state)
        circuit.append(
            RYGate(theta).control(len(controls), annotated=False),
            [*controls, objective],
        )
        _restore_condition(circuit, flipped)
        rotations += 1
    return rotations


def _shock_lookup(model: AsianOracleModel, width: int, weight: int = 1) -> dict[int, int]:
    return {
        digit: _encode_signed(weight * shock, width)
        for digit, shock in enumerate(model.spec.shock_integers)
    }


def _append_shock_loader(circuit: QuantumCircuit, qubits: Sequence, probabilities: Sequence[float]) -> None:
    probabilities_array = np.asarray(probabilities, dtype=float)
    uniform = np.full(len(probabilities_array), 1.0 / len(probabilities_array))
    if np.allclose(probabilities_array, uniform, rtol=0.0, atol=1e-14):
        circuit.h(list(qubits))
    else:
        amplitudes = np.sqrt(probabilities_array).astype(complex)
        circuit.append(StatePreparation(amplitudes), list(qubits))


def _geometric_control_value(model: AsianOracleModel) -> float:
    """Exact expectation of the finite-grid geometric control (no path loop)."""
    spec = model.spec
    # Dynamic programming on the weighted sum is polynomial in its range and
    # independent of the QROM implementation.
    distribution: dict[int, float] = {0: 1.0}
    for date in range(spec.n_dates):
        weight = spec.n_dates - date
        updated: dict[int, float] = {}
        for current, current_probability in distribution.items():
            for shock, probability in zip(spec.shock_integers, spec.shock_probabilities):
                value = current + weight * shock
                updated[value] = updated.get(value, 0.0) + current_probability * probability
        distribution = updated
    expectation_numerator = 0.0
    for weighted_shock, probability in distribution.items():
        encoded_geometric = model.geometric_table[
            _encode_signed(weighted_shock, model.weighted_bits)
        ]
        payoff = max(encoded_geometric - spec.strike_integer, 0)
        expectation_numerator += probability * payoff
    return expectation_numerator / spec.price_scale


def build_asian_oracle(
    spec_or_model: AsianGridSpec | AsianOracleModel,
    kind: OracleKind = "raw",
    *,
    max_residual_lookup_rows: int = 1_000_000,
    residual_method: ResidualMethod = "rectangular_qrom",
) -> AsianOracleCircuit:
    """Build a clean raw or QCV Asian A-operator.

    All non-shock, non-objective registers return to zero.  The objective-one
    probability is a payoff divided by the reported cap.  For ``qcv``, the
    exact finite-grid geometric control must be added in post-processing.

    ``residual_method='rectangular_qrom'`` retains the original two-dimensional
    correctness reference.  ``'factorized_arithmetic'`` loads the arithmetic
    and geometric payoffs through separate one-dimensional QROMs and subtracts
    them reversibly. ``max_residual_lookup_rows`` guards only the rectangular
    method.
    """
    if kind not in ("raw", "qcv"):
        raise ValueError("kind must be 'raw' or 'qcv'")
    if residual_method not in ("rectangular_qrom", "factorized_arithmetic"):
        raise ValueError(
            "residual_method must be 'rectangular_qrom' or 'factorized_arithmetic'"
        )
    model = spec_or_model if isinstance(spec_or_model, AsianOracleModel) else build_asian_model(spec_or_model)
    spec = model.spec

    shock = QuantumRegister(spec.n_dates * spec.shock_qubits, "shock")
    prefix = QuantumRegister(model.prefix_bits, "prefix")
    prefix_addend = QuantumRegister(model.prefix_bits, "p_add")
    total = QuantumRegister(model.total_bits, "total")
    price_addend = QuantumRegister(model.total_bits, "s_add")
    registers: list[QuantumRegister] = [shock, prefix, prefix_addend, total, price_addend]

    weighted = weighted_addend = residual = geometric_payoff = None
    if kind == "qcv":
        weighted = QuantumRegister(model.weighted_bits, "weighted")
        weighted_addend = QuantumRegister(model.weighted_bits, "w_add")
        residual = QuantumRegister(model.residual_bits, "residual")
        registers.extend([weighted, weighted_addend, residual])
        if residual_method == "factorized_arithmetic":
            geometric_payoff = QuantumRegister(model.residual_bits, "g_payoff")
            registers.append(geometric_payoff)
    objective = QuantumRegister(1, "objective")
    registers.append(objective)
    circuit = QuantumCircuit(*registers, name=f"asian_{kind}_A")

    shock_slices = [
        list(shock[date * spec.shock_qubits : (date + 1) * spec.shock_qubits])
        for date in range(spec.n_dates)
    ]
    for qubits in shock_slices:
        _append_shock_loader(circuit, qubits, spec.shock_probabilities)

    prefix_adder = ModularAdderGate(model.prefix_bits)
    total_adder = ModularAdderGate(model.total_bits)
    prefix_lookup = _shock_lookup(model, model.prefix_bits)
    lookup_rows = 0
    lookup_toggles = 0

    # Forward path evolution and arithmetic price accumulation.
    for date, qubits in enumerate(shock_slices):
        rows, toggles = _xor_lookup(circuit, qubits, prefix_addend, prefix_lookup)
        lookup_rows += rows
        lookup_toggles += toggles
        circuit.append(prefix_adder, [*prefix_addend, *prefix])
        rows, toggles = _xor_lookup(circuit, qubits, prefix_addend, prefix_lookup)
        lookup_rows += rows
        lookup_toggles += toggles

        rows, toggles = _xor_lookup(circuit, prefix, price_addend, model.price_tables[date])
        lookup_rows += rows
        lookup_toggles += toggles
        circuit.append(total_adder, [*price_addend, *total])
        rows, toggles = _xor_lookup(circuit, prefix, price_addend, model.price_tables[date])
        lookup_rows += rows
        lookup_toggles += toggles

    rotations = 0
    geometric_control = 0.0
    if kind == "raw":
        strike_sum = spec.n_dates * spec.strike_integer
        payoff_table = {
            total_value: max(total_value - strike_sum, 0)
            for total_value in range(model.maximum_total + 1)
        }
        rotations = _controlled_payoff_rotation(
            circuit,
            total,
            objective[0],
            payoff_table,
            model.raw_cap_numerator,
        )
    else:
        assert weighted is not None and weighted_addend is not None and residual is not None
        weighted_adder = ModularAdderGate(model.weighted_bits)
        # Build the innovation sum required by the geometric average.
        for date, qubits in enumerate(shock_slices):
            weight = spec.n_dates - date
            table = _shock_lookup(model, model.weighted_bits, weight)
            rows, toggles = _xor_lookup(circuit, qubits, weighted_addend, table)
            lookup_rows += rows
            lookup_toggles += toggles
            circuit.append(weighted_adder, [*weighted_addend, *weighted])
            rows, toggles = _xor_lookup(circuit, qubits, weighted_addend, table)
            lookup_rows += rows
            lookup_toggles += toggles

        strike_sum = spec.n_dates * spec.strike_integer
        if residual_method == "rectangular_qrom":
            rectangular_rows = (model.maximum_total + 1) * (
                model.weighted_max - model.weighted_min + 1
            )
            if rectangular_rows > max_residual_lookup_rows:
                raise ValueError(
                    "residual lookup rectangle has "
                    f"{rectangular_rows:,} rows, exceeding max_residual_lookup_rows="
                    f"{max_residual_lookup_rows:,}; call estimate_asian_oracle_resources first"
                )
            residual_table: dict[int, int] = {}
            for weighted_value in range(model.weighted_min, model.weighted_max + 1):
                weighted_code = _encode_signed(weighted_value, model.weighted_bits)
                geometric_integer = model.geometric_table[weighted_code]
                geometric_payoff_sum = spec.n_dates * max(
                    geometric_integer - spec.strike_integer, 0
                )
                for total_value in range(model.maximum_total + 1):
                    arithmetic_payoff_sum = max(total_value - strike_sum, 0)
                    residual_value = max(arithmetic_payoff_sum - geometric_payoff_sum, 0)
                    if residual_value:
                        combined = total_value | (weighted_code << model.total_bits)
                        residual_table[combined] = residual_value

            combined_controls = [*total, *weighted]
            rows, toggles = _xor_lookup(circuit, combined_controls, residual, residual_table)
            lookup_rows += rows
            lookup_toggles += toggles
        else:
            assert geometric_payoff is not None
            arithmetic_payoff_table = {
                total_value: max(total_value - strike_sum, 0)
                for total_value in range(model.maximum_total + 1)
            }
            geometric_payoff_table = {
                _encode_signed(weighted_value, model.weighted_bits): spec.n_dates
                * max(
                    model.geometric_table[
                        _encode_signed(weighted_value, model.weighted_bits)
                    ]
                    - spec.strike_integer,
                    0,
                )
                for weighted_value in range(model.weighted_min, model.weighted_max + 1)
            }
            rows, toggles = _xor_lookup(
                circuit, total, residual, arithmetic_payoff_table
            )
            lookup_rows += rows
            lookup_toggles += toggles
            rows, toggles = _xor_lookup(
                circuit, weighted, geometric_payoff, geometric_payoff_table
            )
            lookup_rows += rows
            lookup_toggles += toggles
            payoff_subtractor = ModularAdderGate(model.residual_bits).inverse()
            circuit.append(payoff_subtractor, [*geometric_payoff, *residual])

        residual_payoff_table = {
            value: value for value in range(1 << model.residual_bits)
        }
        rotations = _controlled_payoff_rotation(
            circuit,
            residual,
            objective[0],
            residual_payoff_table,
            model.residual_cap_numerator,
        )
        if residual_method == "rectangular_qrom":
            rows, toggles = _xor_lookup(circuit, combined_controls, residual, residual_table)
            lookup_rows += rows
            lookup_toggles += toggles
        else:
            assert geometric_payoff is not None
            circuit.append(
                ModularAdderGate(model.residual_bits),
                [*geometric_payoff, *residual],
            )
            rows, toggles = _xor_lookup(
                circuit, weighted, geometric_payoff, geometric_payoff_table
            )
            lookup_rows += rows
            lookup_toggles += toggles
            rows, toggles = _xor_lookup(
                circuit, total, residual, arithmetic_payoff_table
            )
            lookup_rows += rows
            lookup_toggles += toggles

        # Uncompute the weighted innovation sum.
        for date in reversed(range(spec.n_dates)):
            qubits = shock_slices[date]
            weight = spec.n_dates - date
            table = _shock_lookup(model, model.weighted_bits, weight)
            rows, toggles = _xor_lookup(circuit, qubits, weighted_addend, table)
            lookup_rows += rows
            lookup_toggles += toggles
            circuit.append(weighted_adder.inverse(), [*weighted_addend, *weighted])
            rows, toggles = _xor_lookup(circuit, qubits, weighted_addend, table)
            lookup_rows += rows
            lookup_toggles += toggles
        geometric_control = _geometric_control_value(model)

    # Reverse arithmetic accumulation and path evolution.  The shock state and
    # objective remain; every other register returns to |0>.
    for date in reversed(range(spec.n_dates)):
        qubits = shock_slices[date]
        rows, toggles = _xor_lookup(circuit, prefix, price_addend, model.price_tables[date])
        lookup_rows += rows
        lookup_toggles += toggles
        circuit.append(total_adder.inverse(), [*price_addend, *total])
        rows, toggles = _xor_lookup(circuit, prefix, price_addend, model.price_tables[date])
        lookup_rows += rows
        lookup_toggles += toggles

        rows, toggles = _xor_lookup(circuit, qubits, prefix_addend, prefix_lookup)
        lookup_rows += rows
        lookup_toggles += toggles
        circuit.append(prefix_adder.inverse(), [*prefix_addend, *prefix])
        rows, toggles = _xor_lookup(circuit, qubits, prefix_addend, prefix_lookup)
        lookup_rows += rows
        lookup_toggles += toggles

    shock_indices = tuple(circuit.find_bit(qubit).index for qubit in shock)
    objective_index = circuit.find_bit(objective[0]).index
    work_indices = tuple(
        index for index in range(circuit.num_qubits) if index not in (*shock_indices, objective_index)
    )
    return AsianOracleCircuit(
        circuit=circuit,
        model=model,
        kind=kind,
        residual_method=residual_method if kind == "qcv" else None,
        objective_qubit=objective_index,
        shock_qubits=shock_indices,
        work_qubits=work_indices,
        lookup_rows=lookup_rows,
        lookup_bit_toggles=lookup_toggles,
        controlled_rotations=rotations,
        geometric_control_undiscounted=geometric_control,
    )


def enumerate_encoded_asian(
    spec_or_model: AsianGridSpec | AsianOracleModel,
    *,
    max_paths: int = 2_000_000,
) -> EncodedAsianReference:
    """Independently enumerate a small grid for oracle validation only."""
    model = spec_or_model if isinstance(spec_or_model, AsianOracleModel) else build_asian_model(spec_or_model)
    spec = model.spec
    path_count = len(spec.shock_points) ** spec.n_dates
    if path_count > max_paths:
        raise ValueError(f"{path_count:,} paths exceed max_paths={max_paths:,}")

    arithmetic_expectation = 0.0
    geometric_expectation = 0.0
    residual_expectation = 0.0
    clipped_raw = 0.0
    clipped_residual = 0.0
    probability_mass = 0.0
    minimum_residual = math.inf
    strike_sum = spec.n_dates * spec.strike_integer

    for digits in itertools.product(range(len(spec.shock_points)), repeat=spec.n_dates):
        probability = math.prod(spec.shock_probabilities[digit] for digit in digits)
        probability_mass += probability
        cumulative = 0
        total = 0
        weighted = 0
        for date, digit in enumerate(digits):
            shock = spec.shock_integers[digit]
            cumulative += shock
            total += model.price_tables[date][_encode_signed(cumulative, model.prefix_bits)]
            weighted += (spec.n_dates - date) * shock
        geometric = model.geometric_table[_encode_signed(weighted, model.weighted_bits)]
        arithmetic_numerator = max(total - strike_sum, 0)
        geometric_numerator = spec.n_dates * max(geometric - spec.strike_integer, 0)
        residual_numerator = arithmetic_numerator - geometric_numerator
        minimum_residual = min(minimum_residual, residual_numerator)
        if residual_numerator < 0:
            raise AssertionError(
                "discrete AM-GM invariant failed; this indicates a table or rounding bug"
            )

        denominator = spec.n_dates * spec.price_scale
        arithmetic_expectation += probability * arithmetic_numerator / denominator
        geometric_expectation += probability * geometric_numerator / denominator
        residual_expectation += probability * residual_numerator / denominator
        clipped_raw += probability * min(arithmetic_numerator, model.raw_cap_numerator) / denominator
        clipped_residual += probability * min(
            residual_numerator, model.residual_cap_numerator
        ) / denominator

    raw_probability = clipped_raw / model.raw_cap_dollars
    qcv_probability = clipped_residual / model.residual_cap_dollars
    return EncodedAsianReference(
        path_count=path_count,
        probability_mass=probability_mass,
        arithmetic_payoff_undiscounted=arithmetic_expectation,
        geometric_payoff_undiscounted=geometric_expectation,
        residual_payoff_undiscounted=residual_expectation,
        raw_objective_probability=raw_probability,
        qcv_objective_probability=qcv_probability,
        clipped_raw_payoff_undiscounted=clipped_raw,
        clipped_residual_payoff_undiscounted=clipped_residual,
        minimum_residual_numerator=int(minimum_residual),
    )


def residual_cap_from_bias_budget(
    spec_or_model: AsianGridSpec | AsianOracleModel,
    bias_budget_dollars: float,
    *,
    discounted: bool = True,
    max_paths: int = 2_000_000,
) -> float:
    """Smallest grid-representable residual cap meeting a clipping-bias budget.

    Enumerates the exact finite-grid residual distribution and returns the
    smallest cap ``B`` (in dollars) whose exact clipping bias
    ``E[(R - B)_+]`` (discounted by default) does not exceed the budget.
    Pass the result as ``AsianGridSpec.residual_payoff_cap``.  Exact and
    exhaustive, so small grids only.
    """
    if bias_budget_dollars <= 0.0:
        raise ValueError("bias_budget_dollars must be positive")
    model = spec_or_model if isinstance(spec_or_model, AsianOracleModel) else build_asian_model(spec_or_model)
    spec = model.spec
    path_count = len(spec.shock_points) ** spec.n_dates
    if path_count > max_paths:
        raise ValueError(f"{path_count:,} paths exceed max_paths={max_paths:,}")
    factor = math.exp(-spec.rate * spec.maturity) if discounted else 1.0
    denominator = spec.n_dates * spec.price_scale
    strike_sum = spec.n_dates * spec.strike_integer

    distribution: dict[int, float] = {}
    for digits in itertools.product(range(len(spec.shock_points)), repeat=spec.n_dates):
        probability = math.prod(spec.shock_probabilities[digit] for digit in digits)
        cumulative = 0
        total = 0
        weighted = 0
        for date, digit in enumerate(digits):
            shock = spec.shock_integers[digit]
            cumulative += shock
            total += model.price_tables[date][_encode_signed(cumulative, model.prefix_bits)]
            weighted += (spec.n_dates - date) * shock
        geometric = model.geometric_table[_encode_signed(weighted, model.weighted_bits)]
        residual = max(total - strike_sum, 0) - spec.n_dates * max(
            geometric - spec.strike_integer, 0
        )
        distribution[residual] = distribution.get(residual, 0.0) + probability

    def bias(cap_numerator: int) -> float:
        return factor * sum(
            probability * (residual - cap_numerator)
            for residual, probability in distribution.items()
            if residual > cap_numerator
        ) / denominator

    low, high = 1, max(distribution)
    if bias(low) <= bias_budget_dollars:
        return low / denominator
    while low < high:
        mid = (low + high) // 2
        if bias(mid) <= bias_budget_dollars:
            high = mid
        else:
            low = mid + 1
    return high / denominator


def estimate_asian_oracle_resources(
    spec_or_model: AsianGridSpec | AsianOracleModel,
    kind: OracleKind = "raw",
    *,
    residual_method: ResidualMethod = "rectangular_qrom",
) -> AsianOracleResourceEstimate:
    """Estimate the exact high-level structure without constructing the circuit."""
    if kind not in ("raw", "qcv"):
        raise ValueError("kind must be 'raw' or 'qcv'")
    if residual_method not in ("rectangular_qrom", "factorized_arithmetic"):
        raise ValueError(
            "residual_method must be 'rectangular_qrom' or 'factorized_arithmetic'"
        )
    model = spec_or_model if isinstance(spec_or_model, AsianOracleModel) else build_asian_model(spec_or_model)
    spec = model.spec
    shock_lookup = _shock_lookup(model, model.prefix_bits)
    shock_rows = sum(value != 0 for value in shock_lookup.values())
    shock_toggles = sum(value.bit_count() for value in shock_lookup.values())
    price_rows = sum(sum(value != 0 for value in table.values()) for table in model.price_tables)
    price_toggles = sum(sum(value.bit_count() for value in table.values()) for table in model.price_tables)

    # Each lookup is used to compute/uncompute its addend in the forward and
    # reverse arithmetic passes.
    lookup_rows = 4 * spec.n_dates * shock_rows + 4 * price_rows
    lookup_toggles = 4 * spec.n_dates * shock_toggles + 4 * price_toggles
    modular_adders = 4 * spec.n_dates
    controlled_rotations = max(
        0, model.maximum_total - spec.n_dates * spec.strike_integer
    )
    work_qubits = 2 * model.prefix_bits + 2 * model.total_bits
    note = "QROM exponential; counts are pre-synthesis and omit shock-state preparation"

    if kind == "qcv":
        weighted_rows_one_pass = 0
        weighted_toggles_one_pass = 0
        for date in range(spec.n_dates):
            table = _shock_lookup(model, model.weighted_bits, spec.n_dates - date)
            weighted_rows_one_pass += sum(value != 0 for value in table.values())
            weighted_toggles_one_pass += sum(value.bit_count() for value in table.values())
        lookup_rows += 4 * weighted_rows_one_pass
        lookup_toggles += 4 * weighted_toggles_one_pass
        modular_adders += 2 * spec.n_dates
        controlled_rotations = (1 << model.residual_bits) - 1
        work_qubits += 2 * model.weighted_bits + model.residual_bits
        if residual_method == "rectangular_qrom":
            rectangle = (model.maximum_total + 1) * (
                model.weighted_max - model.weighted_min + 1
            )
            # Every nonzero rectangular entry is looked up twice.  This is an upper
            # bound because out-of-the-money and negative-residual entries are zero.
            lookup_rows += 2 * rectangle
            lookup_toggles += 2 * rectangle * model.residual_bits
            note = (
                "QROM exponential plus rectangular residual QROM upper bound; "
                "replace the latter with comparator/subtractor arithmetic for scale"
            )
        else:
            strike_sum = spec.n_dates * spec.strike_integer
            arithmetic_values = (
                max(total_value - strike_sum, 0)
                for total_value in range(model.maximum_total + 1)
            )
            arithmetic_nonzero = [value for value in arithmetic_values if value]
            geometric_values = [
                spec.n_dates
                * max(
                    model.geometric_table[
                        _encode_signed(weighted_value, model.weighted_bits)
                    ]
                    - spec.strike_integer,
                    0,
                )
                for weighted_value in range(model.weighted_min, model.weighted_max + 1)
            ]
            geometric_nonzero = [value for value in geometric_values if value]
            lookup_rows += 2 * (len(arithmetic_nonzero) + len(geometric_nonzero))
            lookup_toggles += 2 * (
                sum(value.bit_count() for value in arithmetic_nonzero)
                + sum(value.bit_count() for value in geometric_nonzero)
            )
            modular_adders += 2
            work_qubits += model.residual_bits
            note = (
                "QROM price exponential; factorized one-dimensional payoff QROMs "
                "plus reversible subtraction; controlled payoff rotations remain"
            )

    shock_qubits = spec.n_dates * spec.shock_qubits
    return AsianOracleResourceEstimate(
        kind=kind,
        residual_method=residual_method if kind == "qcv" else None,
        n_dates=spec.n_dates,
        shock_qubits=shock_qubits,
        work_qubits=work_qubits,
        total_qubits=shock_qubits + work_qubits + 1,
        modular_adders=modular_adders,
        lookup_rows=lookup_rows,
        lookup_bit_toggles=lookup_toggles,
        controlled_rotations=controlled_rotations,
        notes=note,
    )


def objective_probability_from_statevector(oracle: AsianOracleCircuit) -> tuple[float, float]:
    """Return objective probability and nonzero-work leakage for small tests."""
    from qiskit.quantum_info import Statevector

    state = Statevector.from_instruction(oracle.circuit)
    objective_mask = 1 << oracle.objective_qubit
    work_mask = sum(1 << index for index in oracle.work_qubits)
    objective_probability = 0.0
    work_leakage = 0.0
    for basis_index, amplitude in enumerate(state.data):
        probability = float(abs(amplitude) ** 2)
        if basis_index & objective_mask:
            objective_probability += probability
        if basis_index & work_mask:
            work_leakage += probability
    return objective_probability, work_leakage


def objective_probability_from_mps(oracle: AsianOracleCircuit) -> tuple[float, float]:
    """Return objective probability and a work-leakage upper bound (Aer MPS).

    This is the preferred end-to-end validator for the QCV circuit.  It avoids
    allocating a dense ``2**num_qubits`` statevector while still applying the
    fully decomposed circuit, including every compute/uncompute operation.
    The second return value is the union bound ``sum_q P(work qubit q = 1)``;
    it is zero exactly when the uncomputation is clean.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("qiskit-aer is required for MPS validation") from exc

    # Recursive decomposition also avoids a Qiskit 2.4 BasisTranslator issue
    # involving ModularAdderGate's annotated equivalence.
    circuit = oracle.circuit.decompose(reps=8)
    circuit.save_probabilities_dict([oracle.objective_qubit], label="objective")
    # A joint probability dictionary over all work qubits is dense in
    # 2**len(work_qubits) and exhausts memory beyond ~24 work qubits.  The
    # union bound sum_q P(q=1) >= P(any work qubit nonzero) needs only one
    # single-qubit dictionary per work qubit and upper-bounds the leakage,
    # which is the direction a validation needs.
    for position, qubit in enumerate(oracle.work_qubits):
        circuit.save_probabilities_dict([qubit], label=f"work_{position}")
    simulator = AerSimulator(method="matrix_product_state")
    result = simulator.run(circuit, shots=None).result()
    if not result.success:
        raise RuntimeError(f"MPS simulation failed: {result.status}")
    data = result.data(0)
    objective_probability = float(data["objective"].get(1, 0.0))
    work_leakage = sum(
        float(data[f"work_{position}"].get(1, 0.0))
        for position in range(len(oracle.work_qubits))
    )
    # Simulator normalization noise can make this a tiny negative number.
    return objective_probability, max(0.0, work_leakage)


def inverse_roundtrip_leakage_from_mps(oracle: AsianOracleCircuit) -> float:
    """Probability outside |0...0> after applying A followed by A inverse."""
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("qiskit-aer is required for MPS validation") from exc

    circuit = oracle.circuit.compose(oracle.circuit.inverse()).decompose(reps=8)
    circuit.save_probabilities_dict(list(range(circuit.num_qubits)), label="all")
    simulator = AerSimulator(method="matrix_product_state")
    result = simulator.run(circuit, shots=None).result()
    if not result.success:
        raise RuntimeError(f"MPS simulation failed: {result.status}")
    zero_probability = float(result.data(0)["all"].get(0, 0.0))
    return max(0.0, 1.0 - zero_probability)
