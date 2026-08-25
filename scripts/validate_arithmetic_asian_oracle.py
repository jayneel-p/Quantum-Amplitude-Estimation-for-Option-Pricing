"""Generate machine-readable evidence for the arithmetic Asian QCV oracle."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import qiskit
from scipy.special import ndtr

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    arithmetic_objective_probability_from_mps,
    arithmetic_roundtrip_leakage_from_mps,
    build_arithmetic_asian_model,
    build_arithmetic_asian_oracle,
    enumerate_arithmetic_asian,
    estimate_arithmetic_asian_resources,
    primitive_counts_from_circuit,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v8" / "arithmetic_asian_oracle_validation.json"
SEED = 20260716
BIAS_PATHS = 200_000
CONTINUOUS_REFERENCE_PATHS = 1_000_000
CONTINUOUS_REFERENCE_CHUNK = 25_000
CONTINUOUS_REFERENCE_SEED = 20260717
PRICE_ACCURACY_TARGET = 0.01


@dataclass(frozen=True)
class PrecisionCandidate:
    name: str
    price_scale: int
    multiplier_fraction_bits: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_binary_geometric_control_undiscounted(spec: AsianGridSpec) -> tuple[float, int]:
    """Independent DP using the analytic geometric-average path formula."""

    if spec.shock_points != (-1.0, 1.0):
        raise ValueError("this independent check is specialized to binary +/-1 shocks")
    distribution: Counter[int] = Counter({0: 1})
    for date in range(spec.n_dates):
        weight = spec.n_dates - date
        updated: Counter[int] = Counter()
        for weighted_sum, count in distribution.items():
            updated[weighted_sum - weight] += count
            updated[weighted_sum + weight] += count
        distribution = updated
    dt = spec.maturity / spec.n_dates
    drift = spec.rate - 0.5 * spec.volatility**2
    value = sum(
        count
        * max(
            spec.s0
            * math.exp(
                drift * dt * (spec.n_dates + 1) / 2.0
                + spec.volatility
                * math.sqrt(dt)
                * weighted_sum
                / spec.n_dates
            )
            - spec.strike,
            0.0,
        )
        for weighted_sum, count in distribution.items()
    ) / (2**spec.n_dates)
    return value, len(distribution)


def continuous_geometric_asian_call(spec: AsianGridSpec) -> float:
    """Independent Kemna--Vorst formula for the stated monitoring convention."""

    n_dates = spec.n_dates
    sigma_g = spec.volatility * math.sqrt(
        (n_dates + 1) * (2 * n_dates + 1) / (6 * n_dates * n_dates)
    )
    mu_g = (
        (spec.rate - 0.5 * spec.volatility**2) * (n_dates + 1) / (2 * n_dates)
        + 0.5 * sigma_g**2
    )
    d1 = (
        math.log(spec.s0 / spec.strike) + (mu_g + 0.5 * sigma_g**2) * spec.maturity
    ) / (sigma_g * math.sqrt(spec.maturity))
    d2 = d1 - sigma_g * math.sqrt(spec.maturity)
    return math.exp(-spec.rate * spec.maturity) * (
        spec.s0 * math.exp(mu_g * spec.maturity) * ndtr(d1)
        - spec.strike * ndtr(d2)
    )


def continuous_arithmetic_qcv_reference(spec: AsianGridSpec) -> dict[str, float | int]:
    """Independent chunked MC for the continuous arithmetic price.

    The unit-coefficient residual is paired with the analytic geometric control,
    so this estimates exactly the same QCV identity used by the oracle without
    fitting a control coefficient on the simulated paths.
    """

    geometric_price = continuous_geometric_asian_call(spec)
    rng = np.random.default_rng(CONTINUOUS_REFERENCE_SEED)
    n_done = 0
    residual_sum = 0.0
    residual_sum_sq = 0.0
    dt = spec.maturity / spec.n_dates
    drift = (spec.rate - 0.5 * spec.volatility**2) * dt
    diffusion = spec.volatility * math.sqrt(dt)
    discount = math.exp(-spec.rate * spec.maturity)
    while n_done < CONTINUOUS_REFERENCE_PATHS:
        count = min(CONTINUOUS_REFERENCE_CHUNK, CONTINUOUS_REFERENCE_PATHS - n_done)
        shocks = rng.standard_normal((count, spec.n_dates))
        log_prices = math.log(spec.s0) + np.cumsum(drift + diffusion * shocks, axis=1)
        prices = np.exp(log_prices)
        arithmetic = np.maximum(prices.mean(axis=1) - spec.strike, 0.0)
        geometric = np.maximum(np.exp(log_prices.mean(axis=1)) - spec.strike, 0.0)
        residual = discount * (arithmetic - geometric)
        residual_sum += float(residual.sum())
        residual_sum_sq += float(np.dot(residual, residual))
        n_done += count
    residual_mean = residual_sum / CONTINUOUS_REFERENCE_PATHS
    residual_var = (
        residual_sum_sq - CONTINUOUS_REFERENCE_PATHS * residual_mean**2
    ) / (CONTINUOUS_REFERENCE_PATHS - 1)
    residual_se = math.sqrt(residual_var / CONTINUOUS_REFERENCE_PATHS)
    return {
        "paths": CONTINUOUS_REFERENCE_PATHS,
        "seed": CONTINUOUS_REFERENCE_SEED,
        "geometric_control_discounted": geometric_price,
        "residual_discounted": residual_mean,
        "residual_standard_error": residual_se,
        "arithmetic_price_discounted": geometric_price + residual_mean,
        "arithmetic_price_standard_error": residual_se,
    }


def _ceil_stable(value: float) -> int:
    return int(math.ceil(value - 32.0 * math.ulp(max(1.0, abs(value)))))


def _floor_stable(value: float) -> int:
    return int(math.floor(value + 32.0 * math.ulp(max(1.0, abs(value)))))


def precision_sweep(
    spec: AsianGridSpec, exact_binary_geometric_undiscounted: float
) -> list[dict[str, float | int | str]]:
    """Paired finite-model bias sweep, independent of the circuit builder.

    It reports the price error of the capped finite QCV reconstruction relative
    to the same binary path evaluated without directed fixed-point arithmetic.
    The common random path removes almost all ordinary Monte Carlo noise from
    the arithmetic-precision comparison.
    """

    candidates = (
        PrecisionCandidate("current: price_scale=1024, fraction_bits=18", 1024, 18),
        PrecisionCandidate("refined: price_scale=4096, fraction_bits=24", 4096, 24),
        PrecisionCandidate("refined: price_scale=16384, fraction_bits=30", 16384, 30),
    )
    if spec.shock_points != (-1.0, 1.0):
        raise ValueError("the paired precision sweep is specialized to binary shocks")
    rng = np.random.default_rng(SEED)
    dt = spec.maturity / spec.n_dates
    drift = spec.rate - 0.5 * spec.volatility**2
    diffusion = spec.volatility * math.sqrt(dt)
    discount = math.exp(-spec.rate * spec.maturity)
    exact_price = np.full(BIAS_PATHS, spec.s0, dtype=float)
    exact_total = np.zeros(BIAS_PATHS, dtype=float)
    exact_geometric = np.full(
        BIAS_PATHS,
        spec.s0 * math.exp(drift * dt * (spec.n_dates + 1) / 2.0),
        dtype=float,
    )
    states: list[dict[str, object]] = []
    for candidate in candidates:
        factor_scale = 1 << candidate.multiplier_fraction_bits
        price_factors = np.asarray(
            [
                _ceil_stable(
                    math.exp(drift * dt + diffusion * shock) * factor_scale
                )
                for shock in spec.shock_points
            ],
            dtype=np.int64,
        )
        geometric_factors = np.asarray(
            [
                [
                    _floor_stable(
                        math.exp(
                            diffusion
                            * (spec.n_dates - date)
                            * shock
                            / spec.n_dates
                        )
                        * factor_scale
                    )
                    for shock in spec.shock_points
                ]
                for date in range(spec.n_dates)
            ],
            dtype=np.int64,
        )
        states.append(
            {
                "candidate": candidate,
                "factor_scale": factor_scale,
                "price_factors": price_factors,
                "geometric_factors": geometric_factors,
                "price": np.full(
                    BIAS_PATHS, _ceil_stable(spec.s0 * candidate.price_scale), dtype=np.int64
                ),
                "geometric": np.full(
                    BIAS_PATHS,
                    _floor_stable(
                        spec.s0
                        * math.exp(drift * dt * (spec.n_dates + 1) / 2.0)
                        * candidate.price_scale
                    ),
                    dtype=np.int64,
                ),
                "total": np.zeros(BIAS_PATHS, dtype=np.int64),
            }
        )
    for date in range(spec.n_dates):
        digits = rng.integers(0, 2, size=BIAS_PATHS, dtype=np.int8)
        exact_price *= np.where(
            digits == 0,
            math.exp(drift * dt - diffusion),
            math.exp(drift * dt + diffusion),
        )
        exact_total += exact_price
        exact_geometric *= np.where(
            digits == 0,
            math.exp(-diffusion * (spec.n_dates - date) / spec.n_dates),
            math.exp(diffusion * (spec.n_dates - date) / spec.n_dates),
        )
        for state in states:
            fraction_bits = state["candidate"].multiplier_fraction_bits  # type: ignore[union-attr]
            factor_scale = state["factor_scale"]  # type: ignore[assignment]
            price = state["price"]  # type: ignore[assignment]
            geometric = state["geometric"]  # type: ignore[assignment]
            total = state["total"]  # type: ignore[assignment]
            price_factors = state["price_factors"]  # type: ignore[assignment]
            geometric_factors = state["geometric_factors"]  # type: ignore[assignment]
            price[:] = (price * price_factors[digits] + factor_scale - 1) >> fraction_bits
            geometric[:] = (geometric * geometric_factors[date, digits]) >> fraction_bits
            total += price
    exact_arithmetic = np.maximum(exact_total / spec.n_dates - spec.strike, 0.0)
    exact_geometric_payoff = np.maximum(exact_geometric - spec.strike, 0.0)
    rows: list[dict[str, float | int | str]] = []
    for state in states:
        candidate = state["candidate"]  # type: ignore[assignment]
        price = state["price"]  # type: ignore[assignment]
        geometric = state["geometric"]  # type: ignore[assignment]
        total = state["total"]  # type: ignore[assignment]
        arithmetic = np.maximum(
            total / (spec.n_dates * candidate.price_scale) - spec.strike, 0.0
        )
        geometric_payoff = np.maximum(
            geometric / candidate.price_scale - spec.strike, 0.0
        )
        residual = arithmetic - geometric_payoff
        cap = round(spec.residual_payoff_cap * spec.n_dates * candidate.price_scale) / (
            spec.n_dates * candidate.price_scale
        )
        qcv = arithmetic - np.maximum(residual - cap, 0.0)
        difference = discount * (qcv - exact_arithmetic)
        geometric_difference = geometric_payoff - exact_geometric_payoff
        rows.append(
            {
                "name": candidate.name,
                "price_scale": candidate.price_scale,
                "multiplier_fraction_bits": candidate.multiplier_fraction_bits,
                "paired_qcv_minus_exact_binary_price": float(difference.mean()),
                "paired_standard_error": float(
                    difference.std(ddof=1) / math.sqrt(BIAS_PATHS)
                ),
                "clipping_loss_discounted": float(
                    discount * np.maximum(residual - cap, 0.0).mean()
                ),
                "geometric_control_undiscounted_estimate": (
                    exact_binary_geometric_undiscounted
                    + float(geometric_difference.mean())
                ),
                "geometric_control_undiscounted_standard_error": float(
                    geometric_difference.std(ddof=1) / math.sqrt(BIAS_PATHS)
                ),
            }
        )
    return rows


def seeded_rounding_bias(model, exact_binary_geometric_undiscounted: float) -> dict[str, float | int]:
    """Paired finite-model comparison against exact binary path arithmetic."""

    spec = model.spec
    rng = np.random.default_rng(SEED)
    encoded_price = np.full(BIAS_PATHS, model.initial_price, dtype=np.int64)
    encoded_geometric = np.full(
        BIAS_PATHS, model.initial_geometric, dtype=np.int64
    )
    encoded_total = np.zeros(BIAS_PATHS, dtype=np.int64)
    exact_price = np.full(BIAS_PATHS, spec.s0, dtype=float)
    exact_geometric = np.full(
        BIAS_PATHS,
        spec.s0
        * math.exp(
            (spec.rate - 0.5 * spec.volatility**2)
            * (spec.maturity / spec.n_dates)
            * (spec.n_dates + 1)
            / 2.0
        ),
        dtype=float,
    )
    exact_total = np.zeros(BIAS_PATHS, dtype=float)
    dt = spec.maturity / spec.n_dates
    price_factors = np.asarray(
        [
            math.exp(
                (spec.rate - 0.5 * spec.volatility**2) * dt
                + spec.volatility * math.sqrt(dt) * shock
            )
            for shock in spec.shock_points
        ]
    )
    geometric_factors = np.asarray(
        [
            [
                math.exp(
                    spec.volatility
                    * math.sqrt(dt)
                    * (spec.n_dates - date)
                    * shock
                    / spec.n_dates
                )
                for shock in spec.shock_points
            ]
            for date in range(spec.n_dates)
        ]
    )
    factor_scale = model.factor_scale
    for date in range(spec.n_dates):
        digits = rng.integers(0, 2, size=BIAS_PATHS, dtype=np.int8)
        encoded_price = (
            encoded_price * np.asarray(model.price_factors, dtype=np.int64)[digits]
            + factor_scale
            - 1
        ) >> model.multiplier_fraction_bits
        encoded_total += encoded_price
        encoded_geometric = (
            encoded_geometric
            * np.asarray(model.geometric_factors[date], dtype=np.int64)[digits]
        ) >> model.multiplier_fraction_bits
        exact_price *= price_factors[digits]
        exact_total += exact_price
        exact_geometric *= geometric_factors[date, digits]

    encoded_arithmetic = np.maximum(
        encoded_total / (spec.n_dates * spec.price_scale) - spec.strike, 0.0
    )
    exact_arithmetic = np.maximum(exact_total / spec.n_dates - spec.strike, 0.0)
    encoded_geometric_payoff = np.maximum(
        encoded_geometric / spec.price_scale - spec.strike, 0.0
    )
    exact_geometric_payoff = np.maximum(exact_geometric - spec.strike, 0.0)
    arithmetic_difference = encoded_arithmetic - exact_arithmetic
    geometric_difference = encoded_geometric_payoff - exact_geometric_payoff
    encoded_residual = encoded_arithmetic - encoded_geometric_payoff
    exact_residual = exact_arithmetic - exact_geometric_payoff
    requested_cap_dollars = model.requested_residual_cap_numerator / (
        spec.n_dates * spec.price_scale
    )
    normalization_dollars = model.normalization_dollars
    clipping_loss = np.maximum(encoded_residual - requested_cap_dollars, 0.0)
    discount = math.exp(-spec.rate * spec.maturity)
    encoded_clipped_residual = np.minimum(encoded_residual, requested_cap_dollars)
    exact_binary_residual_mean = float(exact_residual.mean())
    exact_binary_residual_se = float(
        exact_residual.std(ddof=1) / math.sqrt(BIAS_PATHS)
    )
    encoded_residual_mean = float(encoded_clipped_residual.mean())
    encoded_residual_se = float(
        encoded_clipped_residual.std(ddof=1) / math.sqrt(BIAS_PATHS)
    )
    return {
        "seed": SEED,
        "paths": BIAS_PATHS,
        "estimand": "encoded directed-rounding payoff minus exact binary-shock payoff",
        "arithmetic_mean_bias_undiscounted": float(arithmetic_difference.mean()),
        "arithmetic_bias_standard_error": float(
            arithmetic_difference.std(ddof=1) / math.sqrt(BIAS_PATHS)
        ),
        "arithmetic_max_path_bias": float(arithmetic_difference.max()),
        "arithmetic_min_path_bias": float(arithmetic_difference.min()),
        "geometric_mean_bias_undiscounted": float(geometric_difference.mean()),
        "geometric_bias_standard_error": float(
            geometric_difference.std(ddof=1) / math.sqrt(BIAS_PATHS)
        ),
        "geometric_max_path_bias": float(geometric_difference.max()),
        "geometric_min_path_bias": float(geometric_difference.min()),
        "requested_residual_cap_dollars": requested_cap_dollars,
        "power_of_two_normalization_dollars": normalization_dollars,
        "encoded_residual_clipping_loss_mean": float(clipping_loss.mean()),
        "encoded_residual_clipping_loss_standard_error": float(
            clipping_loss.std(ddof=1) / math.sqrt(BIAS_PATHS)
        ),
        "exact_binary_arithmetic_price_discounted": discount
        * (exact_binary_geometric_undiscounted + exact_binary_residual_mean),
        "exact_binary_arithmetic_price_standard_error": discount * exact_binary_residual_se,
        "encoded_qcv_capped_price_discounted": discount
        * (model.geometric_control_undiscounted + encoded_residual_mean),
        "encoded_qcv_capped_price_standard_error": discount * encoded_residual_se,
    }


def main() -> None:
    small_spec = AsianGridSpec(
        n_dates=2,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=1.0,
        strike=0.0,
        rate=0.05,
        volatility=0.1,
        maturity=1.0,
        shock_scale=1,
        price_scale=1,
    )
    small_model = build_arithmetic_asian_model(
        small_spec, multiplier_fraction_bits=1
    )
    small_reference = enumerate_arithmetic_asian(small_model)
    small_oracle = build_arithmetic_asian_oracle(small_model)
    small_probability, small_work_hamming = arithmetic_objective_probability_from_mps(
        small_oracle
    )
    small_roundtrip = arithmetic_roundtrip_leakage_from_mps(small_oracle)
    small_actual_counts = primitive_counts_from_circuit(small_oracle)
    small_estimate = estimate_arithmetic_asian_resources(small_model)
    if small_actual_counts != small_estimate.a_counts:
        raise AssertionError("executable and compositional primitive counts disagree")
    if abs(small_probability - small_reference.objective_probability) > 1e-10:
        raise AssertionError("small circuit failed the predeclared objective tolerance")
    if small_work_hamming > 1e-10 or small_roundtrip > 1e-10:
        raise AssertionError("small circuit failed the predeclared cleanup tolerance")

    production_spec = AsianGridSpec(
        n_dates=252,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        shock_scale=1,
        price_scale=1024,
        residual_payoff_cap=2.864,
    )
    production_model = build_arithmetic_asian_model(
        production_spec, multiplier_fraction_bits=18
    )
    production_resources = estimate_arithmetic_asian_resources(production_model)
    exact_binary_geometric, exact_binary_states = (
        exact_binary_geometric_control_undiscounted(production_spec)
    )
    continuous_geometric_discounted = continuous_geometric_asian_call(production_spec)
    continuous_geometric_undiscounted = continuous_geometric_discounted * math.exp(
        production_spec.rate * production_spec.maturity
    )
    continuous_reference = continuous_arithmetic_qcv_reference(production_spec)
    rounding = seeded_rounding_bias(production_model, exact_binary_geometric)
    binary_error = (
        rounding["exact_binary_arithmetic_price_discounted"]
        - continuous_reference["arithmetic_price_discounted"]
    )
    encoded_error = (
        rounding["encoded_qcv_capped_price_discounted"]
        - continuous_reference["arithmetic_price_discounted"]
    )
    binary_combined_se = math.hypot(
        rounding["exact_binary_arithmetic_price_standard_error"],
        continuous_reference["arithmetic_price_standard_error"],
    )
    encoded_combined_se = math.hypot(
        rounding["encoded_qcv_capped_price_standard_error"],
        continuous_reference["arithmetic_price_standard_error"],
    )
    sweep = precision_sweep(production_spec, exact_binary_geometric)
    for row in sweep:
        row["estimated_continuous_price_error"] = (
            binary_error + row["paired_qcv_minus_exact_binary_price"]
        )
        row["conservative_combined_standard_error"] = math.hypot(
            binary_combined_se, row["paired_standard_error"]
        )
        row["meets_target_at_95_percent"] = bool(
            abs(row["estimated_continuous_price_error"])
            + 1.96 * row["conservative_combined_standard_error"]
            <= PRICE_ACCURACY_TARGET
        )
    selected_precision = sweep[-1]
    selected_spec = AsianGridSpec(
        n_dates=production_spec.n_dates,
        shock_points=production_spec.shock_points,
        shock_probabilities=production_spec.shock_probabilities,
        s0=production_spec.s0,
        strike=production_spec.strike,
        rate=production_spec.rate,
        volatility=production_spec.volatility,
        maturity=production_spec.maturity,
        shock_scale=production_spec.shock_scale,
        price_scale=int(selected_precision["price_scale"]),
        residual_payoff_cap=production_spec.residual_payoff_cap,
    )
    selected_model = build_arithmetic_asian_model(
        selected_spec,
        multiplier_fraction_bits=int(selected_precision["multiplier_fraction_bits"]),
        geometric_control_undiscounted_override=float(
            selected_precision["geometric_control_undiscounted_estimate"]
        ),
        geometric_control_standard_error_undiscounted=float(
            selected_precision["geometric_control_undiscounted_standard_error"]
        ),
    )
    selected_resources = estimate_arithmetic_asian_resources(selected_model)

    factorized_path = ROOT / "results" / "v7" / "factorized_asian_oracle_validation.json"
    factorized = json.loads(factorized_path.read_text())
    factorized_252 = next(
        row for row in factorized["coarse_scaling"] if row["dates"] == 252
    )
    source_path = (
        ROOT
        / "src"
        / "qc_option_pricing"
        / "quantum"
        / "arithmetic_asian_oracle.py"
    )
    test_path = ROOT / "tests" / "test_arithmetic_asian_oracle.py"
    result = {
        "schema": "parikh-rayan-arithmetic-asian-oracle-v1",
        "generated_date": date.today().isoformat(),
        "command": "python scripts/validate_arithmetic_asian_oracle.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
        },
        "source_sha256": sha256(source_path),
        "validation_test_sha256": sha256(test_path),
        "factorized_baseline_sha256": sha256(factorized_path),
        "construction": {
            "price_evolution": "shock-selected fixed-point constant multiplication",
            "geometric_control": "direct product of weighted shock-selected factors",
            "payoffs": "ripple-carry subtraction, sign test, and conditional copy",
            "residual": "reversible subtraction with directed-rounding AM-GM certificate",
            "probability_transduction": "uniform threshold and reversible comparator",
            "allowed_primitives": ["h", "x", "cx", "ccx"],
            "qrom_rows": 0,
            "arbitrary_rotations": 0,
        },
        "small_executable_validation": {
            "dates": small_spec.n_dates,
            "qubits": small_oracle.circuit.num_qubits,
            "reference_objective_probability": small_reference.objective_probability,
            "circuit_objective_probability": small_probability,
            "absolute_probability_error": abs(
                small_probability - small_reference.objective_probability
            ),
            "expected_work_hamming_weight": small_work_hamming,
            "roundtrip_hamming_weight_upper_bound": small_roundtrip,
            "reference_arithmetic_payoff_undiscounted": (
                small_reference.arithmetic_payoff_undiscounted
            ),
            "reference_geometric_payoff_undiscounted": (
                small_reference.geometric_payoff_undiscounted
            ),
            "reference_residual_payoff_undiscounted": (
                small_reference.residual_payoff_undiscounted
            ),
            "minimum_residual_numerator": small_reference.minimum_residual_numerator,
            "maximum_residual_numerator": small_reference.maximum_residual_numerator,
            "executable_primitive_counts": small_actual_counts.as_dict(),
            "counts_match_compositional_estimator": True,
        },
        "daily_252_model": {
            "dates": 252,
            "shock_distribution": "uniform binary +/-1",
            "price_units_per_dollar": production_spec.price_scale,
            "multiplier_fraction_bits": production_model.multiplier_fraction_bits,
            "value_bits": production_model.value_bits,
            "multiplier_bits": production_model.multiplier_bits,
            "product_bits": production_model.product_bits,
            "total_bits": production_model.total_bits,
            "residual_bits": production_model.residual_bits,
            "threshold_bits": production_model.threshold_bits,
            "requested_residual_cap_numerator": (
                production_model.requested_residual_cap_numerator
            ),
            "normalization_numerator": production_model.normalization_numerator,
            "geometric_dp_peak_states": production_model.geometric_dp_peak_states,
            "encoded_geometric_control_undiscounted": (
                production_model.geometric_control_undiscounted
            ),
            "exact_binary_geometric_control_undiscounted": exact_binary_geometric,
            "exact_binary_geometric_dp_states": exact_binary_states,
            "continuous_geometric_control_undiscounted": (
                continuous_geometric_undiscounted
            ),
            "continuous_arithmetic_qcv_reference": continuous_reference,
            "rounding_bias_study": rounding,
            "continuous_price_comparison": {
                "predeclared_absolute_target": PRICE_ACCURACY_TARGET,
                "binary_shock_discretization_error": binary_error,
                "binary_shock_combined_standard_error": binary_combined_se,
                "binary_shock_meets_target_at_95_percent": bool(
                    abs(binary_error) + 1.96 * binary_combined_se
                    <= PRICE_ACCURACY_TARGET
                ),
                "encoded_oracle_error_including_rounding_and_clipping": encoded_error,
                "encoded_oracle_combined_standard_error": encoded_combined_se,
                "encoded_oracle_meets_target_at_95_percent": bool(
                    abs(encoded_error) + 1.96 * encoded_combined_se
                    <= PRICE_ACCURACY_TARGET
                ),
            },
            "paired_precision_sweep": sweep,
            "accuracy_qualified_candidate": {
                "price_scale": selected_model.spec.price_scale,
                "multiplier_fraction_bits": selected_model.multiplier_fraction_bits,
                "geometric_control_method": (
                    "paired binary-path calibration against the exact unrounded "
                    "binary geometric control; its standard error is reported below"
                ),
                "geometric_control_undiscounted": (
                    selected_model.geometric_control_undiscounted
                ),
                "geometric_control_standard_error_undiscounted": (
                    selected_model.geometric_control_standard_error_undiscounted
                ),
                "continuous_price_error": selected_precision[
                    "estimated_continuous_price_error"
                ],
                "continuous_price_error_standard_error": selected_precision[
                    "conservative_combined_standard_error"
                ],
                "meets_predeclared_target_at_95_percent": selected_precision[
                    "meets_target_at_95_percent"
                ],
                "resources": selected_resources.as_dict(),
            },
        },
        "complete_resources": production_resources.as_dict(),
        "factorized_baseline_comparison": {
            "lookup_rows": factorized_252["factorized_lookup_rows"],
            "controlled_rotations": factorized_252[
                "factorized_controlled_rotations"
            ],
            "arithmetic_lookup_rows": production_resources.qrom_rows,
            "arithmetic_arbitrary_rotations": production_resources.arbitrary_rotations,
        },
        "evidence_states": {
            "small_executable_A": "VERIFIED",
            "small_module_truth_tables": "VERIFIED by tests/test_arithmetic_asian_oracle.py",
            "252_date_gate_count": "SUPPORTED: compositional formulas match the executable modules on the validated small instance; the 252-date circuit was not materialized",
            "252_date_monolithic_build_or_simulation": "NOT-ASSESSABLE; intentionally not materialized",
            "continuous_model_accuracy_current_config": "CONTRADICTED for the predeclared $0.01 target on the stated daily Black--Scholes instance",
            "continuous_model_accuracy_refined_candidate": "SUPPORTED for the predeclared $0.01 target on the stated daily Black--Scholes instance by an independent unit-QCV Monte Carlo reference and paired finite-model bias study; this is not a uniform error bound",
            "physical_runtime_or_quantum_advantage": "UNVERIFIED",
        },
        "limitations": [
            "The complete count is for the declared uniform binary-shock finite model, not continuous GBM.",
            "The out-of-place construction retains intermediate product registers and is complete but not qubit-optimal.",
            "Shock-selected multiplication scales with the number of shock basis states; only the binary instance is resource-counted here.",
            "The accuracy-qualified candidate restores its finite geometric control from a paired classical calibration; its reported standard error must be propagated and the calibration must be rerun after any model or precision change.",
            "The $0.01 accuracy result is for one daily Black--Scholes parameter set and is not a uniform guarantee over strikes, volatilities, maturities, or monitoring schedules.",
            "The Q reflection count uses a clean-ancilla AND ladder and therefore reports both A-only and Q-with-reflection-ancilla widths.",
            "No physical code distance, magic-state factory, routing, cycle time, runtime, or advantage claim is inferred.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
