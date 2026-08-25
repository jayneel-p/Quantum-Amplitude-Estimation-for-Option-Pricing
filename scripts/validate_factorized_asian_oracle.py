#!/usr/bin/env python3
"""Validate the factorized QCV residual oracle and report its scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qc_option_pricing.quantum.asian_oracle import (
    AsianGridSpec,
    build_asian_model,
    build_asian_oracle,
    enumerate_encoded_asian,
    estimate_asian_oracle_resources,
    inverse_roundtrip_leakage_from_mps,
    objective_probability_from_mps,
)


def small_validation() -> dict:
    spec = AsianGridSpec(
        n_dates=2,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.4, 0.6),
        s0=2.0,
        strike=1.0,
        rate=0.0,
        volatility=0.3,
        maturity=1.0,
        shock_scale=1,
        price_scale=1,
    )
    reference = enumerate_encoded_asian(spec)
    rectangular = build_asian_oracle(spec, "qcv")
    factorized = build_asian_oracle(
        spec, "qcv", residual_method="factorized_arithmetic"
    )
    rectangular_probability, rectangular_leakage = objective_probability_from_mps(rectangular)
    factorized_probability, factorized_leakage = objective_probability_from_mps(factorized)
    return {
        "reference_probability": reference.qcv_objective_probability,
        "rectangular_probability": rectangular_probability,
        "factorized_probability": factorized_probability,
        "factorized_absolute_error": abs(
            factorized_probability - reference.qcv_objective_probability
        ),
        "rectangular_factorized_difference": abs(
            rectangular_probability - factorized_probability
        ),
        "rectangular_work_leakage": rectangular_leakage,
        "factorized_work_leakage": factorized_leakage,
        "factorized_roundtrip_leakage": inverse_roundtrip_leakage_from_mps(factorized),
        "rectangular_qubits": rectangular.circuit.num_qubits,
        "factorized_qubits": factorized.circuit.num_qubits,
        "rectangular_lookup_rows_built": rectangular.lookup_rows,
        "factorized_lookup_rows_built": factorized.lookup_rows,
        "rectangular_lookup_bit_toggles_built": rectangular.lookup_bit_toggles,
        "factorized_lookup_bit_toggles_built": factorized.lookup_bit_toggles,
        "controlled_rotations": factorized.controlled_rotations,
    }


def scaling_rows() -> list[dict]:
    rows = []
    for dates in (2, 4, 8, 16, 32, 64, 128, 252):
        spec = AsianGridSpec(
            n_dates=dates,
            shock_points=(-1.0, 1.0),
            shock_probabilities=(0.5, 0.5),
            s0=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            shock_scale=1,
            price_scale=1,
        )
        model = build_asian_model(spec)
        rectangular = estimate_asian_oracle_resources(model, "qcv")
        factorized = estimate_asian_oracle_resources(
            model, "qcv", residual_method="factorized_arithmetic"
        )
        rows.append({
            "dates": dates,
            "rectangular_qubits": rectangular.total_qubits,
            "factorized_qubits": factorized.total_qubits,
            "rectangular_lookup_row_upper_bound": rectangular.lookup_rows,
            "factorized_lookup_rows": factorized.lookup_rows,
            "row_reduction_factor": rectangular.lookup_rows / factorized.lookup_rows,
            "factorized_lookup_bit_toggles": factorized.lookup_bit_toggles,
            "factorized_modular_adders": factorized.modular_adders,
            "factorized_controlled_rotations": factorized.controlled_rotations,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v7/factorized_asian_oracle_validation.json"),
    )
    args = parser.parse_args()
    payload = {
        "schema": "parikh-rayan-factorized-asian-oracle-v1",
        "method": (
            "Separate total->arithmetic-payoff and weighted-shock->geometric-payoff "
            "QROMs, reversible modular subtraction, and existing capped payoff rotation"
        ),
        "small_encoded_validation": small_validation(),
        "coarse_scaling": scaling_rows(),
        "limitations": [
            "The per-fixing price exponential remains a one-dimensional QROM.",
            "The payoff transduction still uses one controlled rotation per residual-register value.",
            "Counts are logical pre-synthesis structures, not Clifford+T counts.",
            "The 252-date circuit is estimated, not constructed or simulated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
