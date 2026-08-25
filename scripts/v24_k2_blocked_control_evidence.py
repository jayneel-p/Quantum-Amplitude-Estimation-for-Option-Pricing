#!/usr/bin/env python3
"""Evidence for the plain k=2 blocked geometric control at 252 dates.

The published ladder in ``telescoping_asian_ladder`` splits the price into two
estimated increments, ``C0 + (C1 - C0) + (H - C1)``, and pays a
``(sqrt(a1)+sqrt(a2))**2`` allocation penalty for using two amplitudes.  This
script instead scopes the *single-amplitude* two-block control: estimate
``H - C1`` alone and restore ``C1`` classically, exactly as the shipped k=1
oracle estimates ``H - C0`` and restores ``C0``.

Three questions are answered, each fail-closed:

1. Resources.  Build and count the ``blocked_to_target`` oracle at the
   production configuration and compare its per-query cost and implemented
   amplitude scale against the raw and k=1 arithmetic oracles.
2. Rounding.  Re-certify, in eighty-digit decimal at 252 dates, that both
   block exponential chains round down, so the encoded two-block control never
   exceeds its exact value and ``H - C1 >= 0`` holds in fixed point.
3. Restoration.  Compute the exact finite-grid ``C1 = E[(B_2 - K)^+]`` with a
   two-dimensional dynamic program over the joint reachable set of the two
   block weighted sums, and cross-check it against an independent estimate.

Nothing here executes an amplitude-estimation schedule, and every resource
figure is logical rather than physical.

Run::

    .venv/bin/python scripts/v24_k2_blocked_control_evidence.py \
        --output results/v24/k2_blocked_control_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from qc_option_pricing.quantum.arithmetic_asian_oracle import (  # noqa: E402
    build_arithmetic_asian_model,
    estimate_arithmetic_asian_resources,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec  # noqa: E402
from qc_option_pricing.quantum.telescoping_asian_ladder import (  # noqa: E402
    _encoded_block_value,
    build_k2_ladder_model,
    estimate_k2_ladder_resources,
    k2_ladder_path_values,
)

N_DATES = 252
PRICE_SCALE = 16_384
FRACTION_BITS = 30
# Caps selected on seeded binary-shock samples by the k=2 ladder pilot
# (results/v23/k2_ladder_pilot.json) against a $0.001 discounted budget.
RAW_CAP_DOLLARS = 54.885316394624255
K1_CAP_DOLLARS = 2.7676527235243054
K2_CAP_DOLLARS = 0.6808057512555804
CROSS_CHECK_SEED = 20_260_801
CROSS_CHECK_PATHS = 400_000


def _spec(geometric_leg: str) -> AsianGridSpec:
    return AsianGridSpec(
        n_dates=N_DATES,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        shock_scale=1,
        price_scale=PRICE_SCALE,
        geometric_leg=geometric_leg,
    )


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- resources


def _arithmetic_arm(geometric_leg: str, cap_dollars: float) -> dict:
    model = build_arithmetic_asian_model(
        _spec_with_cap(geometric_leg, cap_dollars),
        multiplier_fraction_bits=FRACTION_BITS,
    )
    estimate = estimate_arithmetic_asian_resources(model)
    return _arm_record(
        label=f"arithmetic_{geometric_leg}",
        cap_dollars=cap_dollars,
        cap_numerator=model.requested_residual_cap_numerator,
        threshold_bits=model.threshold_bits,
        normalization_numerator=model.normalization_numerator,
        amplitude_scale_dollars=model.normalization_dollars,
        estimate=estimate,
    )


def _spec_with_cap(geometric_leg: str, cap_dollars: float) -> AsianGridSpec:
    base = dict(
        n_dates=N_DATES,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        shock_scale=1,
        price_scale=PRICE_SCALE,
        geometric_leg=geometric_leg,
    )
    if geometric_leg == "none":
        base["payoff_cap"] = cap_dollars
    else:
        base["residual_payoff_cap"] = cap_dollars
    return AsianGridSpec(**base)


def _arm_record(
    *,
    label: str,
    cap_dollars: float,
    cap_numerator: int,
    threshold_bits: int,
    normalization_numerator: int,
    amplitude_scale_dollars: float,
    estimate,
) -> dict:
    return {
        "label": label,
        "cap_dollars": cap_dollars,
        "cap_numerator": cap_numerator,
        "threshold_bits": threshold_bits,
        "normalization_numerator": normalization_numerator,
        "amplitude_scale_dollars": amplitude_scale_dollars,
        "register_fill_fraction": cap_numerator / normalization_numerator,
        "a_t": estimate.a_counts.t,
        "q_t": estimate.q_counts.t,
        "a_qubits": estimate.a_qubits,
        "q_qubits": estimate.q_qubits_with_clean_reflection_ladder,
        "a_toffoli": estimate.a_counts.ccx,
    }


def _k2_arm() -> dict:
    model = build_k2_ladder_model(
        _spec("collapsed"),
        "blocked_to_target",
        multiplier_fraction_bits=FRACTION_BITS,
        increment_cap_dollars=K2_CAP_DOLLARS,
    )
    estimate = estimate_k2_ladder_resources(model)
    record = _arm_record(
        label="ladder_blocked_to_target",
        cap_dollars=K2_CAP_DOLLARS,
        cap_numerator=model.requested_cap_numerator,
        threshold_bits=model.threshold_bits,
        normalization_numerator=model.normalization_numerator,
        amplitude_scale_dollars=model.normalization_dollars,
        estimate=estimate,
    )
    record["model"] = model
    return record


# ----------------------------------------------------------------- rounding


def _certify_block_rounding(model) -> dict:
    """Re-derive the downward-rounding certificate at eighty digits."""

    shared = model.shared
    spec = shared.spec
    partition = shared.partition_map[2]
    n = spec.n_dates
    low, high = spec.shock_points
    blocks = []
    with localcontext() as context:
        context.prec = 80
        d = lambda value: Decimal(str(value))
        dt = d(spec.maturity) / Decimal(n)
        drift = d(spec.rate) - d(spec.volatility) ** 2 / Decimal(2)
        diffusion = d(spec.volatility) * dt.sqrt()
        for block in partition.blocks:
            m = block.stop_date - block.start_date
            average_fixing_index = Decimal(
                block.start_date + block.stop_date + 1
            ) / Decimal(2)
            worst_overshoot = Decimal(-1)
            worst_deficit = Decimal(0)
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
                        block, weighted_sum, shared.multiplier_fraction_bits
                    )
                )
                # Downward rounding demands encoded <= exact on every value.
                worst_overshoot = max(worst_overshoot, encoded - exact)
                worst_deficit = max(worst_deficit, exact - encoded)
            blocks.append(
                {
                    "block_index": block.block_index,
                    "dates": [block.start_date, block.stop_date],
                    "weighted_sum_values": block.shock_weight_sum + 1,
                    "shock_weight_bits": block.shock_weight_bits,
                    "chain_multiplications": len(block.chain_factors),
                    "worst_upward_violation_units": float(worst_overshoot),
                    "worst_downward_deficit_units": float(worst_deficit),
                    "recorded_error_bound_units": block.rounding_error_bound_units,
                    "rounds_down_everywhere": bool(worst_overshoot <= 0),
                }
            )
    return {
        "strike_adjustment_units": partition.strike_adjustment_units,
        "blocks": blocks,
        "reachable_pairs": (
            (partition.blocks[0].shock_weight_sum + 1)
            * (partition.blocks[1].shock_weight_sum + 1)
        ),
    }


def _sampled_ordering(model, paths: int, seed: int) -> dict:
    """Check H - C1 >= 0 and the telescoping identity on random paths."""

    rng = np.random.default_rng(seed)
    worst = math.inf
    checked = 0
    for _ in range(paths):
        digits = rng.integers(0, 2, size=N_DATES).tolist()
        values = k2_ladder_path_values(model.shared, digits)
        worst = min(worst, values.blocked_to_target)
        checked += 1
    return {
        "paths_checked": checked,
        "minimum_blocked_to_target_numerator": int(worst),
        "nonnegative_on_every_sampled_path": bool(worst >= 0),
    }


# -------------------------------------------------------------- restoration


def _weighted_sum_counts(weights: tuple[int, ...]) -> np.ndarray:
    """Exact integer path counts for one block's weighted shock sum."""

    total = sum(weights)
    counts = np.zeros(total + 1, dtype=object)
    counts[0] = 1
    reach = 0
    for weight in weights:
        if weight == 0:
            continue
        counts[weight : reach + weight + 1] += counts[0 : reach + 1]
        reach += weight
    return counts


def _joint_first_half_counts(
    weights_a: tuple[int, ...], half: int
) -> np.ndarray:
    """Counts of (s_a, number of up-moves) over the first-half shocks.

    The second block's weighted sum is ``126 * up_moves + u``, where ``u``
    comes from the second half alone, so the joint law of the two block sums
    factors through the first-half up-move count.
    """

    sum_a = sum(weights_a[:half])
    table = np.zeros((sum_a + 1, half + 1), dtype=object)
    table[0, 0] = 1
    reach = 0
    for date in range(half):
        weight = weights_a[date]
        new = np.zeros_like(table)
        # down move: neither the weighted sum nor the up-move count changes
        new[0 : reach + 1, : date + 1] += table[0 : reach + 1, : date + 1]
        # up move: add this date's weight and one up-move
        new[weight : reach + weight + 1, 1 : date + 2] += table[
            0 : reach + 1, : date + 1
        ]
        table = new
        reach += weight
    return table


def _exact_blocked_control(model) -> dict:
    """Exact finite-grid E[(B_2 - K)^+] by a two-dimensional dynamic program."""

    shared = model.shared
    spec = shared.spec
    partition = shared.partition_map[2]
    block_a, block_b = partition.blocks
    n = spec.n_dates
    half = n // 2
    fraction_bits = shared.multiplier_fraction_bits

    # Second block's weight vector is 126 on every first-half date and
    # (252 - d) on its own dates, so split it into the shared count and u.
    shared_weight = block_b.shock_weights[0]
    if any(w != shared_weight for w in block_b.shock_weights[:half]):
        raise AssertionError("second block does not carry a constant first-half weight")
    u_weights = tuple(block_b.shock_weights[half:])

    joint_a = _joint_first_half_counts(block_a.shock_weights, half)
    counts_u = _weighted_sum_counts(u_weights)
    if int(joint_a.sum()) != 1 << half or int(counts_u.sum()) != 1 << half:
        raise AssertionError("dynamic program lost probability mass")

    g_a = np.array(
        [
            _encoded_block_value(block_a, s, fraction_bits)
            for s in range(block_a.shock_weight_sum + 1)
        ],
        dtype=object,
    )
    g_b = np.array(
        [
            _encoded_block_value(block_b, s, fraction_bits)
            for s in range(block_b.shock_weight_sum + 1)
        ],
        dtype=object,
    )

    strike_term = partition.block_count * (
        spec.strike_integer + partition.strike_adjustment_units
    )
    u_values = np.nonzero(counts_u)[0]
    u_counts = counts_u[u_values]

    numerator = 0
    for up_moves in range(half + 1):
        column = joint_a[:, up_moves]
        active = np.nonzero(column)[0]
        if active.size == 0:
            continue
        s_b = shared_weight * up_moves + u_values
        gb = g_b[s_b]
        # Sort once so each s_a becomes a tail lookup rather than a full scan.
        order = np.argsort([int(v) for v in gb])
        gb_sorted = [int(gb[i]) for i in order]
        w_sorted = [int(u_counts[i]) for i in order]
        prefix_w = [0]
        prefix_gw = [0]
        for value, weight in zip(gb_sorted, w_sorted):
            prefix_w.append(prefix_w[-1] + weight)
            prefix_gw.append(prefix_gw[-1] + weight * value)
        total_w, total_gw = prefix_w[-1], prefix_gw[-1]
        for s_a in active:
            base = int(g_a[s_a]) - strike_term
            # payoff = max(base + gb, 0), summed against u counts
            cut = -base
            idx = int(np.searchsorted(gb_sorted, cut, side="left"))
            tail_w = total_w - prefix_w[idx]
            tail_gw = total_gw - prefix_gw[idx]
            numerator += int(column[s_a]) * (base * tail_w + tail_gw)

    # The accumulated numerator is sum over paths of
    # max(g_a + g_b - 2K, 0) = block_count * (B_2 - K)^+ in price units,
    # so the block count divides out alongside the path count and price scale.
    denominator = partition.block_count * (1 << n) * spec.price_scale
    control = Decimal(numerator) / Decimal(denominator)
    return {
        "method": "exact two-dimensional dynamic program over both block weighted sums",
        "first_block_sum_values": int(block_a.shock_weight_sum + 1),
        "second_block_sum_values": int(block_b.shock_weight_sum + 1),
        "reachable_pairs": int(
            (block_a.shock_weight_sum + 1) * (block_b.shock_weight_sum + 1)
        ),
        "control_undiscounted": float(control),
        "control_discounted": float(
            control * Decimal(str(math.exp(-spec.rate * spec.maturity)))
        ),
    }


def _cross_check_control(model, paths: int, seed: int) -> dict:
    """Independent Monte Carlo estimate of the same encoded control."""

    shared = model.shared
    spec = shared.spec
    partition = shared.partition_map[2]
    block_a, block_b = partition.blocks
    rng = np.random.default_rng(seed)
    half = spec.n_dates // 2
    weights_a = np.array(block_a.shock_weights, dtype=np.int64)
    weights_b = np.array(block_b.shock_weights, dtype=np.int64)
    strike_term = partition.block_count * (
        spec.strike_integer + partition.strike_adjustment_units
    )
    fraction_bits = shared.multiplier_fraction_bits

    total = 0.0
    total_square = 0.0
    chunk = 20_000
    done = 0
    while done < paths:
        size = min(chunk, paths - done)
        digits = rng.integers(0, 2, size=(size, spec.n_dates))
        s_a = digits @ weights_a
        s_b = digits @ weights_b
        for i in range(size):
            payoff = max(
                _encoded_block_value(block_a, int(s_a[i]), fraction_bits)
                + _encoded_block_value(block_b, int(s_b[i]), fraction_bits)
                - strike_term,
                0,
            ) / (partition.block_count * spec.price_scale)
            total += payoff
            total_square += payoff * payoff
        done += size
    mean = total / paths
    variance = max(0.0, (total_square - paths * mean * mean) / (paths - 1))
    return {
        "paths": paths,
        "seed": seed,
        "control_undiscounted": mean,
        "standard_error": math.sqrt(variance / paths),
    }


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=_REPO / "results" / "v24" / "k2_blocked_control_evidence.json"
    )
    parser.add_argument("--cross-check-paths", type=int, default=CROSS_CHECK_PATHS)
    parser.add_argument("--ordering-paths", type=int, default=200)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("counting the raw, k=1, and k=2 oracles at 252 dates ...", flush=True)
    raw = _arithmetic_arm("none", RAW_CAP_DOLLARS)
    k1 = _arithmetic_arm("collapsed", K1_CAP_DOLLARS)
    k2 = _k2_arm()
    k2_model = k2.pop("model")

    scale_ratio_k1 = raw["normalization_numerator"] / k1["normalization_numerator"]
    scale_ratio_k2 = raw["normalization_numerator"] / k2["normalization_numerator"]
    overhead_k1 = k1["q_t"] / raw["q_t"]
    overhead_k2 = k2["q_t"] / raw["q_t"]

    print("re-certifying the block rounding directions at eighty digits ...", flush=True)
    rounding = _certify_block_rounding(k2_model)
    ordering = _sampled_ordering(k2_model, args.ordering_paths, CROSS_CHECK_SEED)

    print("running the exact two-dimensional control dynamic program ...", flush=True)
    exact_control = _exact_blocked_control(k2_model)
    print("cross-checking the control by independent sampling ...", flush=True)
    sampled_control = _cross_check_control(
        k2_model, args.cross_check_paths, CROSS_CHECK_SEED + 1
    )
    z = abs(
        exact_control["control_undiscounted"] - sampled_control["control_undiscounted"]
    ) / max(sampled_control["standard_error"], 1e-30)

    gates = [
        {
            "name": "k=2 threshold register is narrower than k=1",
            "passed": k2["threshold_bits"] < k1["threshold_bits"],
            "observed": [k1["threshold_bits"], k2["threshold_bits"]],
        },
        {
            "name": "implemented ratios are exact powers of two",
            "passed": bool(
                float(scale_ratio_k1).is_integer()
                and float(scale_ratio_k2).is_integer()
                and (int(scale_ratio_k1) & (int(scale_ratio_k1) - 1)) == 0
                and (int(scale_ratio_k2) & (int(scale_ratio_k2) - 1)) == 0
            ),
            "observed": [scale_ratio_k1, scale_ratio_k2],
        },
        {
            "name": "k=1 arm reproduces the published 16-fold implemented ratio",
            "passed": abs(scale_ratio_k1 - 16.0) < 1e-9,
            "observed": scale_ratio_k1,
        },
        {
            "name": "k=1 arm reproduces the published 1.068 per-query overhead",
            "passed": abs(overhead_k1 - 1.068) < 5e-3,
            "observed": overhead_k1,
        },
        {
            "name": "both block chains round down on every reachable value",
            "passed": all(b["rounds_down_everywhere"] for b in rounding["blocks"]),
            "observed": [b["worst_upward_violation_units"] for b in rounding["blocks"]],
        },
        {
            "name": "the k=2 partition needs no strike shift",
            "passed": rounding["strike_adjustment_units"] == 0,
            "observed": rounding["strike_adjustment_units"],
        },
        {
            "name": "H - C1 is nonnegative on every sampled path",
            "passed": ordering["nonnegative_on_every_sampled_path"],
            "observed": ordering["minimum_blocked_to_target_numerator"],
        },
        {
            "name": "exact control agrees with independent sampling",
            "passed": bool(z <= 3.0),
            "observed_z": z,
        },
    ]

    record = {
        "schema_version": "k2-blocked-control-evidence-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "claim_tested": (
            "A single-amplitude two-block geometric control, with its price "
            "restored classically, reaches a matched price error with fewer "
            "leading T gates than the shipped one-block oracle."
        ),
        "configuration": {
            "n_dates": N_DATES,
            "price_scale": PRICE_SCALE,
            "multiplier_fraction_bits": FRACTION_BITS,
            "caps_selected_by": "results/v23/k2_ladder_pilot.json, $0.001 discounted budget on binary-shock paths",
        },
        "arms": {"raw": raw, "k1": k1, "k2": k2},
        "comparison": {
            "implemented_scale_ratio_k1": scale_ratio_k1,
            "implemented_scale_ratio_k2": scale_ratio_k2,
            "per_query_t_overhead_k1": overhead_k1,
            "per_query_t_overhead_k2": overhead_k2,
            "leading_t_gain_k1": scale_ratio_k1 / overhead_k1,
            "leading_t_gain_k2": scale_ratio_k2 / overhead_k2,
            "k2_over_k1_leading_gain": (scale_ratio_k2 / overhead_k2)
            / (scale_ratio_k1 / overhead_k1),
            "query_model": "queries taken proportional to the implemented amplitude scale; no schedule executed",
        },
        "rounding_certificate": rounding,
        "sampled_ordering": ordering,
        "exact_control_restoration": exact_control,
        "control_cross_check": sampled_control,
        "control_agreement_z": z,
        "limitations": [
            "No amplitude-estimation schedule was executed on any 252-date oracle.",
            "Resource counts are logical and use the exact seven-T Toffoli convention.",
            "Caps are selected on seeded binary-shock samples, not certified from a continuous tail bound.",
            "The k=2 oracle was counted compositionally; it was not transpiled at 252 dates here.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "inputs": {
            "scripts/v24_k2_blocked_control_evidence.py": {
                "sha256": _sha256(Path(__file__).resolve())
            },
            "src/qc_option_pricing/quantum/telescoping_asian_ladder.py": {
                "sha256": _sha256(
                    _REPO / "src/qc_option_pricing/quantum/telescoping_asian_ladder.py"
                )
            },
            "src/qc_option_pricing/quantum/arithmetic_asian_oracle.py": {
                "sha256": _sha256(
                    _REPO / "src/qc_option_pricing/quantum/arithmetic_asian_oracle.py"
                )
            },
        },
        "gates": gates,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")

    for gate in gates:
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] {gate['name']}")
    print()
    for name, arm in (("raw", raw), ("k=1", k1), ("k=2", k2)):
        print(
            f"{name:4s} cap=${arm['cap_dollars']:9.4f} bits={arm['threshold_bits']:2d} "
            f"scale=${arm['amplitude_scale_dollars']:8.4f} fill={arm['register_fill_fraction']:.1%} "
            f"A_T={arm['a_t']:>12,} Q_T={arm['q_t']:>12,} A_qubits={arm['a_qubits']:>6,}"
        )
    print()
    print(f"k=1: implemented {scale_ratio_k1:.0f}x / overhead {overhead_k1:.4f} "
          f"= {scale_ratio_k1 / overhead_k1:.2f}x leading T")
    print(f"k=2: implemented {scale_ratio_k2:.0f}x / overhead {overhead_k2:.4f} "
          f"= {scale_ratio_k2 / overhead_k2:.2f}x leading T")
    print(f"k=2 improves on k=1 by {record['comparison']['k2_over_k1_leading_gain']:.2f}x")
    print()
    print(f"exact C1 = {exact_control['control_undiscounted']:.9f} undiscounted "
          f"over {exact_control['reachable_pairs']:,} reachable pairs")
    print(f"sampled  = {sampled_control['control_undiscounted']:.9f} "
          f"+- {sampled_control['standard_error']:.9f}  (|z| = {z:.2f})")
    print(f"wrote {args.output}")
    return 0 if all(gate["passed"] for gate in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
