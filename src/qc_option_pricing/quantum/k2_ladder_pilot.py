"""Reproducible production-scale pilot for the two-level Asian ladder.

The executable oracles live in :mod:`telescoping_asian_ladder`.  This module
supplies the separate evidence path used to decide whether the extra level is
promising at the paper's 252-date binary-shock specification:

* select integer clipping caps on one seeded sample;
* evaluate their clipping losses on an independent seeded sample; and
* compare ``(sqrt(a1) + sqrt(a2))**2`` with the direct residual coefficient,
  using ``a = implemented amplitude scale * T gates in one Grover query``.

This is a logical-resource and leading-order work comparison.  It is not an
executed amplitude-estimation schedule, a physical-resource estimate, or a
continuous-Black--Scholes tail certificate.

Run, for example::

    python -m qc_option_pricing.quantum.k2_ladder_pilot \
        --paths 2000000 --output results/v23/k2_ladder_pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from qc_option_pricing.quantum.asian_oracle import AsianGridSpec
from qc_option_pricing.quantum.telescoping_asian_ladder import (
    BlockGeometricModel,
    K2LadderIncrementModel,
    K2LadderSharedModel,
    build_k2_ladder_model,
    estimate_k2_ladder_resources,
)


N_DATES = 252
PRICE_SCALE = 16_384
MULTIPLIER_FRACTION_BITS = 30
TOTAL_CLIPPING_BUDGET = 1.0e-3
CAP_SELECTION_Z = 1.96
SELECTION_SEED = 20_260_730
EVALUATION_SEED = 20_260_731


@dataclass(frozen=True)
class IncrementSample:
    coarse_to_blocked: np.ndarray
    blocked_to_target: np.ndarray
    coarse_to_target: np.ndarray
    uncorrected_coarse_to_blocked: np.ndarray


@dataclass(frozen=True)
class _SortedCapSample:
    ordered: np.ndarray
    prefix_sum: np.ndarray
    prefix_square_sum: np.ndarray

    @classmethod
    def from_values(cls, values: np.ndarray) -> "_SortedCapSample":
        ordered = np.sort(values)
        values_float = ordered.astype(float)
        return cls(
            ordered=ordered,
            prefix_sum=np.concatenate(([0.0], np.cumsum(values_float))),
            prefix_square_sum=np.concatenate(
                ([0.0], np.cumsum(values_float * values_float))
            ),
        )

    def clipped_mean_and_standard_error(self, cap: int) -> tuple[float, float]:
        count = self.ordered.size
        start = int(np.searchsorted(self.ordered, cap, side="right"))
        tail_count = count - start
        if tail_count == 0:
            return 0.0, 0.0
        tail_sum = self.prefix_sum[-1] - self.prefix_sum[start]
        tail_square_sum = self.prefix_square_sum[-1] - self.prefix_square_sum[start]
        clipped_sum = tail_sum - cap * tail_count
        clipped_square_sum = (
            tail_square_sum - 2.0 * cap * tail_sum + cap * cap * tail_count
        )
        mean = clipped_sum / count
        variance = max(
            0.0,
            (clipped_square_sum - clipped_sum * clipped_sum / count) / (count - 1),
        )
        return mean, math.sqrt(variance / count)


def production_spec() -> AsianGridSpec:
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
        geometric_leg="collapsed",
    )


def _encoded_block_values(
    block: BlockGeometricModel,
    weighted_sum: np.ndarray,
    fraction_bits: int,
) -> np.ndarray:
    values = np.full(weighted_sum.size, block.initial_geometric, dtype=np.int64)
    for bit, factor in enumerate(block.chain_factors):
        selected = ((weighted_sum >> bit) & 1).astype(bool)
        values = np.where(selected, (values * np.int64(factor)) >> fraction_bits, values)
    return values


def sample_encoded_increments(
    shared: K2LadderSharedModel,
    *,
    paths: int,
    seed: int,
) -> IncrementSample:
    """Sample the exact integer recurrences used by the reversible oracles."""

    if paths < 2:
        raise ValueError("paths must be at least two")
    spec = shared.spec
    rng = np.random.default_rng(seed)
    price = np.full(paths, shared.initial_price, dtype=np.int64)
    total = np.zeros(paths, dtype=np.int64)
    partitions = shared.partition_map
    blocks = (partitions[1].blocks[0], *partitions[2].blocks)
    weighted = [np.zeros(paths, dtype=np.int64) for _ in blocks]
    factor_scale = np.int64(1 << shared.multiplier_fraction_bits)
    price_factors = np.asarray(shared.price_factors, dtype=np.int64)

    largest_factor = max(shared.price_factors, default=0)
    if shared.maximum_prices[-1] * largest_factor >= np.iinfo(np.int64).max:
        raise OverflowError("production sampling would overflow int64 price products")

    for date in range(spec.n_dates):
        digits = rng.integers(0, 2, size=paths, dtype=np.int8)
        price = (
            price * np.take(price_factors, digits)
            + factor_scale
            - 1
        ) >> shared.multiplier_fraction_bits
        total += price
        digits64 = digits.astype(np.int64)
        for accumulator, block in zip(weighted, blocks):
            weight = block.shock_weights[date]
            if weight:
                accumulator += digits64 * np.int64(weight)

    encoded_blocks = tuple(
        _encoded_block_values(block, accumulator, shared.multiplier_fraction_bits)
        for block, accumulator in zip(blocks, weighted)
    )
    strike = spec.strike_integer
    coarse = spec.n_dates * np.maximum(
        encoded_blocks[0] - strike - shared.coarse_strike_adjustment_units,
        0,
    )
    uncorrected_coarse = spec.n_dates * np.maximum(encoded_blocks[0] - strike, 0)
    blocked = (spec.n_dates // 2) * np.maximum(
        encoded_blocks[1] + encoded_blocks[2] - 2 * strike,
        0,
    )
    target = np.maximum(total - spec.n_dates * strike, 0)
    first = blocked - coarse
    second = target - blocked
    direct = target - coarse
    if int(first.min()) < 0:
        raise AssertionError("certified coarse-to-blocked increment became negative")
    if int(second.min()) < 0:
        raise AssertionError("blocked-to-target increment became negative")
    if not np.array_equal(first + second, direct):
        raise AssertionError("sampled finite-grid ladder failed to telescope")
    return IncrementSample(first, second, direct, blocked - uncorrected_coarse)


def _smallest_cap_for_bias(
    sample: _SortedCapSample,
    *,
    discounted_budget: float,
    denominator: int,
    discount: float,
    selection_z: float,
) -> int:
    """Smallest cap whose one-sided sample bound meets the bias budget."""

    def meets(cap: int) -> bool:
        mean, standard_error = sample.clipped_mean_and_standard_error(cap)
        upper = discount * (mean + selection_z * standard_error) / denominator
        return upper <= discounted_budget

    low, high = 1, max(1, int(sample.ordered[-1]))
    while low < high:
        middle = (low + high) // 2
        if meets(middle):
            high = middle
        else:
            low = middle + 1
    if low > 1 and meets(low - 1):
        raise AssertionError("cap selector did not return the smallest integer cap")
    if not meets(low):
        raise AssertionError("cap selector failed its clipping-bias constraint")
    return low


def _resource_row(model: K2LadderIncrementModel) -> dict[str, object]:
    resources = estimate_k2_ladder_resources(model)
    coefficient = model.normalization_dollars * resources.q_counts.t
    return {
        "increment": model.increment,
        "requested_cap_numerator": model.requested_cap_numerator,
        "requested_cap_dollars": model.requested_cap_numerator
        / (model.spec.n_dates * model.spec.price_scale),
        "threshold_bits": model.threshold_bits,
        "implemented_amplitude_scale_dollars": model.normalization_dollars,
        "a_qubits": resources.a_qubits,
        "q_qubits_with_clean_reflection_ladder": (
            resources.q_qubits_with_clean_reflection_ladder
        ),
        "a_counts": resources.a_counts.as_dict(),
        "q_counts": resources.q_counts.as_dict(),
        "leading_work_coefficient_scale_times_q_t": coefficient,
    }


def _clipping_statistics(
    values: np.ndarray,
    *,
    cap: int,
    denominator: int,
    discount: float,
) -> dict[str, object]:
    losses = discount * np.maximum(values - cap, 0).astype(float) / denominator
    standard_error = float(losses.std(ddof=1) / math.sqrt(losses.size))
    return {
        "cap_numerator": cap,
        "cap_dollars": cap / denominator,
        "discounted_loss": float(losses.mean()),
        "standard_error": standard_error,
        "ci95_low": float(losses.mean()) - 1.96 * standard_error,
        "ci95_high": float(losses.mean()) + 1.96 * standard_error,
        "clipped_paths": int(np.count_nonzero(losses)),
        "clipped_fraction": float(np.count_nonzero(losses) / losses.size),
    }


def _git_metadata(root: Path) -> dict[str, object]:
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    return {
        "head": rev.stdout.strip() if rev.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_pilot(
    *,
    paths: int,
    selection_seed: int = SELECTION_SEED,
    evaluation_seed: int = EVALUATION_SEED,
    total_clipping_budget: float = TOTAL_CLIPPING_BUDGET,
    cap_selection_z: float = CAP_SELECTION_Z,
) -> dict[str, object]:
    if not math.isfinite(total_clipping_budget) or total_clipping_budget <= 0.0:
        raise ValueError("total_clipping_budget must be finite and positive")
    if not math.isfinite(cap_selection_z) or cap_selection_z < 0.0:
        raise ValueError("cap_selection_z must be finite and nonnegative")
    spec = production_spec()
    base = build_k2_ladder_model(
        spec,
        "coarse_to_blocked",
        multiplier_fraction_bits=MULTIPLIER_FRACTION_BITS,
    )
    shared = base.shared
    denominator = spec.n_dates * spec.price_scale
    discount = math.exp(-spec.rate * spec.maturity)

    selection = sample_encoded_increments(shared, paths=paths, seed=selection_seed)
    ordered_first = _SortedCapSample.from_values(selection.coarse_to_blocked)
    ordered_second = _SortedCapSample.from_values(selection.blocked_to_target)
    ordered_direct = _SortedCapSample.from_values(selection.coarse_to_target)
    direct_cap = _smallest_cap_for_bias(
        ordered_direct,
        discounted_budget=total_clipping_budget,
        denominator=denominator,
        discount=discount,
        selection_z=cap_selection_z,
    )
    direct_model = build_k2_ladder_model(
        spec,
        "coarse_to_target",
        multiplier_fraction_bits=MULTIPLIER_FRACTION_BITS,
        increment_cap_dollars=direct_cap / denominator,
    )
    direct_resource = _resource_row(direct_model)

    candidates: list[dict[str, object]] = []
    # Keep at least ten percent of the total clipping allowance on each arm;
    # smaller tail budgets are poorly resolved by two million paths.
    for first_fraction in np.linspace(0.10, 0.90, 17):
        first_budget = total_clipping_budget * float(first_fraction)
        second_budget = total_clipping_budget - first_budget
        first_cap = _smallest_cap_for_bias(
            ordered_first,
            discounted_budget=first_budget,
            denominator=denominator,
            discount=discount,
            selection_z=cap_selection_z,
        )
        second_cap = _smallest_cap_for_bias(
            ordered_second,
            discounted_budget=second_budget,
            denominator=denominator,
            discount=discount,
            selection_z=cap_selection_z,
        )
        first_model = build_k2_ladder_model(
            spec,
            "coarse_to_blocked",
            multiplier_fraction_bits=MULTIPLIER_FRACTION_BITS,
            increment_cap_dollars=first_cap / denominator,
        )
        second_model = build_k2_ladder_model(
            spec,
            "blocked_to_target",
            multiplier_fraction_bits=MULTIPLIER_FRACTION_BITS,
            increment_cap_dollars=second_cap / denominator,
        )
        first_resource = _resource_row(first_model)
        second_resource = _resource_row(second_model)
        a1 = float(first_resource["leading_work_coefficient_scale_times_q_t"])
        a2 = float(second_resource["leading_work_coefficient_scale_times_q_t"])
        coefficient = (math.sqrt(a1) + math.sqrt(a2)) ** 2
        candidates.append(
            {
                "first_budget_fraction": float(first_fraction),
                "first_discounted_clipping_budget": first_budget,
                "second_discounted_clipping_budget": second_budget,
                "first_cap_numerator": first_cap,
                "second_cap_numerator": second_cap,
                "first_resource": first_resource,
                "second_resource": second_resource,
                "ladder_leading_work_coefficient": coefficient,
                "optimal_ae_error_fraction_first": math.sqrt(a1)
                / (math.sqrt(a1) + math.sqrt(a2)),
                "optimal_ae_error_fraction_second": math.sqrt(a2)
                / (math.sqrt(a1) + math.sqrt(a2)),
            }
        )
    chosen = min(candidates, key=lambda row: row["ladder_leading_work_coefficient"])
    direct_coefficient = float(
        direct_resource["leading_work_coefficient_scale_times_q_t"]
    )
    ladder_coefficient = float(chosen["ladder_leading_work_coefficient"])

    evaluation = sample_encoded_increments(shared, paths=paths, seed=evaluation_seed)
    first_cap = int(chosen["first_cap_numerator"])
    second_cap = int(chosen["second_cap_numerator"])
    first_evaluation = _clipping_statistics(
        evaluation.coarse_to_blocked,
        cap=first_cap,
        denominator=denominator,
        discount=discount,
    )
    second_evaluation = _clipping_statistics(
        evaluation.blocked_to_target,
        cap=second_cap,
        denominator=denominator,
        discount=discount,
    )
    direct_evaluation = _clipping_statistics(
        evaluation.coarse_to_target,
        cap=direct_cap,
        denominator=denominator,
        discount=discount,
    )
    combined_losses = discount * (
        np.maximum(evaluation.coarse_to_blocked - first_cap, 0)
        + np.maximum(evaluation.blocked_to_target - second_cap, 0)
    ).astype(float) / denominator
    combined_se = float(combined_losses.std(ddof=1) / math.sqrt(paths))

    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).resolve()
    ladder_source = source.with_name("telescoping_asian_ladder.py")
    return {
        "schema": "k2_ladder_pilot.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_tested": (
            "At the implemented N=252 finite grid and a matched $0.001 total "
            "discounted clipping budget, the work-optimised k=2 telescope has "
            "a smaller leading scale-times-query-T coefficient than direct "
            "QAE of H-C0_shifted."
        ),
        "scope_limitations": [
            "finite binary-shock model, not a continuous-model tail certificate",
            "logical Toffoli=7T ledger, not physical resources or runtime",
            "leading 1/epsilon work model, not an executed AE schedule",
            "clipping-budget split selected on a finite grid of 17 allocations",
        ],
        "configuration": {
            "paths_per_arm": paths,
            "selection_seed": selection_seed,
            "evaluation_seed": evaluation_seed,
            "total_discounted_clipping_budget": total_clipping_budget,
            "cap_selection_one_sided_z": cap_selection_z,
            "n_dates": spec.n_dates,
            "price_scale": spec.price_scale,
            "multiplier_fraction_bits": MULTIPLIER_FRACTION_BITS,
            "coarse_strike_adjustment_units": shared.coarse_strike_adjustment_units,
            "coarse_strike_adjustment_dollars": (
                shared.coarse_strike_adjustment_dollars
            ),
            "fine_block_rounding_error_bounds_units": [
                block.rounding_error_bound_units
                for block in shared.partition_map[2].blocks
            ],
        },
        "finite_grid_checks": {
            "selection_minimum_D1": int(selection.coarse_to_blocked.min()),
            "selection_minimum_D2": int(selection.blocked_to_target.min()),
            "selection_uncorrected_minimum_D1": int(
                selection.uncorrected_coarse_to_blocked.min()
            ),
            "selection_uncorrected_negative_D1_paths": int(
                np.count_nonzero(selection.uncorrected_coarse_to_blocked < 0)
            ),
            "selection_telescope_exact": bool(
                np.array_equal(
                    selection.coarse_to_blocked + selection.blocked_to_target,
                    selection.coarse_to_target,
                )
            ),
            "evaluation_minimum_D1": int(evaluation.coarse_to_blocked.min()),
            "evaluation_minimum_D2": int(evaluation.blocked_to_target.min()),
            "evaluation_uncorrected_minimum_D1": int(
                evaluation.uncorrected_coarse_to_blocked.min()
            ),
            "evaluation_uncorrected_negative_D1_paths": int(
                np.count_nonzero(evaluation.uncorrected_coarse_to_blocked < 0)
            ),
            "evaluation_telescope_exact": bool(
                np.array_equal(
                    evaluation.coarse_to_blocked + evaluation.blocked_to_target,
                    evaluation.coarse_to_target,
                )
            ),
        },
        "direct": {
            "selected_cap_numerator": direct_cap,
            "selection": _clipping_statistics(
                selection.coarse_to_target,
                cap=direct_cap,
                denominator=denominator,
                discount=discount,
            ),
            "evaluation": direct_evaluation,
            "resource": direct_resource,
            "leading_work_coefficient": direct_coefficient,
        },
        "ladder": {
            "chosen_allocation": chosen,
            "selection_first": _clipping_statistics(
                selection.coarse_to_blocked,
                cap=first_cap,
                denominator=denominator,
                discount=discount,
            ),
            "selection_second": _clipping_statistics(
                selection.blocked_to_target,
                cap=second_cap,
                denominator=denominator,
                discount=discount,
            ),
            "evaluation_first": first_evaluation,
            "evaluation_second": second_evaluation,
            "evaluation_combined": {
                "discounted_loss": float(combined_losses.mean()),
                "standard_error": combined_se,
                "ci95_low": float(combined_losses.mean()) - 1.96 * combined_se,
                "ci95_high": float(combined_losses.mean()) + 1.96 * combined_se,
            },
            "leading_work_coefficient": ladder_coefficient,
        },
        "decision": {
            "criterion_passes_on_selection": ladder_coefficient < direct_coefficient,
            "direct_over_ladder_leading_work_ratio": (
                direct_coefficient / ladder_coefficient
            ),
            "evaluation_ladder_loss_within_point_budget": bool(
                float(combined_losses.mean()) <= total_clipping_budget
            ),
            "evaluation_direct_loss_within_point_budget": bool(
                float(direct_evaluation["discounted_loss"])
                <= total_clipping_budget
            ),
        },
        "all_allocation_candidates": candidates,
        "provenance": {
            "git": _git_metadata(root),
            "sha256": {
                str(source.relative_to(root)): _sha256(source),
                str(ladder_source.relative_to(root)): _sha256(ladder_source),
            },
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=2_000_000)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--total-clipping-budget", type=float, default=TOTAL_CLIPPING_BUDGET)
    parser.add_argument("--cap-selection-z", type=float, default=CAP_SELECTION_Z)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_pilot(
        paths=args.paths,
        selection_seed=args.selection_seed,
        evaluation_seed=args.evaluation_seed,
        total_clipping_budget=args.total_clipping_budget,
        cap_selection_z=args.cap_selection_z,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
