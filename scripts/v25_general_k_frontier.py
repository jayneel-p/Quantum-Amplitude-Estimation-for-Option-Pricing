#!/usr/bin/env python3
"""Generate the resource-accounted equal-block control frontier.

For each requested block count ``k``, this script:

* selects a residual cap on finite binary-shock paths using a one-sided
  clipping-loss criterion and checks it on an independent seed;
* builds the parameterized ``H - C_k`` model and its compositional logical
  resource ledger;
* optionally materializes and transpiles the full 252-date circuit to verify
  the ledger gate by gate;
* prices the finite-grid control and its continuous Black--Scholes counterpart
  with independently scrambled Sobol' replicates; and
* reports selected, register-implemented, and net leading-order T reductions
  against a separately built raw arithmetic oracle.

The net reduction assumes QAE query count is proportional to the implemented
amplitude scale.  It is not an executed amplitude-estimation schedule, a
physical-resource estimate, or a quantum-advantage claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from qc_option_pricing.quantum.arithmetic_asian_oracle import (  # noqa: E402
    build_arithmetic_asian_model,
    build_arithmetic_asian_oracle,
    estimate_arithmetic_asian_resources,
    primitive_counts_from_circuit,
)
from qc_option_pricing.quantum.blocked_asian_oracle import (  # noqa: E402
    black_scholes_binary_spec,
    build_blocked_asian_model,
    build_blocked_asian_oracle,
    estimate_block_control_price_rqmc,
    estimate_blocked_asian_resources,
    estimate_encoded_block_control_price_rqmc,
    primitive_counts_from_blocked_asian_circuit,
    sample_blocked_payoffs,
)
from qc_option_pricing.quantum.telescoping_asian_ladder import (  # noqa: E402
    build_k2_ladder_model,
    estimate_k2_ladder_resources,
)


DEFAULT_BLOCK_COUNTS = (1, 2, 3, 4, 6, 12)
DEFAULT_TRANSPILE_COUNTS = (1, 2, 3, 4, 6)
DEFAULT_SELECTION_SEED = 20_260_730
DEFAULT_EVALUATION_SEED = 20_260_731
DEFAULT_CONTROL_SEED = 20_260_902


class _SortedCapSample:
    """Sorted integer payoffs with prefix moments for cap selection."""

    def __init__(self, values: np.ndarray):
        self.ordered = np.sort(np.asarray(values, dtype=np.int64))
        values_float = self.ordered.astype(np.longdouble)
        self.prefix_sum = np.concatenate(
            (np.asarray([0.0], dtype=np.longdouble), np.cumsum(values_float))
        )
        self.prefix_square_sum = np.concatenate(
            (
                np.asarray([0.0], dtype=np.longdouble),
                np.cumsum(values_float * values_float),
            )
        )

    def loss_mean_and_standard_error(self, cap: int) -> tuple[float, float]:
        count = self.ordered.size
        start = int(np.searchsorted(self.ordered, cap, side="right"))
        tail_count = count - start
        if tail_count == 0:
            return 0.0, 0.0
        tail_sum = self.prefix_sum[-1] - self.prefix_sum[start]
        tail_square_sum = self.prefix_square_sum[-1] - self.prefix_square_sum[start]
        loss_sum = tail_sum - cap * tail_count
        loss_square_sum = (
            tail_square_sum - 2.0 * cap * tail_sum + cap * cap * tail_count
        )
        mean = loss_sum / count
        variance = max(
            0.0,
            (loss_square_sum - loss_sum * loss_sum / count) / (count - 1),
        )
        return mean, math.sqrt(variance / count)


def _smallest_cap_for_budget(
    values: np.ndarray,
    *,
    denominator: int,
    discount: float,
    discounted_budget: float,
    selection_z: float,
) -> int:
    sample = _SortedCapSample(values)

    def meets(cap: int) -> bool:
        mean, standard_error = sample.loss_mean_and_standard_error(cap)
        upper = discount * (mean + selection_z * standard_error) / denominator
        return upper <= discounted_budget

    low, high = 1, max(1, int(sample.ordered[-1]))
    while low < high:
        middle = (low + high) // 2
        if meets(middle):
            high = middle
        else:
            low = middle + 1
    if not meets(low) or (low > 1 and meets(low - 1)):
        raise AssertionError("cap selector failed to return the smallest passing integer")
    return low


def _clipping_statistics(
    values: np.ndarray,
    *,
    cap: int,
    denominator: int,
    discount: float,
) -> dict[str, object]:
    losses = discount * np.maximum(values - cap, 0).astype(float) / denominator
    mean = float(losses.mean())
    standard_error = float(losses.std(ddof=1) / math.sqrt(losses.size))
    clipped = int(np.count_nonzero(losses))
    return {
        "cap_numerator": cap,
        "cap_dollars": cap / denominator,
        "discounted_loss": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "clipped_paths": clipped,
        "clipped_fraction": clipped / losses.size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance_label(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)


def _git_metadata() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    return {
        "head": revision.stdout.strip() if revision.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _resource_record(model) -> dict[str, object]:
    estimate = estimate_blocked_asian_resources(model)
    return {
        "a_qubits": estimate.a_qubits,
        "q_qubits": estimate.q_qubits_with_clean_reflection_ladder,
        "a_counts": estimate.a_counts.as_dict(),
        "q_counts": estimate.q_counts.as_dict(),
        "qrom_rows": estimate.qrom_rows,
        "arbitrary_rotations": estimate.arbitrary_rotations,
    }


def _raw_resource_record(spec, cap_dollars: float, fraction_bits: int):
    raw_spec = replace(
        spec,
        geometric_leg="none",
        payoff_cap=cap_dollars,
        residual_payoff_cap=None,
    )
    model = build_arithmetic_asian_model(
        raw_spec,
        multiplier_fraction_bits=fraction_bits,
    )
    estimate = estimate_arithmetic_asian_resources(model)
    return model, {
        "requested_cap_numerator": model.requested_residual_cap_numerator,
        "requested_cap_dollars": (
            model.requested_residual_cap_numerator
            / (spec.n_dates * spec.price_scale)
        ),
        "threshold_bits": model.threshold_bits,
        "normalization_numerator": model.normalization_numerator,
        "amplitude_scale_dollars": model.normalization_dollars,
        "a_qubits": estimate.a_qubits,
        "q_qubits": estimate.q_qubits_with_clean_reflection_ladder,
        "a_counts": estimate.a_counts.as_dict(),
        "q_counts": estimate.q_counts.as_dict(),
    }


def _production_k1_resource_record(spec, cap_dollars: float, fraction_bits: int):
    controlled_spec = replace(
        spec,
        geometric_leg="collapsed",
        payoff_cap=None,
        residual_payoff_cap=cap_dollars,
    )
    model = build_arithmetic_asian_model(
        controlled_spec,
        multiplier_fraction_bits=fraction_bits,
    )
    estimate = estimate_arithmetic_asian_resources(model)
    return model, {
        "a_qubits": estimate.a_qubits,
        "q_qubits": estimate.q_qubits_with_clean_reflection_ladder,
        "a_counts": estimate.a_counts.as_dict(),
        "q_counts": estimate.q_counts.as_dict(),
        "qrom_rows": estimate.qrom_rows,
        "arbitrary_rotations": estimate.arbitrary_rotations,
    }


def _transpile_check(model) -> dict[str, object]:
    started = time.perf_counter()
    oracle = build_blocked_asian_oracle(model)
    built = time.perf_counter()
    counted = primitive_counts_from_blocked_asian_circuit(oracle)
    finished = time.perf_counter()
    estimated = estimate_blocked_asian_resources(model)
    matches = {
        "h": counted.h == estimated.a_counts.h,
        "x": counted.x == estimated.a_counts.x,
        "cx": counted.cx == estimated.a_counts.cx,
        "ccx": counted.ccx == estimated.a_counts.ccx,
        "qubits": oracle.circuit.num_qubits == estimated.a_qubits,
    }
    return {
        "materialized": True,
        "basis_gates": ["h", "x", "cx", "ccx"],
        "optimization_level": 0,
        "build_seconds": built - started,
        "transpile_and_count_seconds": finished - built,
        "counted_a_qubits": oracle.circuit.num_qubits,
        "counted_a_counts": counted.as_dict(),
        "matches": matches,
        "all_match": all(matches.values()),
    }


def _production_k1_transpile_check(model) -> dict[str, object]:
    started = time.perf_counter()
    oracle = build_arithmetic_asian_oracle(model)
    built = time.perf_counter()
    counted = primitive_counts_from_circuit(oracle)
    finished = time.perf_counter()
    estimated = estimate_arithmetic_asian_resources(model)
    matches = {
        "h": counted.h == estimated.a_counts.h,
        "x": counted.x == estimated.a_counts.x,
        "cx": counted.cx == estimated.a_counts.cx,
        "ccx": counted.ccx == estimated.a_counts.ccx,
        "qubits": oracle.circuit.num_qubits == estimated.a_qubits,
    }
    return {
        "materialized": True,
        "implementation": "optimized production k=1 oracle",
        "basis_gates": ["h", "x", "cx", "ccx"],
        "optimization_level": 0,
        "build_seconds": built - started,
        "transpile_and_count_seconds": finished - built,
        "counted_a_qubits": oracle.circuit.num_qubits,
        "counted_a_counts": counted.as_dict(),
        "matches": matches,
        "all_match": all(matches.values()),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = [
        "block_count",
        "cap_dollars",
        "evaluation_clipping_loss",
        "threshold_bits",
        "selected_ratio",
        "implemented_ratio",
        "q_t",
        "per_query_t_overhead",
        "net_t_reduction",
        "q_qubits",
        "qubit_overhead",
        "encoded_control_discounted",
        "encoded_control_se",
        "continuous_control_discounted",
        "continuous_control_se",
        "transpiled",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_figure(path: Path, rows: Sequence[dict[str, object]]) -> None:
    blocks = np.asarray([row["block_count"] for row in rows], dtype=float)
    selected = np.asarray([row["selected_ratio"] for row in rows], dtype=float)
    implemented = np.asarray([row["implemented_ratio"] for row in rows], dtype=float)
    net = np.asarray([row["net_t_reduction"] for row in rows], dtype=float)
    t_overhead = np.asarray(
        [100.0 * (row["per_query_t_overhead"] - 1.0) for row in rows],
        dtype=float,
    )
    qubit_overhead = np.asarray(
        [100.0 * (row["qubit_overhead"] - 1.0) for row in rows],
        dtype=float,
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    left, right = axes
    left.plot(
        blocks,
        selected,
        linestyle="--",
        marker="o",
        color="#777777",
        label="selected scale",
    )
    left.plot(
        blocks,
        implemented,
        linestyle="-",
        marker="s",
        color="#2f6f9f",
        label="register scale",
    )
    left.plot(
        blocks,
        net,
        linestyle="-",
        marker="D",
        linewidth=2.0,
        color="#b3492d",
        label="net $T$ reduction",
    )
    left.set_yscale("log", base=2)
    left.set_xticks(blocks)
    left.set_xlabel("number of blocks $k$")
    left.set_ylabel("reduction relative to raw oracle")
    left.grid(axis="y", alpha=0.25)
    left.legend(frameon=False, loc="upper left")
    for index, (x_value, y_value) in enumerate(zip(blocks, net)):
        left.annotate(
            f"{y_value:.1f}",
            (x_value, y_value),
            xytext=(0, 7 if index == 0 else -13),
            textcoords="offset points",
            ha="center",
            va="bottom" if index == 0 else "baseline",
            fontsize=8.5,
            color="#8f3522",
        )
    for row in rows:
        if not row["transpiled"]:
            left.scatter(
                row["block_count"],
                row["net_t_reduction"],
                marker="D",
                s=42,
                facecolor="white",
                edgecolor="#b3492d",
                linewidth=1.4,
                zorder=4,
            )

    width = 0.34
    t_bars = right.bar(
        blocks - width / 2,
        t_overhead,
        width=width,
        color="#2f6f9f",
        label="$T$ count per query",
    )
    qubit_bars = right.bar(
        blocks + width / 2,
        qubit_overhead,
        width=width,
        color="#8c6d9e",
        label="logical qubits",
    )
    for index, row in enumerate(rows):
        if not row["transpiled"]:
            for bar, color in (
                (t_bars[index], "#2f6f9f"),
                (qubit_bars[index], "#8c6d9e"),
            ):
                bar.set_facecolor("white")
                bar.set_edgecolor(color)
                bar.set_linewidth(1.4)
                bar.set_hatch("//")
    right.set_xticks(blocks)
    right.set_xlabel("number of blocks $k$")
    right.set_ylabel("overhead relative to raw oracle (%)")
    right.grid(axis="y", alpha=0.25)
    handles, labels = right.get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch="//",
            label="compositional estimate",
        )
    )
    labels.append("compositional estimate")
    right.legend(handles, labels, frameon=False, loc="upper left", fontsize=8.5)
    figure.tight_layout(w_pad=2.2)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_frontier(args: argparse.Namespace) -> dict[str, object]:
    block_counts = tuple(args.block_counts)
    if len(set(block_counts)) != len(block_counts):
        raise ValueError("block counts must be unique")
    if any(current <= previous for previous, current in zip(block_counts, block_counts[1:])):
        raise ValueError("block counts must be strictly increasing")
    if any(args.n_dates % block_count for block_count in block_counts):
        raise ValueError("every block count must divide n_dates")
    if not set(args.transpile_blocks).issubset(block_counts):
        raise ValueError("transpile blocks must be included in block counts")
    if args.selection_paths < 2 or args.evaluation_paths < 2:
        raise ValueError("selection and evaluation paths must be at least two")
    if args.clipping_budget <= 0.0 or args.selection_z < 0.0:
        raise ValueError("clipping budget must be positive and selection_z nonnegative")
    if args.control_price_halfwidth_budget <= 0.0:
        raise ValueError("control price half-width budget must be positive")

    spec = black_scholes_binary_spec(
        n_dates=args.n_dates,
        s0=args.s0,
        strike=args.strike,
        rate=args.rate,
        volatility=args.volatility,
        maturity=args.maturity,
        price_scale=args.price_scale,
    )
    denominator = spec.n_dates * spec.price_scale
    discount = math.exp(-spec.rate * spec.maturity)
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    detailed_rows = []
    raw_selection = None
    raw_evaluation = None
    raw_cap = None
    raw_model = None
    raw_resource = None
    production_k1_model = None
    generic_k1_comparison = None

    for index, block_count in enumerate(block_counts):
        print(f"k={block_count}: compiling uncapped model", flush=True)
        uncapped = build_blocked_asian_model(
            spec,
            block_count,
            multiplier_fraction_bits=args.multiplier_fraction_bits,
        )
        sample_started = time.perf_counter()
        selection = sample_blocked_payoffs(
            uncapped.shared,
            paths=args.selection_paths,
            seed=args.selection_seed,
            chunk=args.chunk,
        )
        evaluation = sample_blocked_payoffs(
            uncapped.shared,
            paths=args.evaluation_paths,
            seed=args.evaluation_seed,
            chunk=args.chunk,
        )
        sample_seconds = time.perf_counter() - sample_started
        if index == 0:
            raw_selection = selection.target
            raw_evaluation = evaluation.target
            raw_cap = _smallest_cap_for_budget(
                raw_selection,
                denominator=denominator,
                discount=discount,
                discounted_budget=args.clipping_budget,
                selection_z=args.selection_z,
            )
            raw_model, raw_resource = _raw_resource_record(
                spec,
                raw_cap / denominator,
                args.multiplier_fraction_bits,
            )
        elif not np.array_equal(selection.target, raw_selection):
            raise AssertionError("raw selection paths changed across block counts")
        elif not np.array_equal(evaluation.target, raw_evaluation):
            raise AssertionError("raw evaluation paths changed across block counts")

        cap = _smallest_cap_for_budget(
            selection.residual,
            denominator=denominator,
            discount=discount,
            discounted_budget=args.clipping_budget,
            selection_z=args.selection_z,
        )
        model = build_blocked_asian_model(
            spec,
            block_count,
            multiplier_fraction_bits=args.multiplier_fraction_bits,
            residual_cap_dollars=cap / denominator,
        )
        generic_resource = _resource_record(model)
        resource = generic_resource
        if block_count == 1:
            production_k1_model, resource = _production_k1_resource_record(
                spec,
                model.requested_cap_dollars,
                args.multiplier_fraction_bits,
            )
            generic_k1_comparison = {
                "generic_a_t": generic_resource["a_counts"]["t"],
                "production_a_t": resource["a_counts"]["t"],
                "generic_q_t": generic_resource["q_counts"]["t"],
                "production_q_t": resource["q_counts"]["t"],
                "generic_a_qubits": generic_resource["a_qubits"],
                "production_a_qubits": resource["a_qubits"],
                "generic_q_qubits": generic_resource["q_qubits"],
                "production_q_qubits": resource["q_qubits"],
            }
        continuous_started = time.perf_counter()
        continuous_control = estimate_block_control_price_rqmc(
            spec,
            block_count,
            log2_points=args.continuous_control_log2_points,
            replicates=args.control_replicates,
            seed=args.control_seed + 1000 + 100 * block_count,
        )
        continuous_seconds = time.perf_counter() - continuous_started
        encoded_started = time.perf_counter()
        encoded_control = estimate_encoded_block_control_price_rqmc(
            model.shared,
            log2_paths=args.encoded_control_log2_paths,
            replicates=args.control_replicates,
            seed=args.control_seed + 100 * block_count,
            continuous_reference=continuous_control,
        )
        encoded_seconds = time.perf_counter() - encoded_started
        transpile_record = {"materialized": False, "all_match": None}
        if block_count in args.transpile_blocks:
            print(f"k={block_count}: building and transpiling production circuit", flush=True)
            transpile_record = (
                _production_k1_transpile_check(production_k1_model)
                if block_count == 1
                else _transpile_check(model)
            )
            if not transpile_record["all_match"]:
                raise AssertionError(f"k={block_count} circuit disagrees with its ledger")

        selected_ratio = raw_resource["requested_cap_dollars"] / model.requested_cap_dollars
        implemented_ratio = (
            raw_resource["normalization_numerator"] / model.normalization_numerator
        )
        per_query_overhead = (
            resource["q_counts"]["t"] / raw_resource["q_counts"]["t"]
        )
        net_reduction = implemented_ratio / per_query_overhead
        qubit_overhead = resource["q_qubits"] / raw_resource["q_qubits"]
        selection_stats = _clipping_statistics(
            selection.residual,
            cap=cap,
            denominator=denominator,
            discount=discount,
        )
        evaluation_stats = _clipping_statistics(
            evaluation.residual,
            cap=cap,
            denominator=denominator,
            discount=discount,
        )
        selection_stats["one_sided_selection_upper"] = (
            selection_stats["discounted_loss"]
            + args.selection_z * selection_stats["standard_error"]
        )
        evaluation_stats["ci95_upper"] = (
            evaluation_stats["discounted_loss"]
            + 1.96 * evaluation_stats["standard_error"]
        )
        row = {
            "block_count": block_count,
            "cap_dollars": model.requested_cap_dollars,
            "evaluation_clipping_loss": evaluation_stats["discounted_loss"],
            "threshold_bits": model.threshold_bits,
            "selected_ratio": selected_ratio,
            "implemented_ratio": implemented_ratio,
            "q_t": resource["q_counts"]["t"],
            "per_query_t_overhead": per_query_overhead,
            "net_t_reduction": net_reduction,
            "q_qubits": resource["q_qubits"],
            "qubit_overhead": qubit_overhead,
            "encoded_control_discounted": encoded_control.discounted_mean,
            "encoded_control_se": encoded_control.discounted_standard_error,
            "continuous_control_discounted": continuous_control.discounted_mean,
            "continuous_control_se": continuous_control.discounted_standard_error,
            "transpiled": bool(transpile_record["materialized"]),
        }
        rows.append(row)
        detailed_rows.append(
            {
                **row,
                "requested_cap_numerator": model.requested_cap_numerator,
                "normalization_numerator": model.normalization_numerator,
                "amplitude_scale_dollars": model.normalization_dollars,
                "selection_clipping": selection_stats,
                "evaluation_clipping": evaluation_stats,
                "sample_seconds": sample_seconds,
                "resource": resource,
                "rounding_certificates": [
                    asdict(certificate)
                    for certificate in model.shared.rounding_certificates
                ],
                "encoded_control": {
                    **asdict(encoded_control),
                    "runtime_seconds": encoded_seconds,
                },
                "continuous_control": {
                    **asdict(continuous_control),
                    "runtime_seconds": continuous_seconds,
                    "method": "scrambled Sobol' over the k-dimensional joint lognormal law",
                },
                "transpile_check": transpile_record,
            }
        )
        print(
            f"k={block_count}: {implemented_ratio:.0f}x scale / "
            f"{per_query_overhead:.4f} overhead = {net_reduction:.2f}x net T",
            flush=True,
        )

    raw_selection_stats = _clipping_statistics(
        raw_selection,
        cap=raw_cap,
        denominator=denominator,
        discount=discount,
    )
    raw_evaluation_stats = _clipping_statistics(
        raw_evaluation,
        cap=raw_cap,
        denominator=denominator,
        discount=discount,
    )
    raw_selection_stats["one_sided_selection_upper"] = (
        raw_selection_stats["discounted_loss"]
        + args.selection_z * raw_selection_stats["standard_error"]
    )
    raw_evaluation_stats["ci95_upper"] = (
        raw_evaluation_stats["discounted_loss"]
        + 1.96 * raw_evaluation_stats["standard_error"]
    )
    for row in detailed_rows:
        row["encoded_control_primary"] = {
            "method": "randomized QMC",
            "discounted_mean": row["encoded_control"]["discounted_mean"],
            "discounted_standard_error": row["encoded_control"][
                "discounted_standard_error"
            ],
            "ci95_low": row["encoded_control"]["discounted_ci95_low"],
            "ci95_high": row["encoded_control"]["discounted_ci95_high"],
        }
    if production_k1_model is not None:
        k1_row = detailed_rows[block_counts.index(1)]
        k1_exact_discounted = discount * (
            production_k1_model.geometric_control_undiscounted
        )
        k1_row["encoded_control_primary"] = {
            "method": "exact one-dimensional integer dynamic program",
            "discounted_mean": k1_exact_discounted,
            "discounted_standard_error": 0.0,
            "ci95_low": k1_exact_discounted,
            "ci95_high": k1_exact_discounted,
            "dynamic_program_peak_states": (
                production_k1_model.geometric_dp_peak_states
            ),
        }
    k2_exact_path = _REPO / "results/v24/k2_blocked_control_evidence.json"
    if (
        2 in block_counts
        and k2_exact_path.exists()
        and spec.n_dates == 252
        and spec.price_scale == 16_384
        and args.multiplier_fraction_bits == 30
        and math.isclose(spec.s0, 100.0)
        and math.isclose(spec.strike, 100.0)
        and math.isclose(spec.rate, 0.05)
        and math.isclose(spec.volatility, 0.20)
        and math.isclose(spec.maturity, 1.0)
    ):
        k2_exact_record = json.loads(k2_exact_path.read_text(encoding="utf-8"))
        k2_exact_discounted = k2_exact_record["exact_control_restoration"][
            "control_discounted"
        ]
        k2_row = detailed_rows[block_counts.index(2)]
        k2_row["encoded_control_primary"] = {
            "method": "exact two-dimensional integer dynamic program",
            "discounted_mean": k2_exact_discounted,
            "discounted_standard_error": 0.0,
            "ci95_low": k2_exact_discounted,
            "ci95_high": k2_exact_discounted,
            "reachable_states": k2_exact_record["exact_control_restoration"][
                "reachable_pairs"
            ],
            "source": str(k2_exact_path.relative_to(_REPO)),
            "source_sha256": _sha256(k2_exact_path),
        }
    k2_regression = None
    if 2 in block_counts:
        k2_row = detailed_rows[block_counts.index(2)]
        frozen = build_k2_ladder_model(
            spec,
            "blocked_to_target",
            multiplier_fraction_bits=args.multiplier_fraction_bits,
            increment_cap_dollars=k2_row["cap_dollars"],
        )
        frozen_resource = estimate_k2_ladder_resources(frozen)
        k2_regression = {
            "a_counts_match": frozen_resource.a_counts.as_dict()
            == k2_row["resource"]["a_counts"],
            "q_counts_match": frozen_resource.q_counts.as_dict()
            == k2_row["resource"]["q_counts"],
            "a_qubits_match": frozen_resource.a_qubits
            == k2_row["resource"]["a_qubits"],
            "q_qubits_match": frozen_resource.q_qubits_with_clean_reflection_ladder
            == k2_row["resource"]["q_qubits"],
        }
        k2_regression["all_match"] = all(k2_regression.values())
        if not k2_regression["all_match"]:
            raise AssertionError("general k=2 ledger disagrees with the frozen oracle")

    marginal = []
    for index, (previous, current) in enumerate(zip(rows, rows[1:])):
        marginal.append(
            {
                "from_k": previous["block_count"],
                "to_k": current["block_count"],
                "net_gain_multiplier": (
                    current["net_t_reduction"] / previous["net_t_reduction"]
                ),
                "additional_q_qubits": current["q_qubits"] - previous["q_qubits"],
                "encoded_control_runtime_ratio": (
                    detailed_rows[index + 1]["encoded_control"]["runtime_seconds"]
                    / detailed_rows[index]["encoded_control"]["runtime_seconds"]
                ),
            }
        )
    non_improving = [
        transition
        for transition in marginal
        if transition["net_gain_multiplier"] <= 1.0
    ]
    stopping = {
        "rule": (
            "stop at the first tested k whose net leading-order T reduction "
            "does not exceed the preceding tested rung"
        ),
        "first_non_improving_transition": non_improving[0] if non_improving else None,
        "conclusion": (
            "a stopping point was observed"
            if non_improving
            else "no T-count stopping point was observed within the tested block counts"
        ),
        "marginal_transitions": marginal,
    }

    csv_path = output_prefix.with_suffix(".csv")
    figure_path = output_prefix.with_suffix(".png")
    json_path = output_prefix.with_suffix(".json")
    _write_csv(csv_path, rows)
    _write_figure(figure_path, rows)
    input_paths = [
        Path(__file__).resolve(),
        _REPO / "src/qc_option_pricing/quantum/blocked_asian_oracle.py",
        _REPO / "src/qc_option_pricing/quantum/telescoping_asian_ladder.py",
        _REPO / "src/qc_option_pricing/quantum/arithmetic_asian_oracle.py",
    ]
    record = {
        "schema_version": "general-k-blocked-frontier-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_tested": (
            "For the stated finite binary-shock Asian model and clipping rule, "
            "the resource-accounted H-C_k oracle yields the reported leading-order "
            "T reductions when query count is proportional to amplitude scale."
        ),
        "configuration": {
            "n_dates": spec.n_dates,
            "block_counts": list(block_counts),
            "transpile_blocks": list(args.transpile_blocks),
            "s0": spec.s0,
            "strike": spec.strike,
            "rate": spec.rate,
            "volatility": spec.volatility,
            "maturity": spec.maturity,
            "price_scale": spec.price_scale,
            "multiplier_fraction_bits": args.multiplier_fraction_bits,
            "selection_paths": args.selection_paths,
            "evaluation_paths": args.evaluation_paths,
            "selection_seed": args.selection_seed,
            "evaluation_seed": args.evaluation_seed,
            "discounted_clipping_budget": args.clipping_budget,
            "cap_selection_one_sided_z": args.selection_z,
            "encoded_control_log2_paths": args.encoded_control_log2_paths,
            "continuous_control_log2_points": args.continuous_control_log2_points,
            "control_replicates": args.control_replicates,
            "control_seed": args.control_seed,
            "control_price_ci95_halfwidth_budget": (
                args.control_price_halfwidth_budget
            ),
        },
        "raw": {
            **raw_resource,
            "selection_clipping": raw_selection_stats,
            "evaluation_clipping": raw_evaluation_stats,
        },
        "arms": detailed_rows,
        "k1_implementation_comparison": generic_k1_comparison,
        "k2_frozen_regression": k2_regression,
        "stopping_analysis": stopping,
        "gates": {
            "all_rounding_certificates_pass": all(
                certificate["rounds_down_everywhere"]
                for row in detailed_rows
                for certificate in row["rounding_certificates"]
            ),
            "all_materialized_circuits_match_ledgers": all(
                row["transpile_check"]["all_match"]
                for row in detailed_rows
                if row["transpile_check"]["materialized"]
            ),
            "all_selection_bounds_within_budget": all(
                row["selection_clipping"]["one_sided_selection_upper"]
                <= args.clipping_budget
                for row in detailed_rows
            )
            and raw_selection_stats["one_sided_selection_upper"]
            <= args.clipping_budget,
            "all_evaluation_ci95_bounds_within_budget": all(
                row["evaluation_clipping"]["ci95_upper"]
                <= args.clipping_budget
                for row in detailed_rows
            )
            and raw_evaluation_stats["ci95_upper"] <= args.clipping_budget,
            "all_encoded_control_ci95_halfwidths_below_budget": all(
                1.96
                * row["encoded_control_primary"]["discounted_standard_error"]
                <= args.control_price_halfwidth_budget
                for row in detailed_rows
            ),
            "production_regressions_pass": (
                k2_regression is None or k2_regression["all_match"]
            ),
        },
        "limitations": [
            "Logical Clifford+T counts, not physical resources or runtime.",
            "Query count is modeled as proportional to implemented amplitude scale; no AE schedule is executed.",
            "Caps are selected on seeded binary-shock samples and checked on one independent seed.",
            "Control-price intervals are empirical randomized-QMC intervals, not deterministic certificates.",
            "The finite binary-shock model is not the continuous Black--Scholes model.",
            "The k=1 row uses the saved optimized production oracle; the uniform general-k layout has a small avoidable k=1 aggregation surcharge.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "provenance": {
            "git": _git_metadata(),
            "input_sha256": {
                _provenance_label(path): _sha256(path) for path in input_paths
            },
            "output_sha256": {
                _provenance_label(csv_path): _sha256(csv_path),
                _provenance_label(figure_path): _sha256(figure_path),
            },
        },
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {figure_path}", flush=True)
    return record


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dates", type=int, default=252)
    parser.add_argument("--block-counts", type=int, nargs="+", default=DEFAULT_BLOCK_COUNTS)
    parser.add_argument(
        "--transpile-blocks",
        type=int,
        nargs="*",
        default=DEFAULT_TRANSPILE_COUNTS,
    )
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--volatility", type=float, default=0.20)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--price-scale", type=int, default=16_384)
    parser.add_argument("--multiplier-fraction-bits", type=int, default=30)
    parser.add_argument("--selection-paths", type=int, default=2_000_000)
    parser.add_argument("--evaluation-paths", type=int, default=2_000_000)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--clipping-budget", type=float, default=1.0e-3)
    parser.add_argument("--selection-z", type=float, default=6.0)
    parser.add_argument("--chunk", type=int, default=250_000)
    parser.add_argument("--encoded-control-log2-paths", type=int, default=16)
    parser.add_argument("--continuous-control-log2-points", type=int, default=17)
    parser.add_argument("--control-replicates", type=int, default=16)
    parser.add_argument("--control-seed", type=int, default=DEFAULT_CONTROL_SEED)
    parser.add_argument("--control-price-halfwidth-budget", type=float, default=0.005)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=_REPO / "results" / "general_k_frontier",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    record = generate_frontier(args)
    return 0 if all(record["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
