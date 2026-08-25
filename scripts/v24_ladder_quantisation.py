#!/usr/bin/env python3
"""Re-select the whole blocked ladder against the dollar budget and quantise it.

The published ladder figure holds every arm to a matched exceedance fraction and
reports its normalisation ratios rounded to the nearest integer.  Two things are
wrong with reading those as reductions in oracle calls.  The arms are not
selected under the budget that produced the headline raw and one-block figures,
and a threshold register can only implement a power of two, so no arm can
realise a ratio such as 50 or 94.

This script re-selects every arm that divides 252 against the same $0.001
discounted truncation-bias budget, then applies the register rounding, so the
selected ladder and the implementable ladder can be plotted together.

Outputs results/ladder_quantisation.{png,csv} and results/ladder_quantisation.json.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

S0 = STRIKE = 100.0
RATE, VOL, MATURITY, N_DATES = 0.05, 0.20, 1.0, 252
PRICE_SCALE = 16_384
UNITS_PER_DOLLAR = N_DATES * PRICE_SCALE
BUDGET = 1.0e-3
DISCOUNT = math.exp(-RATE * MATURITY)
BLOCK_COUNTS = (1, 2, 3, 4, 6, 12)
SEGMENT = 21  # every block boundary of every arm is a multiple of this
DEFAULT_PATHS = 2_000_000
DEFAULT_CHUNK = 250_000
DEFAULT_SEED = 20_260_802

BLUE = "#2f6f9f"
ORANGE = "#c05a28"
GREY = "#555555"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _simulate(paths: int, chunk: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the arithmetic average and the twelve segment log-sums per path.

    Every arm's block boundaries are multiples of ``SEGMENT``, so the segment
    log-sums are sufficient to rebuild any arm's block geometric means.
    """

    n_segments = N_DATES // SEGMENT
    dt = MATURITY / N_DATES
    mu = (RATE - 0.5 * VOL**2) * dt
    vol = VOL * math.sqrt(dt)
    rng = np.random.default_rng(seed)
    arithmetic = np.empty(paths)
    segments = np.empty((paths, n_segments))
    done = 0
    while done < paths:
        size = min(chunk, paths - done)
        log_price = np.full(size, math.log(S0))
        price_sum = np.zeros(size)
        seg = np.zeros((size, n_segments))
        for date in range(N_DATES):
            log_price = log_price + mu + vol * rng.standard_normal(size)
            price_sum += np.exp(log_price)
            seg[:, date // SEGMENT] += log_price
        arithmetic[done : done + size] = price_sum / N_DATES
        segments[done : done + size] = seg
        done += size
    return arithmetic, segments


def _block_control(segments: np.ndarray, block_count: int) -> np.ndarray:
    """B_k, the mean of the geometric means of k equal contiguous blocks."""

    if N_DATES % block_count:
        raise ValueError(f"{block_count} does not divide {N_DATES}")
    dates_per_block = N_DATES // block_count
    segs_per_block = dates_per_block // SEGMENT
    total = np.zeros(segments.shape[0])
    for index in range(block_count):
        start = index * segs_per_block
        block_log_sum = segments[:, start : start + segs_per_block].sum(axis=1)
        total += np.exp(block_log_sum / dates_per_block)
    return total / block_count


def _budget_cutoff(residual: np.ndarray, budget: float) -> tuple[float, float]:
    """Smallest cutoff whose discounted expected clipping loss fits the budget."""

    ordered = np.sort(residual)[::-1]
    count = ordered.size
    prefix = np.cumsum(ordered)
    index = np.arange(1, count + 1)
    # With cutoff ordered[i] the loss is sum_{j<i}(ordered[j] - ordered[i]),
    # which grows with i because the cutoff falls.
    loss = DISCOUNT * (prefix - index * ordered) / count
    if loss[0] > budget:
        raise ValueError("budget is below the achievable floor")
    position = int(np.searchsorted(loss, budget, side="right")) - 1
    position = min(max(position, 0), count - 1)
    return float(ordered[position]), float(loss[position])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=_REPO / "results")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"simulating {args.paths:,} paths (seed {args.seed}) ...", flush=True)
    arithmetic, segments = _simulate(args.paths, args.chunk, args.seed)
    raw_payoff = np.maximum(arithmetic - STRIKE, 0.0)

    raw_cut, raw_loss = _budget_cutoff(raw_payoff, BUDGET)
    raw_units = int(round(raw_cut * UNITS_PER_DOLLAR))
    raw_bits = max(1, (raw_units - 1).bit_length())

    controls = {k: _block_control(segments, k) for k in BLOCK_COUNTS}
    # Equal blocks nest only when one count divides the other, so B_k rises with
    # k along a divisibility chain and incomparable counts carry no claim.  For
    # example the k=3 block spanning dates 85 to 168 straddles the k=2 boundary
    # at 126, so B_2 and B_3 are not ordered.
    refinement_pairs = [
        (coarse, fine)
        for coarse in BLOCK_COUNTS
        for fine in BLOCK_COUNTS
        if fine > coarse and fine % coarse == 0
    ]
    refinement_ordered = all(
        float(np.min(controls[fine] - controls[coarse])) >= 0.0
        for coarse, fine in refinement_pairs
    )
    incomparable_pairs = [
        (a, b)
        for a in BLOCK_COUNTS
        for b in BLOCK_COUNTS
        if b > a and b % a
    ]
    rows = []
    dominated = True
    for k in BLOCK_COUNTS:
        control = controls[k]
        dominated &= bool(np.min(arithmetic - control) >= 0.0)
        residual = raw_payoff - np.maximum(control - STRIKE, 0.0)
        cut, loss = _budget_cutoff(residual, BUDGET)
        units = int(round(cut * UNITS_PER_DOLLAR))
        bits = max(1, (units - 1).bit_length())
        rows.append(
            {
                "block_count": k,
                "dates_per_block": N_DATES // k,
                "cutoff_dollars": cut,
                "realised_discounted_loss": loss,
                "cutoff_units": units,
                "threshold_bits": bits,
                "amplitude_scale_dollars": (1 << bits) / UNITS_PER_DOLLAR,
                "register_fill_fraction": units / (1 << bits),
                "selected_ratio": raw_cut / cut,
                "implemented_ratio": 2 ** (raw_bits - bits),
                "rounding_factor": (raw_cut / cut) / 2 ** (raw_bits - bits),
            }
        )
        print(
            f"  k={k:2d} cutoff=${cut:8.4f} bits={bits} "
            f"selected={raw_cut / cut:7.2f}x implemented={2 ** (raw_bits - bits):4d}x",
            flush=True,
        )

    gates = [
        {
            "name": "every block count divides the date count",
            "passed": all(N_DATES % k == 0 for k in BLOCK_COUNTS),
        },
        {
            "name": "the arithmetic average dominates every block control pathwise",
            "passed": dominated,
        },
        {
            "name": "block controls rise along every divisibility refinement",
            "passed": refinement_ordered,
            "refinement_pairs": refinement_pairs,
            "note": (
                "incomparable counts carry no ordering claim: "
                + ", ".join(f"{a} vs {b}" for a, b in incomparable_pairs)
            ),
        },
        {
            "name": "every realised loss sits inside the budget",
            "passed": all(row["realised_discounted_loss"] <= BUDGET * 1.02 for row in rows),
            "observed_max": max(row["realised_discounted_loss"] for row in rows),
        },
        {
            "name": "one-block arm reproduces the built 24-bit register",
            "passed": rows[0]["threshold_bits"] == 24,
            "observed": rows[0]["threshold_bits"],
        },
        {
            "name": "two-block arm reproduces the built 22-bit register",
            "passed": rows[1]["threshold_bits"] == 22,
            "observed": rows[1]["threshold_bits"],
        },
        {
            "name": "raw arm reproduces the built 28-bit register",
            "passed": raw_bits == 28,
            "observed": raw_bits,
        },
        {
            "name": "implemented ratios are powers of two",
            "passed": all(
                (row["implemented_ratio"] & (row["implemented_ratio"] - 1)) == 0
                for row in rows
            ),
        },
    ]

    # ------------------------------------------------------------------ figure
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ks = [row["block_count"] for row in rows]
    selected = [row["selected_ratio"] for row in rows]
    implemented = [row["implemented_ratio"] for row in rows]

    levels = sorted({row["implemented_ratio"] for row in rows})
    for level in levels:
        ax.axhline(level, color="0.88", linewidth=0.6, zorder=0)

    # Only k=1 and k=2 have built and transpiled oracles.  Draw those points
    # with solid lines and filled markers.  Continue from k=2 with dashed,
    # open-marker projections so the plot does not imply resource estimates
    # for k>=3.
    built = 2
    ax.plot(
        ks[:built],
        selected[:built],
        color=BLUE,
        linewidth=1.8,
        marker="o",
        markersize=6,
        label="selected scale (built)",
        zorder=4,
    )
    ax.step(
        ks[:built],
        implemented[:built],
        where="mid",
        color=ORANGE,
        linewidth=2.0,
        label="register scale (built)",
        zorder=4,
    )
    ax.plot(
        ks[:built],
        implemented[:built],
        color=ORANGE,
        linestyle="none",
        marker="s",
        markersize=6,
        zorder=5,
    )
    ax.plot(
        ks[built - 1 :],
        selected[built - 1 :],
        color="0.55",
        linewidth=1.3,
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markersize=6,
        label=r"selected scale ($k\geq3$ unbuilt)",
        zorder=2,
    )
    ax.step(
        ks[built - 1 :],
        implemented[built - 1 :],
        where="mid",
        color="0.55",
        linewidth=1.5,
        linestyle="--",
        label=r"register scale ($k\geq3$ unbuilt)",
        zorder=2,
    )
    ax.plot(
        ks[built:],
        implemented[built:],
        color="0.55",
        linestyle="none",
        marker="s",
        markerfacecolor="white",
        markersize=6,
        zorder=3,
    )

    for row in rows:
        ax.annotate(
            f"$2^{{{row['threshold_bits']}}}$",
            (row["block_count"], row["implemented_ratio"]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=ORANGE if row["block_count"] <= 2 else "0.45",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(levels)
    ax.set_yticklabels([str(level) for level in levels])
    # Leave room under the lowest step for its register-width annotation.
    ax.set_ylim(bottom=levels[0] * 0.62, top=max(selected) * 1.35)
    ax.set_xlabel("block count $k$")
    ax.set_ylabel("reduction in the encoded amplitude scale")
    ax.set_title("Built controls and unbuilt amplitude-scale projections")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", ncol=2)
    fig.text(
        0.5,
        0.005,
        f"{args.paths:,} paths, seed {args.seed}; $N=252$, "
        "$2^{28}$ raw register. Amplitude scales only, not query counts.",
        ha="center",
        fontsize=8,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    png = args.out / "ladder_quantisation.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.out / "ladder_quantisation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "schema_version": "ladder-quantisation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "configuration": {
            "n_dates": N_DATES,
            "price_scale": PRICE_SCALE,
            "budget": BUDGET,
            "paths": args.paths,
            "seed": args.seed,
            "block_counts": list(BLOCK_COUNTS),
            "segment_dates": SEGMENT,
        },
        "raw_arm": {
            "cutoff_dollars": raw_cut,
            "realised_discounted_loss": raw_loss,
            "cutoff_units": raw_units,
            "threshold_bits": raw_bits,
            "amplitude_scale_dollars": (1 << raw_bits) / UNITS_PER_DOLLAR,
            "register_fill_fraction": raw_units / (1 << raw_bits),
        },
        "arms": rows,
        "limitations": [
            "Continuous Gaussian increments, not the oracle's binary-shock model.",
            "Amplitude scales only; no per-query cost and no executed query schedule.",
            "Only the one-block and two-block oracles have been built and counted.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "inputs": {
            "scripts/v24_ladder_quantisation.py": {
                "sha256": _sha256(Path(__file__).resolve())
            }
        },
        "outputs": {
            "results/ladder_quantisation.png": {"sha256": _sha256(png)},
            "results/ladder_quantisation.csv": {"sha256": _sha256(csv_path)},
        },
        "gates": gates,
    }
    (args.out / "ladder_quantisation.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )

    print()
    for gate in gates:
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] {gate['name']}")
    print(f"\nraw cutoff ${raw_cut:.4f} -> 2^{raw_bits} (${(1 << raw_bits) / UNITS_PER_DOLLAR:.4f})")
    print(f"wrote {png}")
    return 0 if all(gate["passed"] for gate in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
