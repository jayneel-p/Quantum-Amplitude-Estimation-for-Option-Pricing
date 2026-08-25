"""Movement-3 figure: where the 252-date arithmetic-oracle price error lives.

The continuous-model price error splits into a fixed-point ENCODING error
(rounding + clipping, encoded oracle vs the exact binary model) and a
binary-shock MODEL error (exact binary model vs continuous Black-Scholes, fixed
by the weak-Euler discretisation).  Refining the fraction bits from 18 to 30
drives the encoding error below the fixed model floor and takes the total under
the predeclared $0.01 target, at a larger T-count.

Data: results/v20/collapsed_resource_evidence.json, the COLLAPSED geometric leg
at 4,000,000 paired binary paths and 8,000,000 continuous reference paths.  The
cap arm read here is the manuscript's B_R = $2.864, so the three totals are the
rows of the paper's precision-decision table.  This supersedes
results/v8/arithmetic_asian_oracle_validation.json, which measured the retired
per-date leg at 200,000 paths.

Resource annotations.  The artifact's resource ledger only covers the selected
30-bit configuration.  The 18- and 24-bit collapsed counts are therefore
recomputed here with ``estimate_arithmetic_asian_resources`` at the same cap;
the 30-bit recomputation is asserted against the artifact before any of them is
plotted, so the two paths agree at the one precision they share.

Every bar is a Monte Carlo estimate, so the total carries a 95% interval.

Output: results/oracle_accuracy.png, results/oracle_accuracy_figure_data.json
"""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    build_arithmetic_asian_model,
    estimate_arithmetic_asian_resources,
)
from qc_option_pricing.quantum.asian_oracle import AsianGridSpec

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/v20/collapsed_resource_evidence.json"
FIGURE = ROOT / "results/oracle_accuracy.png"
SIDECAR = ROOT / "results/oracle_accuracy_figure_data.json"

# The arm plotted: collapsed geometric leg, manuscript cap B_R = $2.864.
LEG = "collapsed"
CAP_ARM = "manuscript_B_R_2.864"
CAP_DOLLARS = 2.864

# Instrument, identical to scripts/v21_collapsed_resource_evidence.py.
S0, STRIKE, RATE, SIGMA, MATURITY, N_DATES = 100.0, 100.0, 0.05, 0.20, 1.0, 252

evidence = json.load(open(EVIDENCE))
block1 = evidence["block_1_precision_sweep"]
sweep = block1["precision_sweep"]
target = block1["predeclared_absolute_target"]
shock = block1["binary_shock_model_error"]
ledgers = {row["cap_rule"]: row for row in evidence["block_3_resource_ledger"]["ledgers"]}

bits = [row["multiplier_fraction_bits"] for row in sweep]
scales = [row["price_scale"] for row in sweep]
arms = [row["caps"][LEG][CAP_ARM] for row in sweep]

encoding = [arm["fixed_point_encoding_error_discounted"] for arm in arms]
encoding_se = [arm["fixed_point_encoding_error_standard_error"] for arm in arms]
total = [arm["continuous_price_error"]["point_estimate"] for arm in arms]
total_se = [arm["continuous_price_error"]["standard_error"] for arm in arms]
meets = [arm["continuous_price_error"]["meets_target_at_95_percent"] for arm in arms]

# The model error is leg-independent and common to every row; assert that the
# artifact really carries the same value in each, then use the block-level one.
shock_value = shock["point_estimate"]
shock_se = shock["standard_error"]
for arm in arms:
    if arm["binary_shock_model_error_discounted"] != shock_value:
        raise AssertionError("the binary-shock model error is not common to the sweep")
for enc, tot in zip(encoding, total):
    if abs(enc + shock_value - tot) > 1e-15:
        raise AssertionError("encoding + model error does not reconstruct the total")


def collapsed_resources(fraction_bits: int, price_scale: int) -> dict:
    """A- and Q-operator counts for the collapsed leg at one precision."""

    spec = AsianGridSpec(
        n_dates=N_DATES,
        shock_points=(-1.0, 1.0),
        shock_probabilities=(0.5, 0.5),
        s0=S0,
        strike=STRIKE,
        rate=RATE,
        volatility=SIGMA,
        maturity=MATURITY,
        shock_scale=1,
        price_scale=price_scale,
        residual_payoff_cap=CAP_DOLLARS,
        geometric_leg=LEG,
    )
    model = build_arithmetic_asian_model(spec, multiplier_fraction_bits=fraction_bits)
    estimate = estimate_arithmetic_asian_resources(model)
    return {
        "multiplier_fraction_bits": fraction_bits,
        "price_scale": price_scale,
        "cap_dollars": CAP_DOLLARS,
        "cap_numerator": model.requested_residual_cap_numerator,
        "value_bits": model.value_bits,
        "threshold_bits": model.threshold_bits,
        "geometric_dp_peak_states": model.geometric_dp_peak_states,
        "a_toffoli": estimate.a_counts.ccx,
        "a_t": estimate.a_counts.t,
        "a_logical_qubits": estimate.a_qubits,
        "q_t": estimate.q_counts.t,
        "q_logical_qubits": estimate.q_qubits_with_clean_reflection_ladder,
    }


resources = [collapsed_resources(b, s) for b, s in zip(bits, scales)]

# Cross-check against the one precision the artifact's ledger covers.
published = ledgers["manuscript"][LEG]
recomputed = next(r for r in resources if r["multiplier_fraction_bits"] == 30)
for key in ("a_t", "a_logical_qubits", "q_t", "q_logical_qubits", "threshold_bits",
            "cap_numerator"):
    if recomputed[key] != published[key]:
        raise AssertionError(
            f"30-bit {key}: recomputed {recomputed[key]} != artifact {published[key]}"
        )
for row, arm in zip(resources, arms):
    if row["cap_numerator"] != arm["cap_numerator"]:
        raise AssertionError(
            f"{row['multiplier_fraction_bits']}-bit cap numerator "
            f"{row['cap_numerator']} != sweep {arm['cap_numerator']}"
        )

tlabel = {
    row["multiplier_fraction_bits"]:
        f"{row['a_t'] / 1e6:.0f}M $T$\n{row['a_logical_qubits']:,} qubits"
    for row in resources
}

fig, ax = plt.subplots(figsize=(7.8, 5.0))
x = np.arange(len(bits))
enc = np.array(encoding)
shk = np.full(len(bits), shock_value)
tot = np.array(total)
ax.bar(x, enc, width=0.6, color="#c0392b", zorder=3,
       label="fixed-point encoding error (rounding + clipping)")
ax.bar(x, shk, width=0.6, bottom=enc, color="#3b6ea5", zorder=3,
       label="binary-shock model error (vs continuous)")
ax.errorbar(x, tot, yerr=1.96 * np.array(total_se), fmt="none", ecolor="0.12",
            elinewidth=1.5, capsize=5, capthick=1.5, zorder=6,
            label="95% interval on the total ($1.96\\,$SE)")
ax.axhline(target, ls="--", color="0.35", lw=1.6, zorder=4,
           label=f"predeclared \\${target:g} accuracy target")
xt = [f"{b} fraction bits" + (f"\n{tlabel[b]}" if b in tlabel else "") for b in bits]
ax.set_xticks(x)
ax.set_xticklabels(xt, fontsize=8.5)
ax.set_ylabel("discounted continuous-model price error (\\$)")
ax.set_title("252-date arithmetic oracle: price error vs encoding precision")
ax.set_ylim(0, 0.056)
ax.margins(x=0.08)
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
ax.grid(True, axis="y", alpha=0.25, zorder=0)
fig.tight_layout()
fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
plt.close(fig)


def git_info() -> dict[str, object]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    rev = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "rev": rev.stdout.strip() if rev.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


script = Path(__file__).resolve()
sidecar = {
    "schema_version": "oracle-accuracy-figure-v1",
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "command": "scripts/figure_oracle_accuracy.py",
    "argv": sys.argv[1:],
    "figure": "results/oracle_accuracy.png",
    "purpose": "every value drawn in results/oracle_accuracy.png, so the axis "
               "annotations and bar heights are traceable",
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    },
    "git": git_info(),
    "source": {
        "evidence": "results/v20/collapsed_resource_evidence.json",
        "geometric_leg": LEG,
        "cap_arm": CAP_ARM,
        "cap_dollars": CAP_DOLLARS,
        "binary_paths": block1["binary_paths"],
        "continuous_paths": block1["continuous_reference"]["paths"],
        "predeclared_absolute_target": target,
        "supersedes": "results/v8/arithmetic_asian_oracle_validation.json "
                      "(retired per-date leg, 200,000 paths)",
    },
    "binary_shock_model_error": shock,
    "bars": [
        {
            "multiplier_fraction_bits": b,
            "price_scale": s,
            "fixed_point_encoding_error_discounted": e,
            "fixed_point_encoding_error_standard_error": ese,
            "binary_shock_model_error_discounted": shock_value,
            "binary_shock_model_error_standard_error": shock_se,
            "continuous_price_error": t,
            "continuous_price_error_standard_error": tse,
            "continuous_price_error_ci95": [t - 1.96 * tse, t + 1.96 * tse],
            "meets_target_at_95_percent": m,
        }
        for b, s, e, ese, t, tse, m in zip(
            bits, scales, encoding, encoding_se, total, total_se, meets
        )
    ],
    "axis_resource_annotations": {
        "note": "recomputed here with estimate_arithmetic_asian_resources; the "
                "evidence artifact's block_3 ledger only covers 30 fraction bits, "
                "and the 30-bit row below is asserted equal to it before plotting",
        "checked_against": "block_3_resource_ledger, cap_rule='manuscript', collapsed",
        "rows": resources,
    },
    "source_hashes": {
        "scripts/figure_oracle_accuracy.py": sha256(script),
        "results/v20/collapsed_resource_evidence.json": sha256(EVIDENCE),
    },
}
SIDECAR.write_text(json.dumps(sidecar, indent=1) + "\n")

for b, e, ese, t, tse, m in zip(bits, encoding, encoding_se, total, total_se, meets):
    print(f"  {b} bits: encoding={e:.6f}+-{ese:.6f} + shock={shock_value:.6f}"
          f"+-{shock_se:.6f} = total={t:.6f}+-{tse:.6f}  meets_$0.01={m}")
for row in resources:
    print(f"  {row['multiplier_fraction_bits']} bits: A = {row['a_t']:,} T / "
          f"{row['a_logical_qubits']:,} qubits ; Q = {row['q_t']:,} T / "
          f"{row['q_logical_qubits']:,} qubits")
print(f"wrote {FIGURE.relative_to(ROOT)}")
print(f"wrote {SIDECAR.relative_to(ROOT)}")
