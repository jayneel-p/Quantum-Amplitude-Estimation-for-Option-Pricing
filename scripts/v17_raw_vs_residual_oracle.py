"""Net resource comparison: raw arithmetic Asian oracle vs residual (QCV) oracle.

The manuscript reports a 16x reduction in QAE *queries* from the Kemna-Vorst
residual encoding, and separately reports the residual oracle's T count.  It does
not report the per-query cost of the control, so the net gate-level effect is
unstated.  This script builds the control-free counterpart of the accuracy-
qualified 252-date oracle from the same compositional primitives and reports both.

Method: re-compose the A operator exactly as estimate_arithmetic_asian_resources
does, omitting the four control-specific modules (geometric selected multipliers,
geometric positive part, geometric payoff scaling, residual subtraction) and the
geometric half of the initialization.  Every retained module is byte-identical to
the residual oracle's, so the difference is attributable to the control alone.

Caveat recorded in the output: the threshold encoder is retained unchanged.  A
production raw oracle would normalise against the larger raw cutoff B_A and so
would need a few more threshold bits; that term is ~10^2 Toffoli against ~10^7
and does not move the ratio.
"""
from __future__ import annotations
import json, subprocess, datetime, hashlib, sys

from qc_option_pricing.quantum.arithmetic_asian_oracle import (
    AsianGridSpec, build_arithmetic_asian_model,
    estimate_arithmetic_asian_resources,
    PrimitiveCounts, _selected_multiplier_counts, _base_adder_counts,
    _positive_part_counts, _constant_multiplier_counts, _a_qubit_counts,
)

# accuracy-qualified configuration from results/v8 (price_scale 16384, 30 fraction bits)
SPEC = AsianGridSpec(
    n_dates=252, shock_points=(-1.0, 1.0), shock_probabilities=(0.5, 0.5),
    s0=100.0, strike=100.0, rate=0.05, volatility=0.2, maturity=1.0,
    shock_scale=1, price_scale=16384, residual_payoff_cap=2.864,
)
FRACTION_BITS = 30

# override the exact-DP control price (value taken from results/v8 accuracy-qualified
# candidate); it does not affect any gate count, only the decode constant.
model = build_arithmetic_asian_model(
    SPEC, multiplier_fraction_bits=FRACTION_BITS,
    geometric_control_undiscounted_override=5.849185695523366,
    geometric_control_standard_error_undiscounted=8.796907814680672e-06,
)
resid = estimate_arithmetic_asian_resources(model)

n, q = SPEC.n_dates, SPEC.shock_qubits
v, t, m = model.value_bits, model.total_bits, model.threshold_bits

# --- rebuild the compute block WITHOUT the control ------------------------
price_multipliers = _selected_multiplier_counts(model, model.price_factors, "ceil").scaled(n)
price_sum = _base_adder_counts(t).scaled(n)
arithmetic_positive = _positive_part_counts(
    input_bits=t, output_bits=t, arithmetic_bits=t, subtract=n * SPEC.strike_integer)
initialization_raw = PrimitiveCounts(x=model.initial_price.bit_count())
compute_raw = initialization_raw + price_multipliers + price_sum + arithmetic_positive

threshold = resid.component_counts["uniform_threshold_encoder_in_A"]
state_prep = PrimitiveCounts(h=n * q + m)
a_raw = state_prep + compute_raw.scaled(2) + threshold

# qubits: from the register list in build_arithmetic_asian_oracle, the modules
# specific to the control are geometric0 (v), geometric_products (n*w),
# geometric_payoff (t), scaled_geometric_payoff (t) and residual (t).  A raw
# oracle encodes arithmetic_payoff directly, so all five are dropped and every
# other register (price0, price_products, total, arithmetic_payoff, scratch,
# pad, constant, helper, c3temp, equality, equality_work) is retained.
w = model.product_bits
work_resid = resid.a_work_qubits
control_registers = v + n * w + 3 * t
work_raw = work_resid - control_registers
qubits_raw = n * q + m + 1 + work_raw

T_resid, T_raw = resid.a_counts.t, a_raw.t
QUERY_RATIO = 45.85 / 2.864   # manuscript Eq. (fmax-values), k=1

out = {
    "schema": "raw-vs-residual-oracle-v1",
    "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "command": "python scripts/v17_raw_vs_residual_oracle.py",
    "configuration": {"n_dates": n, "price_scale": SPEC.price_scale,
                      "multiplier_fraction_bits": FRACTION_BITS,
                      "shock_model": "binary +/-1"},
    "residual_oracle": {"a_toffoli": resid.a_counts.ccx, "a_t_gates": T_resid,
                        "a_qubits": resid.a_qubits},
    "raw_oracle": {"a_toffoli": a_raw.ccx, "a_t_gates": T_raw, "a_qubits": qubits_raw},
    "register_widths": {"value_bits": v, "product_bits": w, "total_bits": t,
                        "threshold_bits": m, "control_only_registers": control_registers},
    "per_query_control_overhead_t": T_resid / T_raw,
    "per_query_control_overhead_qubits": resid.a_qubits / qubits_raw,
    "query_ratio_k1_from_cutoffs": QUERY_RATIO,
    "net_t_gate_reduction_k1": QUERY_RATIO * T_raw / T_resid,
    "caveats": [
        "Raw oracle retains the residual oracle's threshold encoder unchanged; a "
        "production raw oracle normalising against B_A would need a few more "
        "threshold bits (~1e2 Toffoli against ~1e7, immaterial to the ratio).",
        "Net reduction applies at k=1 only.  Blocked controls k>=2 compute k block "
        "geometric averages and exponentiate each, so their per-query overhead "
        "exceeds 2x and is NOT given by this script.",
        "Counts are logical, not physical: no routing, distillation, or code distance.",
    ],
}
print(json.dumps(out, indent=1))
with open("results/v17/raw_vs_residual_oracle.json", "w") as f:
    json.dump(out, f, indent=1)
