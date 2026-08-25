"""Matched-error resource comparison: Wang-Kan weak Euler vs + control variate.

Reproduces the published weak-Euler module totals (Tables 6-7), attaches the
control-variate payoff arithmetic, and compares total T-counts at common
dollar amplitude-estimation budgets for three constructions:

  raw            Wang-Kan Asian payoff, range Z (their normalization)
  cv_rotation    same U3 rotation encoder, residual payoff, certified cap
  cv_threshold   repository threshold encoder, power-of-two padded cap

Caps and clipping certificates come from results/v10/wang_kan_cv_feasibility.json
(production part).  Regime A uses Wang-Kan's epsilon_estimate = 1e-3 dollar
equivalent; Regime B uses the handoff's $0.003 amplitude-estimation
sub-budget of a $0.01 total target.  The geometric-control restoration price
is a separate classical input and is charged as classical Monte Carlo paths
at the matched sub-budget; the exact discrete-Heston formula path remains
UNVERIFIED (primary source inaccessible at run time).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from qc_option_pricing.quantum.wang_kan_resources import (
    MatchedComparison,
    calibrate_ppoly,
    delta_threshold_encoder,
    delta_u2_control_variate,
    n_oracle_queries,
    t_add,
    t_add_const,
    t_arcsin_sqrt,
    t_mul_const,
    t_q,
    t_toffoli,
    t_u1_weak,
    t_u2_asian,
    t_u3,
    t_usin,
)

ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY = ROOT / "results" / "v10" / "wang_kan_cv_feasibility.json"
OUT = ROOT / "results" / "v10" / "wang_kan_cv_resources.json"

PUBLISHED = {
    "call_instance_1": {
        "n": 27, "p": 11, "n_steps": 256, "z": 200.0, "rate": 0.03,
        "u1": 6.4e6, "u2": 9.3e6, "u3": 6.0e4, "q": 3.2e7,
        "total": 2.4e11, "queries": 7363, "qubits": 2.2e4,
        "eps_sin": 1e-8, "eps_estimate": 1e-3, "delta": 0.1,
    },
    "put_instance_2": {
        "n": 27, "p": 10, "n_steps": 256, "z": 100.0, "rate": 0.05,
        "u1": 6.4e6, "u2": 9.2e6, "u3": 6.0e4, "q": 3.2e7,
        "total": 2.3e11, "queries": 7363, "qubits": 2.2e4,
        "eps_sin": 1e-8, "eps_estimate": 1e-3, "delta": 0.1,
    },
}
THRESHOLD_BITS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    feas = json.loads(FEASIBILITY.read_text())
    result: dict = {
        "schema_version": "wang-kan-cv-resources-v1",
        "created_at_start": started,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "source_hashes": {
            "src/qc_option_pricing/quantum/wang_kan_resources.py": sha256(
                ROOT / "src" / "qc_option_pricing" / "quantum" / "wang_kan_resources.py"),
            "scripts/validate_wang_kan_cv_resources.py": sha256(Path(__file__)),
            "tests/test_wang_kan_resources.py": sha256(
                ROOT / "tests" / "test_wang_kan_resources.py"),
            "results/v10/wang_kan_cv_feasibility.json": sha256(FEASIBILITY),
        },
        "notes": [
            "Published module totals are rounded to two significant figures; "
            "the EXP and ARCSIN piecewise-polynomial (M, d) are not published "
            "and are calibrated against the published U2 and U3 totals; every "
            "candidate within 2% is recorded.",
            "Eq. (64) at eps=1e-3, delta=0.1 evaluates to 7363.116; the "
            "published 7363 implies nearest rounding, a strict bound requires "
            "7364.",
            "Our calibrated total for instance 1 is 2.33e11, inside the "
            "interval implied by the published rounded factors (7363 x "
            "[3.15, 3.25]e7) though it rounds to 2.3e11, not the published "
            "2.4e11.",
            "The fixed-point/function-approximation error at (n, p) = (27, "
            "11)/(27, 10) is shared by all three constructions and is not "
            "reduced by the control variate; a $0.01 total dollar target "
            "requires precision refinement in every arm.  Ratios below "
            "compare amplitude-estimation cost at equal encoded-range "
            "budgets.",
            "The geometric control price is a classical input: either the "
            "discrete-Heston formula of Kim et al. (2016), UNVERIFIED "
            "(primary source inaccessible at run time), or classical Monte "
            "Carlo charged below.",
        ],
        "instances": {},
    }

    for name, pub in PUBLISHED.items():
        n, p, n_steps = pub["n"], pub["p"], pub["n_steps"]
        disc = math.exp(-pub["rate"] * 1.0)
        prod = feas["production"]["instances"][name]
        cap = prod["residual_cap_study"]["selected"]["cap"]
        clip_bias = prod["residual_cap_study"]["selected"]["clipping_bias"]
        sd_raw = prod["pooled"]["payoff_sd_dollars"]
        var_ratio = prod["pooled"]["variance_ratio_raw_over_residual"]

        u1 = t_u1_weak(n_steps, n, p)
        fixed = ((n_steps - 1) * t_add(n) + t_mul_const(n, p)
                 + t_add_const(n) + n * t_toffoli(3))
        exp_candidates = calibrate_ppoly((pub["u2"] - fixed) / n_steps, n, p)
        t_exp = exp_candidates[0]["t_count"]
        u2 = t_u2_asian(n_steps, n, p, t_exp)
        usin = t_usin(n, pub["eps_sin"])
        arc_candidates = calibrate_ppoly(
            pub["u3"] - usin, n, p,
            builder=lambda nn, pp, m, d: t_arcsin_sqrt(nn, pp, m, d))
        u3 = t_u3(n, p, arc_candidates[0]["m_pieces"],
                  arc_candidates[0]["degree"], pub["eps_sin"])
        t_a_raw = u1 + u2 + u3
        t_q_raw = t_q(t_a_raw, int(pub["qubits"]))

        delta_cv = delta_u2_control_variate(n_steps, n, p, t_exp)
        t_a_cv_rot = t_a_raw + delta_cv
        t_q_cv_rot = t_q(t_a_cv_rot, int(pub["qubits"]) + 3 * n)
        t_a_cv_thr = t_a_raw + delta_cv - u3 + delta_threshold_encoder(n, THRESHOLD_BITS)
        t_q_cv_thr = t_q(t_a_cv_thr, int(pub["qubits"]) + 3 * n + THRESHOLD_BITS)

        padded_cap = 2.0 ** math.ceil(math.log2(cap))
        regimes = {}
        for regime, dollar_ae in (
            ("regime_A_wang_kan_epsilon", disc * pub["z"] * pub["eps_estimate"]),
            ("regime_B_ae_subbudget_of_one_cent", 0.003),
        ):
            rows = {}
            for label, rng, tq in (
                ("raw", pub["z"], t_q_raw),
                ("cv_rotation", cap, t_q_cv_rot),
                ("cv_threshold", padded_cap, t_q_cv_thr),
            ):
                rows[label] = MatchedComparison(
                    dollar_ae_budget=dollar_ae, delta=pub["delta"],
                    discount=disc, range_raw=pub["z"], range_cv=rng,
                    t_q_raw=t_q_raw, t_q_cv=tq).row() if label != "raw" else {
                        "eps_amplitude": dollar_ae / (disc * pub["z"]),
                        "queries": n_oracle_queries(
                            dollar_ae / (disc * pub["z"]), pub["delta"]),
                        "t_q": t_q_raw,
                        "total_t": n_oracle_queries(
                            dollar_ae / (disc * pub["z"]), pub["delta"]) * t_q_raw,
                    }
            regimes[regime] = {"dollar_ae_budget": dollar_ae, **rows}

        classical_mu_g_paths = (disc * sd_raw / 0.003) ** 2
        result["instances"][name] = {
            "published": pub,
            "reproduction": {
                "u1_exact": u1,
                "u2_calibrated": u2,
                "u3_calibrated": u3,
                "t_a": t_a_raw,
                "t_q": t_q_raw,
                "queries_eq64_nearest": n_oracle_queries(
                    pub["eps_estimate"], pub["delta"]),
                "queries_eq64_ceil": n_oracle_queries(
                    pub["eps_estimate"], pub["delta"], rounding="ceil"),
                "total_t_at_published_queries": pub["queries"] * t_q_raw,
                "exp_calibration_candidates": exp_candidates[:5],
                "arcsin_calibration_candidates": arc_candidates[:5],
            },
            "control_variate": {
                "certified_cap_dollars": cap,
                "clipping_bias_discounted": clip_bias,
                "clipping_budget": 0.001,
                "padded_cap_threshold_encoder": padded_cap,
                "padding_factor": padded_cap / cap,
                "range_ratio_rotation": pub["z"] / cap,
                "range_ratio_threshold": pub["z"] / padded_cap,
                "classical_variance_ratio": var_ratio,
                "delta_u2_t_count": delta_cv,
                "t_q_overhead_rotation": t_q_cv_rot / t_q_raw,
                "t_q_overhead_threshold": t_q_cv_thr / t_q_raw,
            },
            "matched_comparison": regimes,
            "classical_geometric_control_input": {
                "method": "classical Monte Carlo (formula path UNVERIFIED)",
                "target_se_dollars": 0.003,
                "required_paths": classical_mu_g_paths,
            },
        }
        ra = regimes["regime_A_wang_kan_epsilon"]
        print(f"{name}:")
        print(f"  reproduction: U1={u1:,} U2={u2:,.0f} U3={u3:,.0f} "
              f"T(Q)={t_q_raw:,.0f}")
        print(f"  cap ${cap:.3f} (clip bias ${clip_bias:.2e}), "
              f"range ratio {pub['z']/cap:.1f}x rotation / "
              f"{pub['z']/padded_cap:.1f}x threshold")
        print(f"  Regime A: raw {ra['raw']['queries']:,} queries, "
              f"cv_rotation {ra['cv_rotation']['queries_cv']:,} "
              f"-> total-T ratio {ra['cv_rotation']['total_t_ratio']:.1f}x "
              f"(threshold {ra['cv_threshold']['total_t_ratio']:.1f}x)")
        rb = regimes["regime_B_ae_subbudget_of_one_cent"]
        print(f"  Regime B ($0.003 AE): total-T ratio "
              f"{rb['cv_rotation']['total_t_ratio']:.1f}x rotation / "
              f"{rb['cv_threshold']['total_t_ratio']:.1f}x threshold")

    result["created_at_end"] = datetime.now(
        timezone.utc).astimezone().isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    tmp.replace(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
