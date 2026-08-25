#!/usr/bin/env python3
"""Measured IAE calls versus target amplitude error, with the empirical
Clopper--Pearson envelope reported by Grinko et al. overlaid.

Runs IterativeAmplitudeEstimation on the European-call circuit (n=6 distribution
qubits, c=0.10) at alpha = 0.32 (68% confidence, matching the paper's convention
and Chakrabarti Sec. 4.1.2), across a grid of amplitude-error targets.

Empirical envelope (Grinko et al. Eq. 27 / Chakrabarti Eq. 19):
    N_emp(eps_a) = (1.4 / eps_a) * ln( (2/alpha) * log2(pi / (4 eps_a)) )

The coefficient 1.4 is empirical.  Grinko et al. Theorem 1 uses coefficient 50.

Outputs: results/v3/iae_scaling.png, results/v3/iae_scaling.txt,
results/v3/iae_scaling.json, and results/v3/iae_scaling_trials.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_option_pricing.classical import european_call
from qc_option_pricing.quantum.european_ae import price_european_call_quantum

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N_QUBITS = 6
ALPHA = 0.32                      # 68% confidence, matches Chakrabarti convention
C_APPROX = 0.10
EPS_GRID = [2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4]   # amplitude-error targets
N_TRIALS = 15
MASTER_SEED = 20_260_716
DEFAULT_SHOTS = 1024

OUT = _REPO / "results" / "v3"
OUT.mkdir(parents=True, exist_ok=True)


def grinko_empirical_envelope(eps_a: float, alpha: float = ALPHA) -> float:
    return (1.4 / eps_a) * math.log((2.0 / alpha) * math.log2(math.pi / (4.0 * eps_a)))


def main() -> int:
    bs = european_call(S0, K, R, SIGMA, T)
    rows = []
    trial_rows = []
    for point_index, eps in enumerate(EPS_GRID):
        queries, errs = [], []
        for trial in range(N_TRIALS):
            seed = MASTER_SEED + point_index * N_TRIALS + trial
            t0 = time.perf_counter()
            res = price_european_call_quantum(
                S0, K, R, SIGMA, T,
                n_qubits=N_QUBITS, ae_method="iae",
                epsilon=eps, alpha=ALPHA, c_approx=C_APPROX,
                shots=DEFAULT_SHOTS, seed=seed,
            )
            dt = time.perf_counter() - t0
            queries.append(res.n_oracle_queries)
            errs.append(abs(res.price - bs))
            trial_rows.append({
                "eps_a": eps,
                "point_index": point_index,
                "trial_index": trial,
                "seed": seed,
                "sampler_default_shots": res.sampler_default_shots,
                "price": float(res.price),
                "undiscounted_payoff": float(res.undiscounted_payoff),
                "raw_amplitude": float(res.raw_amplitude),
                "absolute_price_error": float(abs(res.price - bs)),
                "num_oracle_queries": int(res.n_oracle_queries),
                "powers": list(res.powers),
                "ratios": list(res.ratios),
                "round_shots": list(res.round_shots),
                "round_oracle_queries": list(res.round_oracle_queries),
                "confidence_interval": res.confidence_interval,
                "confidence_interval_processed": res.confidence_interval_processed,
                "estimate_intervals": res.estimate_intervals,
            })
            print(f"eps_a={eps:.0e} trial {trial+1}/{N_TRIALS} seed={seed}: "
                  f"queries={res.n_oracle_queries:,}, |err|={abs(res.price-bs):.5f}, {dt:.1f}s",
                  flush=True)
        rows.append({
            "eps_a": eps,
            "mean_queries": float(np.mean(queries)),
            "median_queries": float(np.median(queries)),
            "std_queries": float(np.std(queries, ddof=1)) if len(queries) > 1 else 0.0,
            "mean_abs_price_err": float(np.mean(errs)),
            "empirical_envelope": grinko_empirical_envelope(eps),
        })

    with open(OUT / "iae_scaling.txt", "w") as f:
        f.write("IAE measured oracle queries vs amplitude-error target\n")
        f.write(f"European call, n={N_QUBITS} dist qubits, c={C_APPROX}, alpha={ALPHA} "
                f"(68% conf), {N_TRIALS} trials/point\n")
        f.write(f"master_seed={MASTER_SEED}; default_shots={DEFAULT_SHOTS}; "
                "trial schedules and counts are in iae_scaling_trials.json\n")
        f.write("NOTE: with the linear-rescaling payoff (c=0.10), price error = "
                "e^-rT * (f_max/(2c)) * amplitude error.\n\n")
        f.write(f"{'eps_a':>9} {'mean_queries':>14} {'median':>12} {'std':>12} "
                f"{'emp_envelope':>14} {'med/emp':>7} {'mean|price err|':>16}\n")
        for r_ in rows:
            f.write(f"{r_['eps_a']:>9.0e} {r_['mean_queries']:>14,.0f} "
                    f"{r_['median_queries']:>12,.0f} {r_['std_queries']:>12,.0f} "
                    f"{r_['empirical_envelope']:>14,.0f} "
                    f"{r_['median_queries']/r_['empirical_envelope']:>7.2f} "
                    f"{r_['mean_abs_price_err']:>16.6f}\n")
    (OUT / "iae_scaling.json").write_text(json.dumps(rows, indent=1))
    trial_payload = {
        "schema": "parikh-rayan-iae-trials-v1",
        "config": {
            "s0": S0,
            "strike": K,
            "rate": R,
            "volatility": SIGMA,
            "maturity": T,
            "distribution_qubits": N_QUBITS,
            "alpha": ALPHA,
            "c_approx": C_APPROX,
            "epsilon_grid": EPS_GRID,
            "trials_per_point": N_TRIALS,
            "sampler_default_shots": DEFAULT_SHOTS,
            "master_seed": MASTER_SEED,
            "seed_rule": "master_seed + point_index * trials_per_point + trial_index",
            "black_scholes_price": float(bs),
        },
        "trials": trial_rows,
    }
    (OUT / "iae_scaling_trials.json").write_text(json.dumps(trial_payload, indent=1))

    eps = np.array([r_["eps_a"] for r_ in rows])
    q = np.array([r_["median_queries"] for r_ in rows])
    sd = np.array([r_["std_queries"] for r_ in rows])
    wc = np.array([r_["empirical_envelope"] for r_ in rows])

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.errorbar(eps, q, yerr=sd, fmt="s-", color="#7c3aed", capsize=3,
                label=f"measured IAE queries (median of {N_TRIALS} trials)")
    ax.loglog(eps, wc, "--", color="#c0392b",
              label=r"Grinko empirical $1.4$ envelope $\frac{1.4}{\epsilon_a}\ln(\frac{2}{\alpha}\log_2\frac{\pi}{4\epsilon_a})$")
    ax.loglog(eps, q[0] * eps[0] / eps, ":", color="grey",
              label=r"$\propto 1/\epsilon_a$ (anchored)")
    ax.set_xlabel(r"target amplitude error $\epsilon_a$")
    ax.set_ylabel("oracle queries (applications of $Q$)")
    ax.set_title(f"IAE query cost, European call ($n={N_QUBITS}$, $\\alpha={ALPHA}$)")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "iae_scaling.png", dpi=200, bbox_inches="tight")
    print(f"-> wrote {OUT}/iae_scaling.{{png,txt,json}} and iae_scaling_trials.json")
    return 0


def replot_existing() -> int:
    rows = json.loads((OUT / "iae_scaling.json").read_text())
    for row in rows:
        if "empirical_envelope" not in row:
            row["empirical_envelope"] = row.pop("wc_bound")
    eps = np.array([r_["eps_a"] for r_ in rows])
    q = np.array([r_["median_queries"] for r_ in rows])
    sd = np.array([r_["std_queries"] for r_ in rows])
    empirical = np.array([r_["empirical_envelope"] for r_ in rows])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.errorbar(eps, q, yerr=sd, fmt="s-", color="#7c3aed", capsize=3,
                label=f"measured IAE queries (median of {N_TRIALS} trials)")
    ax.loglog(eps, empirical, "--", color="#c0392b",
              label=r"Grinko empirical $1.4$ envelope")
    ax.loglog(eps, q[0] * eps[0] / eps, ":", color="grey",
              label=r"$\propto 1/\epsilon_a$ (anchored)")
    ax.set_xlabel(r"target amplitude error $\epsilon_a$")
    ax.set_ylabel("oracle queries (applications of $Q$)")
    ax.set_title(f"IAE query cost, European call ($n={N_QUBITS}$, $\\alpha={ALPHA}$)")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "iae_scaling.png", dpi=200, bbox_inches="tight")
    (OUT / "iae_scaling.json").write_text(json.dumps(rows, indent=1))
    print(f"-> replotted {OUT}/iae_scaling.png from existing JSON")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--plot-existing"]:
        raise SystemExit(replot_existing())
    raise SystemExit(main())
