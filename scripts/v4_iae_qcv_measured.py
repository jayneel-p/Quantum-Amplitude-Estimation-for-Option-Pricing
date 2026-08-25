#!/usr/bin/env python3
"""v4: measured IAE oracle queries, raw versus residual encoding, at equal
dollar error.

For each finite-grid Asian circuit
(2, 3, 4 dates) and each dollar-error target eps, IterativeAmplitudeEstimation
runs on the raw-payoff encoding and on the residual encodings at the amplitude
target eps_a = eps * e^{rT} / f_max implied by each encoding's payoff bound,
with the same sampler stack as the rest of the repo.  Reported: median oracle
queries, measured price errors, the measured query ratio, and the ratio from
the empirical coefficient 1.4 reported by Grinko et al.  Their Theorem 1 uses
coefficient 50; the 1.4 curve is not a rigorous worst-case bound.

Gates (fail-closed): median dollar error within target for every
configuration; measured ratio > 3 everywhere.

Outputs: results/v4/iae_qcv_measured.txt, results/v4/iae_qcv_measured.png
"""
from __future__ import annotations

import itertools
import math
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
from scipy.stats import norm

S0, K, RATE, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
ALPHA = 0.32
N_TRIALS = 8
TARGETS = {2: [0.20, 0.10, 0.05], 3: [0.20, 0.10, 0.05], 4: [0.20, 0.10]}
OUT = _REPO / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def grinko_empirical_envelope(eps_a: float) -> float:
    return (1.4 / eps_a) * math.log((2.0 / ALPHA) * math.log2(math.pi / (4.0 * eps_a)))


def normal_bin_grid():
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    probs = np.diff(norm.cdf(edges))
    reps = [(norm.pdf(lo) - norm.pdf(hi)) / p
            for lo, hi, p in zip(edges[:-1], edges[1:], probs)]
    return np.array(reps), probs


def path_payoffs(n_dates: int):
    shocks, shock_probs = normal_bin_grid()
    dt = T / n_dates
    drift = (RATE - 0.5 * SIGMA**2) * dt
    vol = SIGMA * math.sqrt(dt)
    probs, pay = [], {"A": [], "G": [], "B2": []}
    for idx in itertools.product(range(4), repeat=n_dates):
        s, prices, p = S0, [], 1.0
        for i in idx:
            s *= math.exp(drift + vol * shocks[i])
            prices.append(s)
            p *= shock_probs[i]
        prices = np.array(prices)
        probs.append(p)
        pay["A"].append(max(prices.mean() - K, 0.0))
        pay["G"].append(max(math.exp(np.log(prices).mean()) - K, 0.0))
        if n_dates % 2 == 0:
            h = n_dates // 2
            b2 = 0.5 * (math.exp(np.log(prices[:h]).mean())
                        + math.exp(np.log(prices[h:]).mean()))
            pay["B2"].append(max(b2 - K, 0.0))
    return np.array(probs), {k: np.array(v) for k, v in pay.items() if v}


def joint_state(probs, f_norm):
    """Exact joint state |i>(sqrt(1-f)|0> + sqrt(f)|1>), objective last qubit."""
    a0 = np.sqrt(probs * (1.0 - f_norm))
    a1 = np.sqrt(probs * f_norm)
    return np.concatenate([a0, a1])


def encoding_circuit(probs, payoffs):
    f_max = float(payoffs.max())
    psi = joint_state(probs, np.clip(payoffs / f_max, 0.0, 1.0))
    n = int(math.log2(len(psi)))
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(psi), range(n))
    return qc, n - 1, f_max


def plot_two_date(rows) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    eps_list = TARGETS[2]
    for tag, color, label in (("raw", "#c0392b", "raw payoff"),
                              ("R1", "#16a34a", "residual ($k=1$)")):
        qs = [r["med"][tag]["q"] for r in rows if r["n"] == 2]
        ax.loglog(eps_list, qs, "o-", color=color, markersize=7,
                  linewidth=2.0, zorder=3, label=f"{label}, measured")
        envelope = [r["med"][tag]["envelope"] for r in rows if r["n"] == 2]
        ax.loglog(eps_list, envelope, "--", color=color, linewidth=1.6,
                  alpha=0.85, zorder=2,
                  label=f"{label}, empirical 1.4 envelope")
    ax.set_xlabel("dollar error target $\\varepsilon$ (\\$)")
    ax.set_ylabel("$A$-operator applications")
    ax.set_title("Measured IAE cost at equal dollar error (2-date grid)")
    ax.set_xticks(eps_list)
    ax.set_xticklabels([f"{e:.2f}" for e in eps_list])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(0.045, 0.225)
    ax.grid(True, which="major", axis="both", alpha=0.35, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.legend(fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "iae_qcv_measured.png", dpi=200, bbox_inches="tight")


def load_existing_rows():
    grouped = {}
    for line in (OUT / "iae_qcv_measured.txt").read_text().splitlines():
        fields = line.split()
        if len(fields) != 8 or not fields[0].isdigit():
            continue
        n_dates = int(fields[0])
        eps = float(fields[1])
        tag = fields[2]
        eps_a = float(fields[3])
        q = float(fields[4].replace(",", ""))
        key = (n_dates, eps)
        grouped.setdefault(key, {"n": n_dates, "eps": eps, "med": {}})
        grouped[key]["med"][tag] = {
            "q": q,
            "eps_a": eps_a,
            "envelope": grinko_empirical_envelope(eps_a),
        }
    order = [(n, eps) for n, targets in TARGETS.items() for eps in targets]
    return [grouped[key] for key in order if key in grouped]


def main() -> int:
    disc = math.exp(-RATE * T)
    rows = []
    for n_dates, targets in TARGETS.items():
        probs, pay = path_payoffs(n_dates)
        encs = {"raw": pay["A"], "R1": np.maximum(pay["A"] - pay["G"], 0.0)}
        if n_dates == 4:
            encs["R2"] = np.maximum(pay["A"] - pay["B2"], 0.0)
        built = {}
        for tag, payoffs in encs.items():
            qc, obj, fm = encoding_circuit(probs, payoffs)
            a_exact = float(probs @ payoffs) / fm
            built[tag] = (qc, obj, fm, a_exact)
        for eps in targets:
            med = {}
            for tag, (qc, obj, fm, a_exact) in built.items():
                eps_a = min(0.45, eps * math.exp(RATE * T) / fm)
                acalls, errs = [], []
                shots_per_round = 1024
                for trial in range(N_TRIALS):
                    sampler = StatevectorSampler(
                        default_shots=shots_per_round,
                        seed=1000 * n_dates + trial)
                    iae = IterativeAmplitudeEstimation(
                        epsilon_target=eps_a, alpha=ALPHA, sampler=sampler)
                    problem = EstimationProblem(
                        state_preparation=qc, objective_qubits=[obj])
                    t0 = time.perf_counter()
                    res = iae.estimate(problem)
                    # each shot of a round with Grover power k applies A (2k+1)
                    # times; num_oracle_queries counts shots*k over rounds
                    ac = 2 * res.num_oracle_queries \
                        + shots_per_round * len(res.powers)
                    acalls.append(ac)
                    errs.append(disc * fm * abs(res.estimation - a_exact))
                    if trial == 0:
                        print(f"  n={n_dates} {tag} eps=${eps:.2f} "
                              f"(eps_a={eps_a:.3g}): {ac:,} A-calls "
                              f"({res.num_oracle_queries:,} Grover), "
                              f"{time.perf_counter()-t0:.1f}s", flush=True)
                med[tag] = dict(q=float(np.median(acalls)),
                                err=float(np.median(errs)),
                                eps_a=eps_a, envelope=grinko_empirical_envelope(eps_a))
            row = dict(n=n_dates, eps=eps, med=med)
            rows.append(row)
            for tag in med:
                check(f"err n={n_dates} {tag} eps={eps}",
                      med[tag]["err"] <= eps,
                      f"median |price err| {med[tag]['err']:.4f} <= {eps}")
            for tag in [t for t in med if t != "raw"]:
                ratio = med["raw"]["q"] / med[tag]["q"]
                check(f"ratio n={n_dates} {tag} eps={eps}", ratio > 3.0,
                      f"measured {ratio:.1f}x (empirical-envelope ratio "
                      f"{med['raw']['envelope']/med[tag]['envelope']:.1f}x)")

    lines = ["Measured IAE oracle queries at equal dollar error, raw vs residual",
             f"alpha={ALPHA}, {N_TRIALS} trials/point, medians reported",
             "",
             "  n_dates  eps($)  enc   eps_a      A-calls    |err|($)   "
             "meas.ratio  envelope.ratio"]
    for row in rows:
        for tag, m in row["med"].items():
            fm = {"raw": None, "R1": None, "R2": None}
            ratio = (f"{row['med']['raw']['q']/m['q']:10.1f}"
                     if tag != "raw" else "         -")
            bratio = (f"{row['med']['raw']['envelope']/m['envelope']:14.1f}"
                      if tag != "raw" else "          -")
            lines.append(f"     {row['n']}     {row['eps']:.2f}   {tag:3s} "
                         f"        {m['eps_a']:8.3g}  {m['q']:9,.0f}  "
                         f"{m['err']:8.4f}  {ratio} {bratio}")
    lines += ["", "checks: " + ("ALL PASS" if not FAILURES else f"FAILED {FAILURES}")]
    (OUT / "iae_qcv_measured.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_two_date(rows)
    print(f"-> wrote {OUT}/iae_qcv_measured.png")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--plot-existing"]:
        plot_two_date(load_existing_rows())
        print(f"-> replotted {OUT}/iae_qcv_measured.png from existing table")
        raise SystemExit(0)
    raise SystemExit(main())
