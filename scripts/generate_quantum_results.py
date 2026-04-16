#!/usr/bin/env python3
"""Generate quantum figures/tables under results/. Use --quick for circuits + resource table only."""

from __future__ import annotations

import os
import sys
import warnings
import math
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MPLCONFIGDIR = _REPO_ROOT / "results" / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qc_option_pricing.classical import european_call
from qc_option_pricing.quantum.european_ae import (
    build_european_call_circuit,
    price_european_call_quantum,
)
from qc_option_pricing.visualization import save_european_a_operator_sta20_style

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
BS_PRICE = european_call(S0, K, R, SIGMA, T)

RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_convergence():
    print("=" * 60)
    print("QUANTUM EUROPEAN CALL — CONVERGENCE vs DISTRIBUTION QUBITS")
    print("=" * 60)
    print(f"Black-Scholes price: {BS_PRICE:.4f}\n")

    qubit_range = [3, 4, 5, 6, 7, 8]
    n_trials = 5

    mean_prices, std_prices = [], []
    mean_queries = []

    for nq in qubit_range:
        trial_prices, trial_queries = [], []
        for trial in range(n_trials):
            res = price_european_call_quantum(
                S0, K, R, SIGMA, T,
                n_qubits=nq, ae_method="iae",
                epsilon=0.005, alpha=0.05,
            )
            trial_prices.append(res.price)
            trial_queries.append(res.n_oracle_queries)
        mp = np.mean(trial_prices)
        sp = np.std(trial_prices, ddof=1)
        mq = np.mean(trial_queries)
        mean_prices.append(mp)
        std_prices.append(sp)
        mean_queries.append(mq)
        print(f"  n_qubits={nq:2d}  |  price={mp:.4f} ± {sp:.4f}  |  "
              f"|err|={abs(mp - BS_PRICE):.4f}  |  queries={mq:,.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(qubit_range, mean_prices, yerr=std_prices,
                 fmt="s-", color="#7c3aed", capsize=4, markersize=6,
                 label=f"QAE price (IAE, {n_trials} trials)")
    ax1.axhline(BS_PRICE, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Black-Scholes = {BS_PRICE:.4f}")
    ax1.set_xlabel("Distribution qubits  $n$", fontsize=12)
    ax1.set_ylabel("Option price estimate", fontsize=12)
    ax1.set_title("QAE vs Black–Scholes (IAE, fixed $\\epsilon$)", fontsize=13)
    ax1.set_xticks(qubit_range)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    errors = [abs(mp - BS_PRICE) for mp in mean_prices]
    ax2.semilogy(qubit_range, errors, "o-", color="#7c3aed", markersize=6,
                 label="|QAE − BS|")

    ref_x = np.array(qubit_range)
    ref_y = errors[0] * (2.0 ** qubit_range[0]) / (2.0 ** ref_x)
    ax2.semilogy(ref_x, ref_y, ":", color="gray", linewidth=1.5,
                 label=r"Illustrative $2^{-n}$ mesh scaling (anchored)")

    ax2.set_xlabel("Distribution qubits  $n$", fontsize=12)
    ax2.set_ylabel("|QAE − Black–Scholes|", fontsize=12)
    ax2.set_title(
        "|QAE − BS| (grid + payoff map + IAE tolerance, not pure discretisation)",
        fontsize=11,
    )
    ax2.set_xticks(qubit_range)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("European Call — Quantum Amplitude Estimation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "quantum_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → Saved quantum_convergence.png")

    return qubit_range, mean_prices, std_prices, mean_queries


def run_circuit_diagrams():
    print("\n" + "=" * 60)
    print("CIRCUIT DIAGRAMS (Sta20-style schematic; transpiled stats below)")
    print("=" * 60)

    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["cx", "u"])

    for nq in [3, 5]:
        qc, _, _ = build_european_call_circuit(S0, K, R, SIGMA, T, nq)
        qc_transpiled = pm.run(qc)

        ops = dict(qc_transpiled.count_ops())
        depth = qc_transpiled.depth()
        n_cx = ops.get("cx", 0)
        n_u = ops.get("u", 0)
        print(f"\n  n_qubits={nq}: total_qubits={qc_transpiled.num_qubits}, "
              f"depth={depth}, CX={n_cx}, U={n_u}, total_gates={n_cx + n_u}")
        fig_path = RESULTS_DIR / f"quantum_circuit_{nq}q.png"
        save_european_a_operator_sta20_style(nq, fig_path)
        print(f"  → Saved {fig_path.name}")


def run_resource_table():
    print("\n" + "=" * 60)
    print("RESOURCE TABLE (transpiled to CX + U basis)")
    print("=" * 60)

    qubit_range = [3, 4, 5, 6, 7, 8]
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["cx", "u"])

    rows = []
    print(f"\n  {'n':>4s} {'Qubits':>8s} {'Depth':>8s} {'CX':>8s} {'U':>8s} "
          f"{'Total':>8s} {'2^n grid':>10s}")
    print("  " + "-" * 56)

    for nq in qubit_range:
        qc, _, _ = build_european_call_circuit(S0, K, R, SIGMA, T, nq)
        qc_t = pm.run(qc)
        ops = dict(qc_t.count_ops())
        depth = qc_t.depth()
        n_cx = ops.get("cx", 0)
        n_u = ops.get("u", 0)
        total = n_cx + n_u
        grid = 2 ** nq
        rows.append((nq, qc_t.num_qubits, depth, n_cx, n_u, total, grid))
        print(f"  {nq:>4d} {qc_t.num_qubits:>8d} {depth:>8d} {n_cx:>8d} {n_u:>8d} "
              f"{total:>8d} {grid:>10d}")

    with open(RESULTS_DIR / "quantum_resource_table.txt", "w") as f:
        f.write("European Call — Quantum Circuit Resources\n")
        f.write(f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}\n")
        f.write("Transpiled to CX + U basis (optimization_level=1)\n\n")
        f.write(f"{'n_dist':>8s} {'Qubits':>8s} {'Depth':>8s} {'CX':>8s} {'U':>8s} "
                f"{'Total':>8s} {'Grid pts':>10s}\n")
        f.write("-" * 58 + "\n")
        for nq, nqt, d, cx, u, tot, g in rows:
            f.write(f"{nq:>8d} {nqt:>8d} {d:>8d} {cx:>8d} {u:>8d} {tot:>8d} {g:>10d}\n")
    print(f"\n  → Saved quantum_resource_table.txt")

    return rows


def run_ae_scaling():
    print("\n" + "=" * 60)
    print("IAE ORACLE QUERY SCALING vs EPSILON")
    print("=" * 60)

    nq = 6
    epsilons = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    n_trials = 3

    mean_queries_list, mean_errors = [], []

    for eps in epsilons:
        trial_queries, trial_errors = [], []
        for trial in range(n_trials):
            res = price_european_call_quantum(
                S0, K, R, SIGMA, T,
                n_qubits=nq, ae_method="iae",
                epsilon=eps, alpha=0.05,
            )
            trial_queries.append(res.n_oracle_queries)
            trial_errors.append(abs(res.price - BS_PRICE))
        mq = np.mean(trial_queries)
        me = np.mean(trial_errors)
        mean_queries_list.append(mq)
        mean_errors.append(me)
        print(f"  eps={eps:.3f}  |  queries={mq:>10,.0f}  |  |err|={me:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.loglog(
        epsilons,
        mean_queries_list,
        "s-",
        color="#7c3aed",
        markersize=6,
        linewidth=2,
        zorder=3,
        label=f"IAE oracle queries ({nq} dist. qubits)",
    )

    ref_eps = np.array(epsilons)
    eps0 = float(epsilons[0])
    q0 = float(mean_queries_list[0])
    ref_q = q0 * eps0 / ref_eps
    ref_mc = q0 * (eps0**2) / (ref_eps**2)
    ax.loglog(
        ref_eps,
        ref_q,
        ":",
        color="#7c3aed",
        alpha=0.85,
        linewidth=2,
        zorder=2,
        label=r"$O(1/\epsilon)$ slope (anchored at $\epsilon_{\mathrm{max}}$)",
    )
    ax.loglog(
        ref_eps,
        ref_mc,
        "--",
        color="#dc2626",
        alpha=0.85,
        linewidth=2,
        zorder=2,
        label=r"$O(1/\epsilon^2)$ slope (anchored at $\epsilon_{\mathrm{max}}$)",
    )

    ax.set_xlabel(r"Target accuracy  $\epsilon$ (IAE)", fontsize=12)
    ax.set_ylabel("Oracle query count", fontsize=12)
    ax.set_title(
        "IAE cost vs $\\epsilon$ (dashed curves: schematic scaling, same anchor)",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(
        0.5,
        0.02,
        "Dashed references share the IAE count at the coarsest ε for slope comparison only; "
        "they are not Monte Carlo sample counts.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(RESULTS_DIR / "quantum_ae_scaling.png", dpi=200)
    plt.close(fig)
    print(f"\n  → Saved quantum_ae_scaling.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate quantum figures and tables.")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only circuit PNGs + resource table (skip IAE convergence and ε scaling).",
    )
    args = ap.parse_args()

    print(f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}")
    print(f"Black-Scholes: {BS_PRICE:.4f}\n")

    if args.quick:
        print("(quick mode: no IAE sweeps)\n")
        run_circuit_diagrams()
        run_resource_table()
    else:
        run_convergence()
        run_circuit_diagrams()
        run_resource_table()
        run_ae_scaling()

    print("\nDone! All quantum results in results/ directory.")
