#!/usr/bin/env python3
"""
Validate the European-call quantum circuit against Black-Scholes.

This is the lightweight, deterministic validation path. It reuses
qc_option_pricing.quantum.european_ae.build_european_call_circuit and evaluates
the encoded payoff exactly with a statevector, avoiding slow stochastic AE sweeps.

Outputs:
  results/quantum_european_statevector_table.txt
  results/quantum_european_statevector_convergence.png
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MPLCONFIGDIR = _REPO_ROOT / "results" / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qc_option_pricing.classical import european_call
from qc_option_pricing.quantum.european_ae import build_european_call_circuit

RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
BASIS_GATES = ["ry", "rz", "x", "h", "cx"]


def exact_circuit_price(n_qubits: int) -> tuple[float, float, int, int, dict[str, int]]:
    qc, objective_qubit, post_processing = build_european_call_circuit(
        S0, K, R, SIGMA, T, n_qubits
    )
    state = Statevector.from_instruction(qc)
    raw_amplitude = float(state.probabilities([objective_qubit])[1])
    undiscounted_payoff = float(post_processing(raw_amplitude))
    price = math.exp(-R * T) * undiscounted_payoff

    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=BASIS_GATES)
    transpiled = pm.run(qc)
    return price, raw_amplitude, transpiled.num_qubits, transpiled.depth(), dict(transpiled.count_ops())


def main() -> None:
    bs_price = european_call(S0, K, R, SIGMA, T)
    rows = []
    for n_qubits in range(2, 9):
        price, amp, total_qubits, depth, ops = exact_circuit_price(n_qubits)
        rows.append((n_qubits, price, amp, total_qubits, depth, ops))
        print(
            f"n={n_qubits}: price={price:.6f}, "
            f"|err|={abs(price - bs_price):.6f}, depth={depth}"
        )

    table_path = RESULTS_DIR / "quantum_european_statevector_table.txt"
    with open(table_path, "w") as f:
        f.write("European Call Quantum Circuit Validation\n")
        f.write("=" * 58 + "\n")
        f.write(
            "Source pipeline: Stamatopoulos et al. (2020), Sec. 3.1 and 4.1.1: "
            "load a log-normal terminal distribution, encode the European-call "
            "payoff into an objective qubit, and recover the expected payoff from "
            "the objective-qubit probability. Black-Scholes is the classical benchmark.\n"
        )
        f.write(
            "Gate basis for resource counts: "
            + ", ".join(BASIS_GATES)
            + " (standard simulator gates; no opaque library gates counted).\n"
        )
        f.write(
            f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}, "
            "c_approx=0.10, log-normal bounds=±3 sigma.\n"
        )
        f.write(f"Black-Scholes price: {bs_price:.6f}\n\n")
        f.write(
            f"{'n':>3} {'grid':>6} {'price':>12} {'abs_err':>12} "
            f"{'raw_amp':>12} {'qubits':>8} {'depth':>8} "
            f"{'ry':>7} {'rz':>7} {'x':>7} {'h':>7} {'cx':>7}\n"
        )
        f.write("-" * 103 + "\n")
        for n, price, amp, total_qubits, depth, ops in rows:
            f.write(
                f"{n:>3d} {2**n:>6d} {price:>12.6f} {abs(price - bs_price):>12.6f} "
                f"{amp:>12.8f} {total_qubits:>8d} {depth:>8d} "
                f"{ops.get('ry', 0):>7d} {ops.get('rz', 0):>7d} "
                f"{ops.get('x', 0):>7d} {ops.get('h', 0):>7d} {ops.get('cx', 0):>7d}\n"
            )

    ns_all = np.array([r[0] for r in rows])
    prices_all = np.array([r[1] for r in rows])
    errors_all = np.abs(prices_all - bs_price)

    # The n=2 circuit is kept in the table as a coarse-grid failure mode, but
    # omitting it from the figure makes the convergence region readable.
    plot_mask = ns_all >= 3
    ns = ns_all[plot_mask]
    prices = prices_all[plot_mask]
    errors = errors_all[plot_mask]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax0.plot(
        ns,
        prices,
        "o-",
        color="#2563eb",
        linewidth=1.8,
        markersize=5.5,
        label="Encoded circuit expectation",
    )
    ax0.axhline(
        bs_price,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.4,
        label=f"Black-Scholes = {bs_price:.4f}",
    )
    ax0.set_xlabel(r"Distribution qubits $n$")
    ax0.set_ylabel("Discounted call price")
    ax0.set_title("European Call Circuit Validation")
    ax0.set_xticks(ns)
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=9)

    ax1.semilogy(
        ns,
        errors,
        "s-",
        color="#111827",
        linewidth=1.8,
        markersize=5.2,
        label="Absolute error",
    )
    ax1.set_xlabel(r"Distribution qubits $n$")
    ax1.set_ylabel(r"$|C_{\mathrm{circuit}}-C_{\mathrm{BS}}|$")
    ax1.set_title("Discretization and Payoff-Encoding Error")
    ax1.set_xticks(ns)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=9)

    fig.tight_layout()
    fig_path = RESULTS_DIR / "quantum_european_statevector_convergence.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {table_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
