#!/usr/bin/env python3
"""
Generate a small arithmetic-Asian quantum toy validation.

The circuit follows the Stamatopoulos et al. path-dependent idea at toy scale:
prepare a finite grid over two GBM time steps, encode
max((S_1 + S_2)/2 - K, 0) into an objective qubit, and read the expected payoff
from that qubit's probability. The payoff rotation is exact on the finite grid.

Outputs:
  results/quantum_asian_toy_table.txt
  results/quantum_asian_toy_validation.png
"""

from __future__ import annotations

import argparse
import itertools
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
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.stats import norm

from qc_option_pricing.classical import arithmetic_asian_call_mc

RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N_DATES = 2
QUBITS_PER_DATE = 2
BASIS_GATES = ["ry", "rz", "x", "h", "cx"]


def normal_bin_grid() -> tuple[np.ndarray, np.ndarray]:
    """Four-bin standard-normal grid with exact bin probabilities."""
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    probs = np.diff(norm.cdf(edges))
    reps = []
    for lo, hi, p in zip(edges[:-1], edges[1:], probs):
        # Conditional mean of Z in [lo, hi]. This gives each bin a representative
        # Brownian shock while preserving exact bin probability.
        reps.append((norm.pdf(lo) - norm.pdf(hi)) / p)
    return np.array(reps), probs


def path_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shocks, shock_probs = normal_bin_grid()
    dt = T / N_DATES
    drift = (R - 0.5 * SIGMA**2) * dt
    vol = SIGMA * math.sqrt(dt)

    probs, payoffs, averages = [], [], []
    for shock_indices in itertools.product(range(len(shocks)), repeat=N_DATES):
        s = S0
        path_prices = []
        prob = 1.0
        for shock_index in shock_indices:
            z = shocks[shock_index]
            s *= math.exp(drift + vol * z)
            path_prices.append(s)
            prob *= shock_probs[shock_index]
        avg = float(np.mean(path_prices))
        probs.append(prob)
        payoffs.append(max(avg - K, 0.0))
        averages.append(avg)
    return np.array(probs), np.array(payoffs), np.array(averages)


def _select_basis_state(qc: QuantumCircuit, controls: list[int], index: int) -> list[int]:
    flipped = []
    for bit, qubit in enumerate(controls):
        if ((index >> bit) & 1) == 0:
            qc.x(qubit)
            flipped.append(qubit)
    return flipped


def build_toy_asian_circuit() -> tuple[QuantumCircuit, int, float, float, np.ndarray, np.ndarray]:
    probs, payoffs, averages = path_grid()
    f_max = float(payoffs.max())
    if f_max <= 0:
        raise ValueError("Toy grid produced no positive call payoffs.")

    n_state = N_DATES * QUBITS_PER_DATE
    objective = n_state
    qc = QuantumCircuit(n_state + 1, name="asian_toy")
    qc.append(StatePreparation(np.sqrt(probs)), range(n_state))

    controls = list(range(n_state))
    for index, payoff in enumerate(payoffs):
        if payoff <= 0:
            continue
        normalized = min(1.0, float(payoff / f_max))
        theta = 2.0 * math.asin(math.sqrt(normalized))
        flipped = _select_basis_state(qc, controls, index)
        qc.mcry(theta, controls, objective, None, mode="noancilla")
        for qubit in reversed(flipped):
            qc.x(qubit)

    exact_grid_price = math.exp(-R * T) * float(np.dot(probs, payoffs))
    return qc, objective, f_max, exact_grid_price, probs, averages


def main() -> None:
    global N_DATES

    parser = argparse.ArgumentParser(description="Generate a toy quantum Asian validation.")
    parser.add_argument(
        "--dates",
        type=int,
        default=N_DATES,
        help="Number of Asian monitoring dates. Keep small; grid size is 4^dates.",
    )
    args = parser.parse_args()
    if args.dates < 2:
        raise ValueError("--dates must be at least 2")
    N_DATES = args.dates

    qc, objective, f_max, exact_grid_price, probs, averages = build_toy_asian_circuit()
    state = Statevector.from_instruction(qc)
    objective_prob = float(state.probabilities([objective])[1])
    quantum_price = math.exp(-R * T) * f_max * objective_prob

    mc_price, mc_se = arithmetic_asian_call_mc(
        S0,
        K,
        R,
        SIGMA,
        T,
        N_DATES,
        300_000,
        rng=np.random.default_rng(20260415),
    )

    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=BASIS_GATES)
    transpiled = pm.run(qc)
    ops = dict(transpiled.count_ops())

    if N_DATES == 2:
        table_path = RESULTS_DIR / "quantum_asian_toy_table.txt"
        fig_path = RESULTS_DIR / "quantum_asian_toy_validation.png"
    else:
        table_path = RESULTS_DIR / f"quantum_asian_{N_DATES}date_table.txt"
        fig_path = RESULTS_DIR / f"quantum_asian_{N_DATES}date_validation.png"

    with open(table_path, "w") as f:
        f.write("Toy Arithmetic Asian Quantum Circuit Validation\n")
        f.write("=" * 58 + "\n")
        f.write(
            "Source pipeline: Stamatopoulos et al. (2020), Sec. 4.2.2, Eq. (32)-(35): "
            "the arithmetic Asian payoff is max(Sbar-K,0), with Sbar the average "
            f"over selected time points. This script uses the same idea on a {N_DATES}-date "
            "finite GBM grid so the quantum payoff can be checked exactly.\n"
        )
        f.write(
            f"Important scope: this is a toy {N_DATES}-date grid validation, not a realistic "
            "daily-monitored Asian circuit. Realistic Asian pricing is handled by the "
            "classical MC/Kemna-Vorst scripts.\n"
        )
        f.write(
            "Gate basis for resource counts: "
            + ", ".join(BASIS_GATES)
            + " (standard simulator gates; no opaque library gates counted).\n"
        )
        f.write(
            f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}, "
            f"monitoring_dates={N_DATES}, qubits_per_date={QUBITS_PER_DATE}, "
            f"grid_paths={len(probs)}.\n\n"
        )
        f.write(f"Exact finite-grid Asian price: {exact_grid_price:.6f}\n")
        f.write(f"Quantum statevector price:    {quantum_price:.6f}\n")
        f.write(f"Absolute circuit/grid error:  {abs(quantum_price - exact_grid_price):.6e}\n")
        f.write(f"Classical MC {N_DATES}-date price:  {mc_price:.6f} ± {mc_se:.6f}\n")
        f.write(f"Objective probability:        {objective_prob:.8f}\n")
        f.write(f"Payoff scale f_max:           {f_max:.6f}\n")
        f.write(f"Average-price grid range:     [{averages.min():.4f}, {averages.max():.4f}]\n\n")
        f.write("Transpiled resources\n")
        f.write("-" * 58 + "\n")
        f.write(f"Qubits: {transpiled.num_qubits}\n")
        f.write(f"Depth:  {transpiled.depth()}\n")
        for gate in BASIS_GATES:
            f.write(f"{gate:>3}: {ops.get(gate, 0)}\n")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["Exact grid", "Quantum circuit", "Classical MC"]
    values = [exact_grid_price, quantum_price, mc_price]
    ax0.bar(labels, values, color=["#2563eb", "#16a34a", "#dc2626"])
    ax0.errorbar([2], [mc_price], yerr=[mc_se], fmt="none", ecolor="#111827", capsize=4)
    ax0.set_ylabel("Discounted Asian call price")
    ax0.set_title(f"{N_DATES}-Date Asian Toy Validation")
    ax0.grid(True, axis="y", alpha=0.25)

    gate_labels = [gate.upper() for gate in BASIS_GATES]
    gate_counts = [ops.get(gate, 0) for gate in BASIS_GATES]
    ax1.bar(gate_labels, gate_counts, color="#64748b")
    ax1.set_ylabel("Transpiled gate count")
    ax1.set_title("Known-Gate Resource Count")
    ax1.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Exact finite-grid price: {exact_grid_price:.6f}")
    print(f"Quantum statevector price: {quantum_price:.6f}")
    print(f"Classical MC {N_DATES}-date price: {mc_price:.6f} ± {mc_se:.6f}")
    print(f"Wrote {table_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
