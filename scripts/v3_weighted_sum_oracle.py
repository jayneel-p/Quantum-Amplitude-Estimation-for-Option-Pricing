#!/usr/bin/env python3
"""v3: scalable weighted-sum oracle for the geometric-Asian leg
(Stamatopoulos et al. Sec. 4.2.2 / App. A architecture).

Construction, per monitoring date j = 1..N:
  * a small shock register of m qubits holding q_j in {0..2^m-1}, loaded
    independently (increments are independent, so loading is LINEAR in N);
    grid z(q) equally spaced on [-w_z, w_z] sigma, probabilities proportional
    to the standard-normal pdf at the grid points (the Sta20/Qiskit
    convention used throughout the paper).
  * the log of the geometric average is affine in the integer weighted sum
        s = sum_j w_j q_j,   w_j = N - j + 1,
    because  (1/N) sum_i log S_{t_i} = log S0 + drift (N+1)/2
             + (vol/N) * [ 2*dz*s - w_z * W ]  with W = sum_j w_j
    (equally spaced z-grid makes z affine in q; dz = grid half-spacing).
  * a WeightedAdder (Sta20 App. A weighted-sum operator; Draper-style
    reversible addition) computes |s> into a sum register of
    ceil(log2(1+3W)) qubits -- O(log N^2) qubits, O(N^2) additions.
  * the payoff rotation is applied per sum value s (exact table,
    O(N^2) rotations, still polynomial; a comparator + piecewise-linear
    rotation would reduce this to O(bits)).

The payoff is therefore an EXACT function of the sum register: the oracle
prices the geometric-Asian leg with polynomial resources, in contrast to the
exact-grid construction (one multi-controlled rotation per path, O(4^N)).
This is precisely the cheap leg that the quantum control variate adds to an
arithmetic-average oracle (Sec. 5.2 of the manuscript); the arithmetic leg
additionally needs fixed-point exponentiation (Chakrabarti et al.).

Validation: statevector price == exact grid price computed independently by
convolution of the digit distributions (no path enumeration). Resources:
transpiled to {ry, rz, x, h, cx}, optimization level 1, compared against the
brute-force exact-grid construction on the same grid.

Outputs: results/v3/weighted_sum_table.txt, results/v3/weighted_sum_scaling.png
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, WeightedAdder
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.stats import norm

from qc_option_pricing.classical.asian_mc import geometric_asian_call_exact

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
BASIS_GATES = ["ry", "rz", "x", "h", "cx"]
W_Z = 3.0                      # z-grid half-width (sigmas)
OUT = _REPO / "results" / "v3"
OUT.mkdir(parents=True, exist_ok=True)


def shock_grid(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced z-grid on [-W_Z, W_Z], pdf-proportional probabilities."""
    z = np.linspace(-W_Z, W_Z, 2**m)
    p = norm.pdf(z)
    return z, p / p.sum()


def geo_payoff_from_sum(s: int, n_dates: int, m: int) -> float:
    """Geometric-Asian payoff as an exact function of the weighted sum s."""
    z, _ = shock_grid(m)
    dz = z[1] - z[0]
    dt = T / n_dates
    drift = (R - 0.5 * SIGMA**2) * dt
    vol = SIGMA * math.sqrt(dt)
    w_total = n_dates * (n_dates + 1) // 2
    log_g = (math.log(S0) + drift * (n_dates + 1) / 2.0
             + (vol / n_dates) * (dz * s - W_Z * w_total))
    return max(math.exp(log_g) - K, 0.0)


def sum_distribution(n_dates: int, m: int) -> np.ndarray:
    """P(s) by convolving the per-date weighted digit distributions."""
    _, probs = shock_grid(m)
    w_total = n_dates * (n_dates + 1) // 2
    s_max = (2**m - 1) * w_total
    dist = np.zeros(s_max + 1)
    dist[0] = 1.0
    for j in range(1, n_dates + 1):
        w = n_dates - j + 1
        digit = np.zeros(w * (2**m - 1) + 1)
        for q, p in enumerate(probs):
            digit[w * q] = p
        dist = np.convolve(dist, digit)
    return dist


def build_weighted_sum_oracle(n_dates: int, m: int) -> tuple[QuantumCircuit, int, float]:
    """Shock loading + WeightedAdder + exact payoff rotation on the sum register."""
    _, probs = shock_grid(m)
    n_state = n_dates * m
    weights = []
    for j in range(1, n_dates + 1):
        w = n_dates - j + 1
        weights += [w * (2**k) for k in range(m)]

    adder = WeightedAdder(num_state_qubits=n_state, weights=weights)
    n_sum = adder.num_sum_qubits
    n_total = adder.num_qubits + 1                 # + objective
    objective = n_total - 1
    sum_qubits = list(range(n_state, n_state + n_sum))

    qc = QuantumCircuit(n_total, name=f"wsum_{n_dates}d")
    prep = StatePreparation(np.sqrt(probs))
    for j in range(n_dates):
        qc.append(prep, range(j * m, (j + 1) * m))
    qc.append(adder, range(adder.num_qubits))

    s_max = (2**m - 1) * n_dates * (n_dates + 1) // 2
    payoffs = np.array([geo_payoff_from_sum(s, n_dates, m) for s in range(s_max + 1)])
    f_max = float(payoffs.max())
    for s, pay in enumerate(payoffs):
        if pay <= 0:
            continue
        theta = 2.0 * math.asin(math.sqrt(min(1.0, pay / f_max)))
        flipped = []
        for bit, qubit in enumerate(sum_qubits):
            if ((s >> bit) & 1) == 0:
                qc.x(qubit)
                flipped.append(qubit)
        qc.mcry(theta, sum_qubits, objective, None, mode="noancilla")
        for qubit in reversed(flipped):
            qc.x(qubit)
    return qc, objective, f_max


def build_brute_force_oracle(n_dates: int, m: int) -> QuantumCircuit:
    """Exact-grid construction on the SAME grid: one mcry per positive-payoff
    path over all 2^{mN} paths (for the scaling comparison)."""
    import itertools
    z, probs = shock_grid(m)
    dt = T / n_dates
    drift = (R - 0.5 * SIGMA**2) * dt
    vol = SIGMA * math.sqrt(dt)

    path_probs, path_pay = [], []
    for idx in itertools.product(range(2**m), repeat=n_dates):
        log_s, p = math.log(S0), 1.0
        logs = []
        for q in idx:
            log_s += drift + vol * z[q]
            logs.append(log_s)
            p *= probs[q]
        g = math.exp(sum(logs) / n_dates)
        path_probs.append(p)
        path_pay.append(max(g - K, 0.0))
    path_probs = np.array(path_probs)
    path_pay = np.array(path_pay)
    f_max = float(path_pay.max())

    n_state = n_dates * m
    qc = QuantumCircuit(n_state + 1, name=f"brute_{n_dates}d")
    qc.append(StatePreparation(np.sqrt(path_probs)), range(n_state))
    controls = list(range(n_state))
    for index, pay in enumerate(path_pay):
        if pay <= 0:
            continue
        theta = 2.0 * math.asin(math.sqrt(min(1.0, pay / f_max)))
        flipped = []
        for bit, qubit in enumerate(controls):
            if ((index >> bit) & 1) == 0:
                qc.x(qubit)
                flipped.append(qubit)
        qc.mcry(theta, controls, n_state, None, mode="noancilla")
        for qubit in reversed(flipped):
            qc.x(qubit)
    return qc


def resources(qc: QuantumCircuit) -> tuple[int, int, int]:
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=BASIS_GATES)
    tq = pm.run(qc)
    return tq.num_qubits, tq.depth(), dict(tq.count_ops()).get("cx", 0)


def main() -> int:
    m = 2
    disc = math.exp(-R * T)
    lines = [
        "Weighted-sum (adder-based) geometric-Asian oracle vs exact-grid construction",
        f"(Sta20 Sec 4.2.2 / App A architecture; {2**m}-point equally spaced z-grid on "
        f"[-{W_Z:.0f},{W_Z:.0f}] sigma, pdf-proportional probs; basis {', '.join(BASIS_GATES)}, O1)",
        f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}",
        "",
        f"{'N':>3} {'qubits':>7} {'depth':>7} {'cx':>7} {'exact grid':>12} {'quantum':>12} "
        f"{'|err|':>10} {'C_G closed':>11} | {'BF qubits':>9} {'BF depth':>9} {'BF cx':>7}",
        "-" * 112,
    ]
    ws_rows, bf_rows = [], []
    for n_dates in (2, 3, 4, 5):
        dist = sum_distribution(n_dates, m)
        pays = np.array([geo_payoff_from_sum(s, n_dates, m) for s in range(len(dist))])
        exact = disc * float(dist @ pays)

        qc, obj, f_max = build_weighted_sum_oracle(n_dates, m)
        amp = float(Statevector.from_instruction(qc).probabilities([obj])[1])
        price = disc * f_max * amp
        nq, depth, cx = resources(qc)
        ws_rows.append((n_dates, nq, depth, cx))

        cg = geometric_asian_call_exact(S0, K, R, SIGMA, T, n_dates)

        if n_dates <= 4:
            bq, bd, bcx = resources(build_brute_force_oracle(n_dates, m))
            bf_rows.append((n_dates, bq, bd, bcx))
            bf_str = f"{bq:>9d} {bd:>9,d} {bcx:>7,d}"
        else:
            bf_str = f"{'-':>9} {'-':>9} {'-':>7}"

        lines.append(
            f"{n_dates:>3d} {nq:>7d} {depth:>7,d} {cx:>7,d} {exact:>12.6f} {price:>12.6f} "
            f"{abs(price - exact):>10.2e} {cg:>11.6f} | {bf_str}"
        )
        print(lines[-1], flush=True)

    lines += [
        "",
        "Loading is per-date (linear in N); the weighted sum register has "
        "O(log N^2) qubits; the payoff table on the sum register has O(N^2) "
        "entries (polynomial), vs O(4^N) path rotations for the exact-grid "
        "construction. 'C_G closed' is the continuous closed form; the gap to "
        "'exact grid' is the 4-point-shock discretization, identical for both "
        "circuit constructions.",
    ]
    (OUT / "weighted_sum_table.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.semilogy([r_[0] for r_ in ws_rows], [r_[2] for r_ in ws_rows], "o-",
                color="#16a34a", label="weighted-sum oracle (adder-based)")
    ax.semilogy([r_[0] for r_ in bf_rows], [r_[2] for r_ in bf_rows], "s--",
                color="#c0392b", label="exact-grid construction ($O(4^N)$)")
    ax.set_xlabel("monitoring dates $N$")
    ax.set_ylabel("transpiled circuit depth")
    ax.set_title("Adder-based vs exact-grid Asian oracle (same shock grid)")
    ax.set_xticks([r_[0] for r_ in ws_rows])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "weighted_sum_scaling.png", dpi=200, bbox_inches="tight")
    print(f"-> wrote {OUT}/weighted_sum_table.txt, weighted_sum_scaling.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
