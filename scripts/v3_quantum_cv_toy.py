#!/usr/bin/env python3
"""v3: quantum control variate (QCV) on the toy Asian grids.

Idea: C_A = C_G + e^{-rT} E[R] with residual R = max(A-K,0) - max(G-K,0).
By AM-GM, A >= G pathwise and max(.-K,0) is nondecreasing, so R >= 0 on every
path -- the residual is itself a valid nonnegative payoff and drops directly
into the standard amplitude encoding. QAE query cost scales with the payoff
bound f_max, so the win is f_max(A) / f_max(R).

For each grid (2-date, 3-date; two-qubit shocks per date, same construction as
generate_quantum_asian_toy.py):
  * verify R >= 0 on every grid path (the AM-GM lemma, numerically),
  * build raw and residual circuits, check both statevector prices against the
    exact grid expectations to numerical precision,
  * report f_max(A), f_max(R), amplitudes, and the oracle-query ratio at fixed
    target error (= f_max ratio, IQAE constant cancels),
  * report transpiled resources for both circuits in the same basis.

Outputs: results/v3/quantum_cv_table.txt, results/v3/quantum_cv_validation.png
"""
from __future__ import annotations

import itertools
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
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.stats import norm

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
QUBITS_PER_DATE = 2
BASIS_GATES = ["ry", "rz", "x", "h", "cx"]

OUT = _REPO / "results" / "v3"
OUT.mkdir(parents=True, exist_ok=True)


def normal_bin_grid() -> tuple[np.ndarray, np.ndarray]:
    """Four-bin standard-normal grid: conditional-mean representatives, exact bin probs."""
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    probs = np.diff(norm.cdf(edges))
    reps = [(norm.pdf(lo) - norm.pdf(hi)) / p
            for lo, hi, p in zip(edges[:-1], edges[1:], probs)]
    return np.array(reps), probs


def path_grid(n_dates: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs, arith payoffs, geo payoffs) over the 4^n_dates grid paths."""
    shocks, shock_probs = normal_bin_grid()
    dt = T / n_dates
    drift = (R - 0.5 * SIGMA**2) * dt
    vol = SIGMA * math.sqrt(dt)

    probs, pay_a, pay_g = [], [], []
    for idx in itertools.product(range(len(shocks)), repeat=n_dates):
        s = S0
        prices = []
        p = 1.0
        for i in idx:
            s *= math.exp(drift + vol * shocks[i])
            prices.append(s)
            p *= shock_probs[i]
        prices = np.array(prices)
        a = float(prices.mean())
        g = float(np.exp(np.log(prices).mean()))
        probs.append(p)
        pay_a.append(max(a - K, 0.0))
        pay_g.append(max(g - K, 0.0))
    return np.array(probs), np.array(pay_a), np.array(pay_g)


def build_payoff_circuit(probs: np.ndarray, payoffs: np.ndarray,
                         n_dates: int) -> tuple[QuantumCircuit, int, float]:
    """Exact finite-grid payoff encoding (one mcry per positive-payoff path)."""
    f_max = float(payoffs.max())
    n_state = n_dates * QUBITS_PER_DATE
    objective = n_state
    qc = QuantumCircuit(n_state + 1)
    qc.append(StatePreparation(np.sqrt(probs)), range(n_state))
    controls = list(range(n_state))
    for index, payoff in enumerate(payoffs):
        if payoff <= 0:
            continue
        theta = 2.0 * math.asin(math.sqrt(min(1.0, payoff / f_max)))
        flipped = []
        for bit, qubit in enumerate(controls):
            if ((index >> bit) & 1) == 0:
                qc.x(qubit)
                flipped.append(qubit)
        qc.mcry(theta, controls, objective, None, mode="noancilla")
        for qubit in reversed(flipped):
            qc.x(qubit)
    return qc, objective, f_max


def statevector_price(qc: QuantumCircuit, objective: int, f_max: float) -> tuple[float, float]:
    amp = float(Statevector.from_instruction(qc).probabilities([objective])[1])
    return math.exp(-R * T) * f_max * amp, amp


def resources(qc: QuantumCircuit) -> tuple[int, int, dict]:
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=BASIS_GATES)
    tq = pm.run(qc)
    return tq.num_qubits, tq.depth(), dict(tq.count_ops())


def main() -> int:
    disc = math.exp(-R * T)
    lines = [
        "Quantum control variate (QCV) toy validation",
        "=" * 60,
        "Residual R = max(A-K,0) - max(G-K,0); by AM-GM, R >= 0 pathwise, so it",
        "is encoded directly as a nonnegative payoff. C_A = C_G(grid-exact) +",
        "e^{-rT} E[R]. QAE queries scale with f_max, so the query ratio at fixed",
        "target error is f_max(A)/f_max(R) (the IQAE constant cancels).",
        f"Basis for resource counts: {', '.join(BASIS_GATES)}.",
        f"Parameters: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}, "
        f"qubits_per_date={QUBITS_PER_DATE}.",
        "",
    ]
    fig_rows = []
    for n_dates in (2, 3):
        probs, pay_a, pay_g = path_grid(n_dates)
        res = pay_a - pay_g
        assert res.min() >= -1e-12, "AM-GM violated?!"
        res = np.maximum(res, 0.0)

        exact_a = disc * float(probs @ pay_a)
        exact_g = disc * float(probs @ pay_g)
        exact_r = disc * float(probs @ res)

        qc_a, obj_a, fmax_a = build_payoff_circuit(probs, pay_a, n_dates)
        qc_r, obj_r, fmax_r = build_payoff_circuit(probs, res, n_dates)
        price_a, amp_a = statevector_price(qc_a, obj_a, fmax_a)
        price_r, amp_r = statevector_price(qc_r, obj_r, fmax_r)
        price_qcv = exact_g + price_r

        nq_a, d_a, ops_a = resources(qc_a)
        nq_r, d_r, ops_r = resources(qc_r)

        lines += [
            f"--- {n_dates}-date grid ({len(probs)} paths) ---",
            f"min grid residual:              {float((pay_a - pay_g).min()):.3e}  (>= 0: AM-GM)",
            f"exact grid arithmetic price:    {exact_a:.6f}",
            f"exact grid geometric price:     {exact_g:.6f}",
            f"exact grid residual price:      {exact_r:.6f}",
            f"raw quantum price:              {price_a:.6f}   (|err| {abs(price_a - exact_a):.2e})",
            f"QCV quantum price (C_G + resid):{price_qcv:.6f}   (|err| {abs(price_qcv - exact_a):.2e})",
            f"f_max(A) = {fmax_a:.4f},  f_max(R) = {fmax_r:.4f},  "
            f"query ratio f_max(A)/f_max(R) = {fmax_a / fmax_r:.1f}x",
            f"objective amplitude: raw {amp_a:.6f}, residual {amp_r:.6f}",
            f"resources raw:  qubits {nq_a}, depth {d_a}, cx {ops_a.get('cx', 0)}",
            f"resources QCV:  qubits {nq_r}, depth {d_r}, cx {ops_r.get('cx', 0)}",
            "",
        ]
        fig_rows.append((n_dates, fmax_a, fmax_r, exact_a, price_qcv, exact_g))

    (OUT / "quantum_cv_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = np.arange(len(fig_rows))
    w = 0.35
    ax0.bar(x - w / 2, [r[1] for r in fig_rows], w, color="#c0392b", label="$f_{\\max}$ raw payoff")
    ax0.bar(x + w / 2, [r[2] for r in fig_rows], w, color="#16a34a", label="$f_{\\max}$ residual")
    for i, r_ in enumerate(fig_rows):
        ax0.text(i, r_[1] + 0.6, f"{r_[1]/r_[2]:.0f}$\\times$", ha="center", fontsize=10)
    ax0.set_xticks(x, [f"{r[0]} dates" for r in fig_rows])
    ax0.set_ylabel("payoff bound $f_{\\max}$ (\\$)")
    ax0.set_title("(a) Control variate shrinks the payoff bound")
    ax0.legend(fontsize=9); ax0.grid(True, axis="y", alpha=0.25)

    labels = [f"{r[0]}d" for r in fig_rows]
    ax1.bar(x - w / 2, [r[3] for r in fig_rows], w, color="#2563eb", label="exact grid price")
    ax1.bar(x + w / 2, [r[4] for r in fig_rows], w, color="#7c3aed",
            label="QCV: $C_G$ + quantum residual")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("discounted Asian call price")
    ax1.set_title("(b) QCV validation (bars agree to $10^{-13}$)")
    ax1.legend(fontsize=9); ax1.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "quantum_cv_validation.png", dpi=200, bbox_inches="tight")
    print(f"-> wrote {OUT}/quantum_cv_validation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
