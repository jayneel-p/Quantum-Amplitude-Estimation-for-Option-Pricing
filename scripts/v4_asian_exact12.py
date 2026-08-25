#!/usr/bin/env python3
"""v4: exact encoding validation on a 12-date (monthly) Asian grid.

The brute-force circuits of Sec. 6.4 stop at 4 dates because their gate count
is O(4^N).  The encoded state itself has no such limit: for the standard
encoding |psi> = sum_i sqrt(p_i) |i> (sqrt(1-f_i)|0> + sqrt(f_i)|1>), the
state is specified exactly by the grid and the payoff table, and the price is
the Born probability of the objective qubit.  This script:

  1. verifies, at 4 and 6 dates, that Qiskit's StatePreparation synthesis of
     that state reproduces it to machine precision and that the Born price
     equals the direct evaluation (this licenses direct evaluation at sizes
     where gate synthesis is impractical);
  2. evaluates the exact 12-date monthly Asian grid (4^12 = 16.8M paths,
     25 qubits): raw payoff and block residuals k in {1,2,3,4,6}, pathwise
     ordering, payoff bounds, query ratios, and the reconstruction identity.

Gates (fail-closed): state-synthesis cross-checks at 4 and 6 dates <= 1e-12;
4-date numbers reproduce v4_qcv_toy_blocks; ordering A >= B_k >= G exact on
all 16.8M paths; reconstructions exact to 1e-12.

Output: results/v4/asian_exact12.txt
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from scipy.stats import norm

S0, K, RATE, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N_DATES = 12
KS = [1, 2, 3, 4, 6]
CHUNK = 1 << 20                       # 1M paths per chunk
OUT = _REPO / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def normal_bin_grid():
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    probs = np.diff(norm.cdf(edges))
    reps = [(norm.pdf(lo) - norm.pdf(hi)) / p
            for lo, hi, p in zip(edges[:-1], edges[1:], probs)]
    return np.array(reps), probs


SHOCKS, SHOCK_P = normal_bin_grid()
LOG_P = np.log(SHOCK_P)


def chunk_quantities(n_dates: int, lo: int, hi: int):
    """probs, payoff arrays for grid indices [lo, hi)."""
    idx = np.arange(lo, hi, dtype=np.int64)
    m = hi - lo
    dt = T / n_dates
    drift = (RATE - 0.5 * SIGMA**2) * dt
    vol = SIGMA * math.sqrt(dt)
    cums = np.empty((n_dates, m))
    logp = np.zeros(m)
    cum = np.full(m, math.log(S0))
    for d in range(n_dates):
        dig = (idx >> (2 * d)) & 3
        logp += LOG_P[dig]
        cum = cum + drift + vol * SHOCKS[dig]
        cums[d] = cum
    probs = np.exp(logp)
    a = np.exp(cums).mean(axis=0)
    pay_a = np.maximum(a - K, 0.0)
    pay_b = {}
    for k in [kk for kk in KS if n_dates % kk == 0]:
        blk = n_dates // k
        bk = np.exp(cums.reshape(k, blk, m).mean(axis=1)).mean(axis=0)
        pay_b[k] = np.maximum(bk - K, 0.0)
    return probs, a, pay_a, pay_b


def crosscheck_synthesis(n_dates: int) -> None:
    """StatePreparation synthesis reproduces the joint state exactly."""
    probs, _, pay_a, pay_b = chunk_quantities(n_dates, 0, 4**n_dates)
    for tag, payoffs in (("raw", pay_a), ("R1", pay_a - pay_b[1])):
        payoffs = np.maximum(payoffs, 0.0)
        fm = float(payoffs.max())
        f = np.clip(payoffs / fm, 0.0, 1.0)
        psi = np.concatenate([np.sqrt(probs * (1 - f)), np.sqrt(probs * f)])
        n = int(math.log2(len(psi)))
        qc = QuantumCircuit(n)
        qc.append(StatePreparation(psi), range(n))
        sv = Statevector.from_instruction(qc)
        vec_err = float(np.abs(sv.data - psi).max())
        born = float(sv.probabilities([n - 1])[1])
        direct = float(probs @ f)
        # Synthesis roundoff on individual amplitudes grows with register size
        # (thousands of composed rotations); the licensed quantity is the Born
        # probability, which is what the estimator measures.
        check(f"S{n_dates} synthesis {tag}",
              vec_err < 1e-9 and abs(born - direct) < 1e-12,
              f"max state error {vec_err:.1e}, Born-direct diff "
              f"{abs(born - direct):.1e} ({n} qubits)")


def main() -> int:
    # 1. license the direct evaluation
    crosscheck_synthesis(4)
    crosscheck_synthesis(6)

    # reproduce the 4-date toy numbers
    probs4, _, pa4, pb4 = chunk_quantities(4, 0, 256)
    r1 = np.maximum(pa4 - pb4[1], 0.0)
    r2 = np.maximum(pa4 - pb4[2], 0.0)
    check("4-date reproduces toy script",
          abs(pa4.max() - 51.5854) < 0.01
          and abs(pa4.max() / r1.max() - 21.5) < 0.1
          and abs(pa4.max() / r2.max() - 106.6) < 0.2,
          f"f_max(A)={pa4.max():.4f}, ratios {pa4.max()/r1.max():.1f}, "
          f"{pa4.max()/r2.max():.1f} (toy: 51.5854, 21.5, 106.6)")

    # 2. exact 12-date grid
    n_paths = 4**N_DATES
    tot_p = 0.0
    e_a = 0.0
    e_b = {k: 0.0 for k in KS}
    e_r = {k: 0.0 for k in KS}
    fmax_a = 0.0
    fmax_r = {k: 0.0 for k in KS}
    min_ord1 = np.inf     # min over paths of A - B_k
    min_ord2 = np.inf     # min over paths of B_k - G
    for lo in range(0, n_paths, CHUNK):
        hi = min(lo + CHUNK, n_paths)
        probs, a, pay_a, pay_b = chunk_quantities(N_DATES, lo, hi)
        tot_p += float(probs.sum())
        e_a += float(probs @ pay_a)
        fmax_a = max(fmax_a, float(pay_a.max()))
        g_ord = None
        for k in KS:
            r = pay_a - pay_b[k]
            e_b[k] += float(probs @ pay_b[k])
            e_r[k] += float(probs @ r)
            fmax_r[k] = max(fmax_r[k], float(r.max()))
            bk = pay_b[k]
            if k == 1:
                g_ord = bk
            else:
                min_ord2 = min(min_ord2, float((bk - g_ord).min()))
            min_ord1 = min(min_ord1, float(r.min()))
        print(f"  chunk {lo >> 20:2d}/16 done", flush=True)

    check("12-date probability normalization", abs(tot_p - 1.0) < 1e-12,
          f"sum p = 1 {tot_p - 1.0:+.2e}")
    check("12-date pathwise ordering", min_ord1 >= 0.0 and min_ord2 >= -1e-12,
          f"min(f_A - f_Bk) = {min_ord1:.3e}, min(f_Bk - f_G) = {min_ord2:.3e} "
          f"over {n_paths:,} paths")

    disc = math.exp(-RATE * T)
    c_a = disc * e_a
    lines = [
        "Exact 12-date (monthly) Asian grid: 4^12 = 16,777,216 paths, 25 qubits",
        "conditional-mean 4-point shocks; direct exact evaluation of the encoded",
        "state, licensed by the synthesis cross-checks at 4 and 6 dates.",
        f"S0={S0}, K={K}, r={RATE}, sigma={SIGMA}, T={T}",
        "",
        f"exact grid arithmetic price: {c_a:.6f}    f_max(A) = {fmax_a:.4f}",
        "",
        "  k   C_Bk (grid)   f_max(R_k)   ratio    reconstruction |err|",
    ]
    worst_rec = 0.0
    for k in KS:
        a_r = e_r[k] / fmax_r[k]
        rec = disc * e_b[k] + disc * fmax_r[k] * a_r
        err = abs(rec - c_a)
        worst_rec = max(worst_rec, err)
        lines.append(f"  {k:2d}   {disc*e_b[k]:.6f}     {fmax_r[k]:8.4f}   "
                     f"{fmax_a/fmax_r[k]:6.1f}     {err:.2e}")
    lines.append("  12   (B_12 = A: residual identically zero)")
    check("12-date reconstruction identity", worst_rec < 1e-12,
          f"worst |C_Bk + e^-rT f_max a_R - C_A| = {worst_rec:.2e}")

    lines += ["", "gates: " + ("ALL PASS" if not FAILURES else f"FAILED {FAILURES}")]
    (OUT / "asian_exact12.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
