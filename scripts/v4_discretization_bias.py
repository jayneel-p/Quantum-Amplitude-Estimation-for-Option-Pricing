#!/usr/bin/env python3
"""How much does discretizing the increment onto a finite grid bias the price,
for the raw arithmetic payoff versus the residual?

The finite-grid oracle cannot load a continuous Gaussian increment.  With b
qubits per date it loads 2^b values, each value being the mean of the Gaussian over its
bin (the conditional mean), with the bins chosen equally likely.  Replacing the
continuous increment by this finite set moves the price away from the continuous
reference used by the simulation.  When the continuous closed form C_G is added
to a grid estimate of E[R], this difference remains (Sec. 6.4).

We measure that bias for two quantities on the same grids, using the same
Gaussian draws for the continuous and the discretized run so the difference is
paired and low variance:
  raw payoff       E[max(A - K, 0)]
  residual         E[R] = E[max(A - K, 0) - max(G - K, 0)]
The bias is (price on the grid) - (price with the continuous increment), in
discounted dollars.

The output includes paired Monte Carlo standard errors for every difference.

Outputs: results/v4/discretization_bias.txt.  Exits nonzero if any gate fails.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

S0, K, RATE, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N = 252
DISC = math.exp(-RATE * T)
N_PATHS = 2_000_000
CHUNK = 250_000
BITS = [1, 2, 3]
SEED = 20260705

GATES: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    GATES.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def bin_edges_and_means(b: int) -> tuple[np.ndarray, np.ndarray]:
    """2^b equally likely bins of the standard normal, each represented by the
    mean of the Gaussian over the bin (its conditional mean)."""
    nb = 2 ** b
    edges = norm.ppf(np.linspace(0.0, 1.0, nb + 1))
    means = np.empty(nb)
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        means[i] = (norm.pdf(lo) - norm.pdf(hi)) / (1.0 / nb)   # E[Z | bin]
    return edges, means


def snap(z: np.ndarray, edges: np.ndarray, means: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(means) - 1)
    return means[idx]


def main() -> int:
    rng = np.random.default_rng(SEED)
    dt = T / N
    drift = (RATE - 0.5 * SIGMA ** 2) * dt
    vol = SIGMA * math.sqrt(dt)

    grids = {b: bin_edges_and_means(b) for b in BITS}
    cont = {"a": 0.0, "r": 0.0}
    disc = {b: {"a": 0.0, "r": 0.0} for b in BITS}
    diff = {b: {"a": [0.0, 0.0], "r": [0.0, 0.0]} for b in BITS}
    n = 0
    done = 0
    while done < N_PATHS:
        m = min(CHUNK, N_PATHS - done)
        z = rng.standard_normal((m, N))
        logs_c = math.log(S0) + np.cumsum(drift + vol * z, axis=1)
        a_c = np.exp(logs_c).mean(axis=1)
        g_c = np.exp(logs_c.mean(axis=1))
        pay_a_c = np.maximum(a_c - K, 0.0)
        pay_r_c = pay_a_c - np.maximum(g_c - K, 0.0)
        cont["a"] += pay_a_c.sum()
        cont["r"] += pay_r_c.sum()
        for b in BITS:
            edges, means = grids[b]
            logs_d = math.log(S0) + np.cumsum(drift + vol * snap(z, edges, means), axis=1)
            a_d = np.exp(logs_d).mean(axis=1)
            g_d = np.exp(logs_d.mean(axis=1))
            pay_a_d = np.maximum(a_d - K, 0.0)
            pay_r_d = pay_a_d - np.maximum(g_d - K, 0.0)
            disc[b]["a"] += pay_a_d.sum()
            disc[b]["r"] += pay_r_d.sum()
            da = DISC * (pay_a_d - pay_a_c)
            dr = DISC * (pay_r_d - pay_r_c)
            diff[b]["a"][0] += da.sum()
            diff[b]["a"][1] += np.square(da).sum()
            diff[b]["r"][0] += dr.sum()
            diff[b]["r"][1] += np.square(dr).sum()
        n += m
        done += m

    cont_a = DISC * cont["a"] / n
    cont_r = DISC * cont["r"] / n
    rows = []
    for b in BITS:
        bias_a = diff[b]["a"][0] / n
        bias_r = diff[b]["r"][0] / n
        var_a = (diff[b]["a"][1] - diff[b]["a"][0] ** 2 / n) / (n - 1)
        var_r = (diff[b]["r"][1] - diff[b]["r"][0] ** 2 / n) / (n - 1)
        se_a = math.sqrt(max(var_a, 0.0) / n)
        se_r = math.sqrt(max(var_r, 0.0) / n)
        rows.append((b, bias_a, se_a, bias_r, se_r, abs(bias_a / bias_r)))

    gate("bias negative at three-SE margin",
         all(r[1] + 3 * r[2] < 0 and r[3] + 3 * r[4] < 0 for r in rows),
         "raw and residual paired estimates are below zero by more than three standard errors")
    gate("absolute bias decreases at three-SE margin",
         all(abs(rows[i - 1][1]) - abs(rows[i][1]) > 3 * (rows[i - 1][2] + rows[i][2])
             and abs(rows[i - 1][3]) - abs(rows[i][3]) > 3 * (rows[i - 1][4] + rows[i][4])
             for i in range(1, len(rows))),
         "each b-to-b+1 change exceeds three times the sum of its paired standard errors")
    gate("residual absolute bias smaller at three-SE margin",
         all(abs(r[1]) - abs(r[3]) > 3 * (r[2] + r[4]) for r in rows),
         f"point-estimate ratio |raw/residual| in [{min(r[5] for r in rows):.1f}, {max(r[5] for r in rows):.1f}]")

    lines = [
        "Bias from discretizing the increment onto a finite grid, raw payoff vs residual",
        f"daily Asian call, N={N}, {N_PATHS} paths, equally likely bins, conditional-mean values",
        f"seed {SEED}",
        f"continuous reference: E[disc max(A-K,0)] = {cont_a:.5f}, E[disc R] = {cont_r:.5f}",
        "",
        "Bias and SE use paired per-path differences between the grid and continuous draws.",
        f"{'qubits/date':>11}  {'raw bias ($)':>14} {'raw SE':>10}  "
        f"{'residual bias ($)':>18} {'resid SE':>10}  {'|raw|/|residual|':>16}",
    ]
    for b, ba, sea, br, ser, ratio in rows:
        lines.append(f"{b:>11}  {ba:>+14.5f} {sea:>10.5f}  {br:>+18.5f} {ser:>10.5f}  {ratio:>16.1f}")
    lines += ["", "gates:"]
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in GATES]
    (OUT / "discretization_bias.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> wrote {OUT}/discretization_bias.txt")

    failed = [name for name, ok, _ in GATES if not ok]
    if failed:
        print(f"FAILED GATES: {failed}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
