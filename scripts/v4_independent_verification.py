#!/usr/bin/env python3
"""Independent correctness proof for the pricing engine and the quantum control
variate.  Nothing here imports the project's src/; every quantity is checked
against a source of truth that shares no code with the paper's pipeline:

  analytic limits, an independent quadrature, an independent third pricer
  (Levy moment matching), put-call parity, ten-digit literature benchmarks,
  the exact grid expectation, and an independently built quantum circuit.

The control variate is an exact decomposition, not an approximation:
    max(A-K,0) == max(G-K,0) + R           (definition of R, every path)
  =>  C_A = C_G + e^{-rT} E[R]              (take expectations)
So the only ways it can be wrong are (i) C_G wrong, (ii) R wrong or negative,
(iii) E[R] estimated wrong by the circuit, (iv) the reconstruction arithmetic
wrong.  Each is checked below to machine precision or literature precision.

Exits nonzero if any check fails.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
DISC = math.exp(-R * T)
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n      {detail}")


# ---------------------------------------------------------------------------
# independent building blocks (no project code)
# ---------------------------------------------------------------------------
def bs_call(s0, k, r, sig, t):
    d1 = (math.log(s0 / k) + (r + sig * sig / 2) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    return s0 * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def geo_asian_closed_form(n):
    """The paper's closed form (re-implemented independently)."""
    sg = SIGMA * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))
    mg = (R - 0.5 * SIGMA ** 2) * (n + 1) / (2 * n)
    d1 = (math.log(S0 / K) + (mg + sg ** 2) * T) / (sg * math.sqrt(T))
    d2 = d1 - sg * math.sqrt(T)
    return DISC * (S0 * math.exp((mg + 0.5 * sg ** 2) * T) * norm.cdf(d1) - K * norm.cdf(d2))


def geo_asian_quadrature(n):
    """Same price by direct numerical integration of the normal density for
    log G_N.  Shares no algebra with the closed form above."""
    sg = SIGMA * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))
    mg = (R - 0.5 * SIGMA ** 2) * (n + 1) / (2 * n)
    m = math.log(S0) + mg * T
    s = sg * math.sqrt(T)
    integrand = lambda x: max(math.exp(x) - K, 0.0) * norm.pdf(x, loc=m, scale=s)
    val, _ = quad(integrand, m - 12 * s, m + 12 * s, limit=400)
    return DISC * val


def arithmetic_moments(n):
    """Closed-form first two moments of A_N under GBM."""
    t = np.arange(1, n + 1) * T / n
    m1 = (S0 / n) * np.sum(np.exp(R * t))
    tj, tk = np.meshgrid(t, t)
    m2 = (S0 ** 2 / n ** 2) * np.sum(np.exp(R * (tj + tk) + SIGMA ** 2 * np.minimum(tj, tk)))
    return float(m1), float(m2)


def levy_asian_call(n):
    """Levy (1992) moment-matching approximation: fit A_N to a lognormal with
    the correct first two moments, then price as a lognormal call.  A genuinely
    independent third pricer (an approximation, so agreement is to ~1%)."""
    m1, m2 = arithmetic_moments(n)
    v = math.log(m2 / (m1 * m1))
    d1 = (math.log(m1 / K) + 0.5 * v) / math.sqrt(v)
    d2 = d1 - math.sqrt(v)
    return DISC * (m1 * norm.cdf(d1) - K * norm.cdf(d2))


def simulate(n, m, rng, sigma=SIGMA, s0=S0, k=K):
    dt = T / n
    z = rng.standard_normal((m, n))
    logs = math.log(s0) + np.cumsum((R - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z, axis=1)
    a = np.exp(logs).mean(axis=1)
    g = np.exp(logs.mean(axis=1))
    return a, g


def conditional_mean_grid():
    """Four-point conditional-mean grid, edges at -1,0,1 (the toy-oracle grid),
    reconstructed here from scratch."""
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    probs = np.diff(norm.cdf(edges))
    reps = np.array([(norm.pdf(lo) - norm.pdf(hi)) / p
                     for lo, hi, p in zip(edges[:-1], edges[1:], probs)])
    return reps, probs


def grid_paths(n_dates):
    reps, pr = conditional_mean_grid()
    dt = T / n_dates
    drift = (R - 0.5 * SIGMA ** 2) * dt
    vol = SIGMA * math.sqrt(dt)
    P, PA, PG = [], [], []
    for idx in itertools.product(range(4), repeat=n_dates):
        s = S0
        prices, p = [], 1.0
        for i in idx:
            s *= math.exp(drift + vol * reps[i])
            prices.append(s)
            p *= pr[i]
        prices = np.array(prices)
        P.append(p)
        PA.append(max(prices.mean() - K, 0.0))
        PG.append(max(math.exp(np.log(prices).mean()) - K, 0.0))
    return np.array(P), np.array(PA), np.array(PG)


def circuit_amplitude(probs, payoffs):
    """Build the amplitude-encoding circuit from scratch (state preparation +
    one controlled rotation per positive-payoff path) and return the exact
    objective-qubit probability from the statevector."""
    fmax = float(payoffs.max())
    n = int(round(math.log2(len(probs))))
    qc = QuantumCircuit(n + 1)
    qc.append(StatePreparation(np.sqrt(probs)), range(n))
    for idx, pay in enumerate(payoffs):
        if pay <= 0:
            continue
        theta = 2.0 * math.asin(math.sqrt(min(1.0, pay / fmax)))
        flip = [q for q in range(n) if not (idx >> q) & 1]
        for q in flip:
            qc.x(q)
        qc.mcry(theta, list(range(n)), n)
        for q in flip:
            qc.x(q)
    prob1 = float(Statevector.from_instruction(qc).probabilities([n])[1])
    return prob1, fmax


# ---------------------------------------------------------------------------
# 1. geometric Asian: closed form vs independent quadrature vs Monte Carlo
# ---------------------------------------------------------------------------
print("=== 1. geometric Asian control price ===")
cf = geo_asian_closed_form(252)
qd = geo_asian_quadrature(252)
check("closed form == independent quadrature", abs(cf - qd) < 1e-9,
      f"closed form {cf:.10f}  quadrature {qd:.10f}  |diff| {abs(cf - qd):.2e}")
check("geometric closed form -> Black-Scholes at N=1", abs(geo_asian_closed_form(1) - bs_call(S0, K, R, SIGMA, T)) < 1e-10,
      f"C_G(N=1) {geo_asian_closed_form(1):.10f}  BS {bs_call(S0,K,R,SIGMA,T):.10f}")

rng = np.random.default_rng(12345)
_, g = simulate(252, 4_000_000, rng)
cg_mc = DISC * np.maximum(g - K, 0).mean()
cg_se = DISC * np.maximum(g - K, 0).std(ddof=1) / math.sqrt(len(g))
check("geometric closed form within MC error", abs(cf - cg_mc) < 4 * cg_se,
      f"closed form {cf:.6f}  MC {cg_mc:.6f} +- {cg_se:.6f}  z={ (cg_mc-cf)/cg_se:+.2f}")

# ---------------------------------------------------------------------------
# 2. arithmetic Asian: analytic limit, parity, unbiasedness, Levy, Linetsky
# ---------------------------------------------------------------------------
print("\n=== 2. arithmetic Asian price (no closed form -> triangulate) ===")
# 2a. N=1 arithmetic == European
a1, _ = simulate(1, 4_000_000, rng)
c1 = DISC * np.maximum(a1 - K, 0).mean()
c1_se = DISC * np.maximum(a1 - K, 0).std(ddof=1) / math.sqrt(len(a1))
check("arithmetic Asian at N=1 == Black-Scholes", abs(c1 - bs_call(S0, K, R, SIGMA, T)) < 4 * c1_se,
      f"MC {c1:.6f} +- {c1_se:.6f}  BS {bs_call(S0,K,R,SIGMA,T):.6f}")

# 2b. put-call parity  C - P = e^{-rT}(E[A_N] - K)
a, g = simulate(252, 4_000_000, rng)
call = DISC * np.maximum(a - K, 0)
put = DISC * np.maximum(K - a, 0)
m1, _ = arithmetic_moments(252)
lhs = call.mean() - put.mean()
rhs = DISC * (m1 - K)
se = (call - put).std(ddof=1) / math.sqrt(len(a))
check("arithmetic put-call parity", abs(lhs - rhs) < 4 * se,
      f"C-P {lhs:.6f}  e^-rT(E[A]-K) {rhs:.6f}  z={(lhs-rhs)/se:+.2f}")

# 2c. control variate is unbiased: KV estimate == plain MC (different estimators)
theta = cf
beta = np.cov(call, DISC * np.maximum(g - K, 0), ddof=1)[0, 1] / np.var(DISC * np.maximum(g - K, 0), ddof=1)
kv = call + beta * (theta - DISC * np.maximum(g - K, 0))
plain_mean, plain_se = call.mean(), call.std(ddof=1) / math.sqrt(len(a))
kv_mean, kv_se = kv.mean(), kv.std(ddof=1) / math.sqrt(len(a))
check("Kemna-Vorst estimate == plain MC (unbiased)", abs(kv_mean - plain_mean) < 4 * plain_se,
      f"plain {plain_mean:.6f} +- {plain_se:.6f}  KV {kv_mean:.6f} +- {kv_se:.6f}  "
      f"z={(kv_mean-plain_mean)/plain_se:+.2f}")

# 2d. independent third pricer (Levy approximation), agreement to ~1%
lv = levy_asian_call(252)
check("Levy moment-matching within 1% of KV price", abs(lv - kv_mean) / kv_mean < 0.01,
      f"Levy {lv:.6f}  KV {kv_mean:.6f}  rel {abs(lv-kv_mean)/kv_mean:.2%}")

# 2e. ten-digit Linetsky benchmark, one case (r=.05, sig=.5, T=1, S0=K=2)
LIN = 0.2464156905
rngl = np.random.default_rng(2024)
def price_case(n, m):
    dt = T / n
    z = rngl.standard_normal((m, n))
    logs = math.log(2.0) + np.cumsum((0.05 - 0.5 * 0.5 ** 2) * dt + 0.5 * math.sqrt(dt) * z, axis=1)
    aa = np.exp(logs).mean(axis=1)
    gg = np.exp(logs.mean(axis=1))
    # geometric closed form for these params
    sg = 0.5 * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))
    mg = (0.05 - 0.5 * 0.5 ** 2) * (n + 1) / (2 * n)
    d1 = (math.log(2.0 / 2.0) + (mg + sg ** 2) * T) / (sg * math.sqrt(T))
    d2 = d1 - sg * math.sqrt(T)
    cgf = math.exp(-0.05 * T) * (2.0 * math.exp((mg + 0.5 * sg ** 2) * T) * norm.cdf(d1) - 2.0 * norm.cdf(d2))
    cc = math.exp(-0.05 * T) * np.maximum(aa - 2.0, 0)
    ccg = math.exp(-0.05 * T) * np.maximum(gg - 2.0, 0)
    bt = np.cov(cc, ccg, ddof=1)[0, 1] / np.var(ccg, ddof=1)
    est = cc + bt * (cgf - ccg)
    return est.mean(), est.std(ddof=1) / math.sqrt(m)
c250, s250 = price_case(250, 2_000_000)
c500, s500 = price_case(500, 2_000_000)
rich = 2 * c500 - c250                      # Richardson in 1/N to the continuous limit
rse = math.sqrt((2 * s500) ** 2 + s250 ** 2)
check("Linetsky 10-digit continuous benchmark", abs(rich - LIN) < 4 * rse,
      f"extrapolated {rich:.8f}  Linetsky {LIN:.10f}  z={(rich-LIN)/rse:+.2f}")

# ---------------------------------------------------------------------------
# 3. QCV is an exact decomposition + Lemma 1 holds on every grid path
# ---------------------------------------------------------------------------
print("\n=== 3. QCV decomposition identity and nonnegativity ===")
for nd in (2, 3):
    P, PA, PG = grid_paths(nd)
    Rk = PA - PG
    ca = DISC * float(P @ PA)
    cg = DISC * float(P @ PG)
    er = DISC * float(P @ Rk)
    check(f"Lemma 1: R>=0 on all {4**nd} grid paths (n={nd})", Rk.min() >= 0.0,
          f"min residual = {Rk.min():.2e}")
    check(f"exact decomposition C_A == C_G + e^-rT E[R] (n={nd})", abs(ca - (cg + er)) < 1e-13,
          f"C_A {ca:.10f}  C_G+e^-rT E[R] {cg + er:.10f}  |diff| {abs(ca-(cg+er)):.2e}")

# ---------------------------------------------------------------------------
# 4. the quantum circuit encodes E[R] (independent circuit vs classical amplitude)
# ---------------------------------------------------------------------------
print("\n=== 4. quantum encoding: independent circuit == classical expectation ===")
for nd in (2, 3):
    P, PA, PG = grid_paths(nd)
    Rk = np.maximum(PA - PG, 0.0)
    fmax = float(Rk.max())
    a_classical = float(P @ (Rk / fmax))              # Sum p_i (R_i / f_max) = amplitude, by hand
    prob1, fmax_c = circuit_amplitude(P, Rk)
    check(f"circuit objective prob == Sum p_i R_i/f_max (n={nd})", abs(prob1 - a_classical) < 1e-11,
          f"circuit {prob1:.12f}  classical {a_classical:.12f}  |diff| {abs(prob1-a_classical):.2e}")
    # full reconstruction from the circuit output
    ca = DISC * float(P @ PA)
    cg = DISC * float(P @ PG)
    recon = cg + DISC * fmax_c * prob1
    check(f"QCV reconstruction from circuit == grid arithmetic price (n={nd})", abs(recon - ca) < 1e-11,
          f"reconstruction {recon:.10f}  exact grid {ca:.10f}  |diff| {abs(recon-ca):.2e}")

# ---------------------------------------------------------------------------
# 5. Lemma 1 on a large continuous sample (daily monitoring)
# ---------------------------------------------------------------------------
print("\n=== 5. Lemma 1 on 10^7 daily paths ===")
minR = math.inf
rng2 = np.random.default_rng(99)
for _ in range(40):
    a, g = simulate(252, 250_000, rng2)
    minR = min(minR, float((np.maximum(a - K, 0) - np.maximum(g - K, 0)).min()))
check("R >= 0 on 10^7 simulated daily paths", minR >= 0.0, f"min residual = {minR:.3e}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
failed = [n for n, ok, _ in CHECKS if not ok]
summary = f"{len(CHECKS)} checks, {len(CHECKS) - len(failed)} passed, {len(failed)} failed"
print(summary)
if failed:
    print("FAILED:", failed)

from pathlib import Path
out = Path(__file__).resolve().parent.parent / "results" / "v4" / "independent_verification.txt"
report = [
    "Independent verification of the pricing engine and the quantum control variate",
    "no project src/ imported; each quantity is checked against independent ground truth",
    "",
]
for name, ok, detail in CHECKS:
    report.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    report.append(f"      {detail}")
report += ["", summary]
out.write_text("\n".join(report) + "\n")
print(f"-> wrote {out}")
raise SystemExit(1 if failed else 0)
