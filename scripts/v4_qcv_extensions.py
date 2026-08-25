#!/usr/bin/env python3
"""v4: how far does the quantum control variate go?

Three extensions of the KV quantum control variate, each validated against a
known result before any number is reported:

1. Block-geometric ladder (BS Asian, N=252 daily).
   Partition the dates into k equal blocks, G_j = geometric mean of block j,
   B_k = (1/k) sum_j G_j. Block AM-GM gives A >= B_k >= G pathwise, so the
   residual R_k = max(A-K,0) - max(B_k-K,0) is nonnegative and encodable.
   k=1 is Kemna-Vorst; k=N is the payoff itself (zero residual).
   Gates: k=1 must reproduce the v3 numbers (f_max(R)=2.864, ratio 16.0);
   the analytic Gaussian moments of (log G_1..log G_k) must match sample
   moments; the k=2 control price from deterministic 2-D quadrature must
   match Monte Carlo; residuals must be >= 0 on every path.

2. Basket (d=3 equal-weight arithmetic basket call, geometric-basket control).
   Weighted AM-GM gives the same pathwise ordering. Gates: MC geometric-
   basket price vs the lognormal closed form; put-call parity for the
   arithmetic basket; rho -> 1 limit vs the single-asset Black-Scholes price.

3. Heston Asian (full-truncation Euler, daily).
   AM-GM is model-free, so the residual stays nonnegative; only the control
   price needs the model. Gates: the semi-analytic Heston European pricer is
   self-tested in its xi -> 0 Black-Scholes limit, then the Euler simulator
   is tested against it on the European payoff.

Outputs: results/v4/qcv_extensions.txt, results/v4/qcv_ladder.png,
results/v4/qcv_generality.png.  Exits nonzero if any gate fails.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / "results" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.stats import norm

S0, K, RATE, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N = 252
TAIL = float(norm.cdf(-3.0))          # 1.3499e-3, the paper's convention
N_PATHS = 2_000_000
CHUNK = 100_000
KS = [1, 2, 3, 4, 6, 12]
SEED = 20260704

# crossover model constants (must match v3_crossover_band.py)
D_CHK = 9.8e9 / 6000 * 252 / 26       # Chakrabarti-scaled per-oracle T-count
VAR_CV = 0.0494                        # KV-CV variance from the parameter sweep
QCV_CHARGE = {1: 1.1, 2: 1.2}          # conservative oracle multipliers

OUT = _REPO / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

GATES: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str) -> None:
    GATES.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def bs_call(s0: float, k: float, r: float, sig: float, t: float) -> float:
    d1 = (math.log(s0 / k) + (r + sig * sig / 2) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    return s0 * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def lognormal_call(mu: float, var: float, k: float, r: float, t: float) -> float:
    """e^{-rt} E[max(e^X - k, 0)] for X ~ N(mu, var)."""
    sd = math.sqrt(var)
    d1 = (mu - math.log(k) + var) / sd
    d2 = d1 - sd
    return math.exp(-r * t) * (math.exp(mu + var / 2) * norm.cdf(d1) - k * norm.cdf(d2))


# ---------------------------------------------------------------- 1. ladder
def block_moments(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Analytic mean vector and covariance of (log G_1, .., log G_k)."""
    m = N // k
    dt = T / N
    j = np.arange(1, k + 1)
    tbar = dt * ((j - 1) * m + (m + 1) / 2.0)              # mean time of block j
    mu = math.log(S0) + (RATE - SIGMA**2 / 2) * tbar
    cov = SIGMA**2 * np.minimum.outer(tbar, tbar)
    var = SIGMA**2 * dt * ((j - 1) * m + (m + 1) * (2 * m + 1) / (6.0 * m))
    np.fill_diagonal(cov, var)
    return mu, cov


def control_price_quad_k2() -> float:
    """Deterministic price of e^{-rT} E[max((e^Y1+e^Y2)/2 - K, 0)], k=2."""
    mu, cov = block_moments(2)
    L = np.linalg.cholesky(cov)

    def payoff(z2: float, z1: float) -> float:
        y1 = mu[0] + L[0, 0] * z1
        y2 = mu[1] + L[1, 0] * z1 + L[1, 1] * z2
        val = 0.5 * (math.exp(y1) + math.exp(y2)) - K
        if val <= 0.0:
            return 0.0
        return val * norm.pdf(z1) * norm.pdf(z2)

    val, _ = integrate.dblquad(payoff, -8, 8, -8, 8, epsabs=1e-10, epsrel=1e-10)
    return math.exp(-RATE * T) * val


def control_price_cond_k2() -> float:
    """k=2 control price by conditioning: the inner integral over the second
    block is a lognormal partial expectation with a closed form, so the price
    reduces to one smooth 1-D integral."""
    mu, cov = block_moments(2)
    L = np.linalg.cholesky(cov)

    def inner(z1: float) -> float:
        y1 = mu[0] + L[0, 0] * z1
        m2 = mu[1] + L[1, 0] * z1
        s2 = L[1, 1]
        e1 = math.exp(y1)
        if e1 >= 2 * K:                       # payoff positive for all z2
            return (e1 / 2 - K) + 0.5 * math.exp(m2 + s2 * s2 / 2)
        zstar = (math.log(2 * K - e1) - m2) / s2
        return ((e1 / 2 - K) * norm.cdf(-zstar)
                + 0.5 * math.exp(m2 + s2 * s2 / 2) * norm.cdf(s2 - zstar))

    val, _ = integrate.quad(lambda z: inner(z) * norm.pdf(z), -10, 10,
                            limit=400, epsabs=1e-12, epsrel=1e-12)
    return math.exp(-RATE * T) * val


def run_ladder(rng: np.random.Generator) -> dict:
    dt = T / N
    drift = (RATE - SIGMA**2 / 2) * dt
    vol = SIGMA * math.sqrt(dt)
    diffs = {k: [] for k in KS}
    a_all, g_pay, b2_pay = [], [], []
    min_resid = {k: np.inf for k in KS}
    sample_logg2 = []
    for _ in range(N_PATHS // CHUNK):
        z = rng.standard_normal((CHUNK, N))
        logs = math.log(S0) + np.cumsum(drift + vol * z, axis=1)
        a = np.exp(logs).mean(axis=1)
        a_all.append(a)
        for k in KS:
            lg = logs.reshape(CHUNK, k, N // k).mean(axis=2)   # log G_j
            bk = np.exp(lg).mean(axis=1)
            d = a - bk
            min_resid[k] = min(min_resid[k], float(d.min()))
            diffs[k].append(d.astype(np.float32))
            if k == 1:
                g_pay.append(np.maximum(np.exp(lg[:, 0]) - K, 0.0))
            if k == 2:
                b2_pay.append(np.maximum(bk - K, 0.0))
                if len(sample_logg2) < 3:
                    sample_logg2.append(lg.copy())
    a = np.concatenate(a_all)
    fmax_a = float(np.quantile(a, 1 - TAIL)) - K
    rows = {}
    for k in KS:
        d = np.concatenate(diffs[k])
        q = float(np.quantile(d, 1 - TAIL))
        clip = float(np.maximum(d - q, 0.0).mean())
        rows[k] = dict(fmax_r=q, ratio=fmax_a / q, mean=float(d.mean()),
                       clip=clip, min=min_resid[k])
    # gates
    gate("L1 reproduce v3 k=1", abs(rows[1]["fmax_r"] - 2.864) < 0.05
         and abs(rows[1]["ratio"] - 16.0) < 0.4,
         f"f_max(R)={rows[1]['fmax_r']:.3f} (v3: 2.864), ratio={rows[1]['ratio']:.1f} (v3: 16.0)")
    gate("L2 pathwise A>=B_k", all(rows[k]["min"] >= -1e-10 for k in KS),
         f"min residual over {N_PATHS} paths x {len(KS)} ladders = "
         f"{min(rows[k]['min'] for k in KS):.2e}")
    # analytic vs sample moments (k=2)
    mu2, cov2 = block_moments(2)
    lg = np.concatenate(sample_logg2)
    mu_err = float(np.abs(lg.mean(axis=0) - mu2).max())
    cov_err = float(np.abs(np.cov(lg.T) - cov2).max() / np.abs(cov2).max())
    gate("L3 block moments analytic=sample", mu_err < 5e-4 and cov_err < 5e-3,
         f"max mean err {mu_err:.1e}, max rel cov err {cov_err:.1e} (300k paths)")
    # control price: closed form (k=1), quadrature vs MC (k=2)
    gp = np.concatenate(g_pay)
    cg_mc = math.exp(-RATE * T) * float(gp.mean())
    cg_se = math.exp(-RATE * T) * float(gp.std(ddof=1)) / math.sqrt(len(gp))
    mu1, cov1 = block_moments(1)
    cg_cf = lognormal_call(float(mu1[0]), float(cov1[0, 0]), K, RATE, T)
    z1 = (cg_mc - cg_cf) / cg_se
    gate("L4a k=1 control price vs closed form", abs(z1) < 3.5,
         f"MC {cg_mc:.6f} +- {cg_se:.6f} vs C_G {cg_cf:.6f} (z={z1:+.2f})")
    b2 = np.concatenate(b2_pay)
    cb2_mc = math.exp(-RATE * T) * float(b2.mean())
    cb2_se = math.exp(-RATE * T) * float(b2.std(ddof=1)) / math.sqrt(len(b2))
    cb2_quad = control_price_quad_k2()
    cb2_cond = control_price_cond_k2()
    z2 = (cb2_mc - cb2_cond) / cb2_se
    gate("L4b k=2 control price cond-1D vs MC", abs(z2) < 3.5,
         f"cond-1D {cb2_cond:.6f} | MC {cb2_mc:.6f} +- {cb2_se:.6f} (z={z2:+.2f})")
    gate("L4c k=2 cond-1D vs dblquad internal", abs(cb2_quad - cb2_cond) < 1e-6,
         f"|dblquad - cond-1D| = {abs(cb2_quad - cb2_cond):.2e}")
    # crossover implications
    disc = math.exp(-RATE * T)
    eps = {}
    for k in (1, 2):
        eps[k] = N * VAR_CV / (6 * QCV_CHARGE[k] * D_CHK * disc * rows[k]["fmax_r"])
    gate("L5 reproduce v3 crossover k=1", abs(eps[1] - 4.4e-8) / 4.4e-8 < 0.03,
         f"eps*_QCV,CV(k=1) = {eps[1]:.2e} (v3: 4.4e-8)")
    return dict(fmax_a=fmax_a, rows=rows, eps=eps,
                cg_cf=cg_cf, cb2_quad=cb2_quad, cb2_cond=cb2_cond)


# ---------------------------------------------------------------- 2. basket
def run_basket(rng: np.random.Generator) -> dict:
    d = 3
    out = {}
    for rho in (0.2, 0.5, 0.8, 0.999):
        c = np.full((d, d), rho)
        np.fill_diagonal(c, 1.0)
        lmat = np.linalg.cholesky(c)
        z = rng.standard_normal((N_PATHS, d)) @ lmat.T
        st = S0 * np.exp((RATE - SIGMA**2 / 2) * T + SIGMA * math.sqrt(T) * z)
        a = st.mean(axis=1)
        g = np.exp(np.log(st).mean(axis=1))
        resid = a - g
        disc = math.exp(-RATE * T)
        pay_a = np.maximum(a - K, 0.0)
        pay_g = np.maximum(g - K, 0.0)
        ca = disc * float(pay_a.mean())
        ca_se = disc * float(pay_a.std(ddof=1)) / math.sqrt(len(a))
        cg = disc * float(pay_g.mean())
        cg_se = disc * float(pay_g.std(ddof=1)) / math.sqrt(len(a))
        # closed-form geometric basket
        mu_g = math.log(S0) + (RATE - SIGMA**2 / 2) * T
        var_g = SIGMA**2 * T * float(np.ones(d) @ c @ np.ones(d)) / d**2
        cg_cf = lognormal_call(mu_g, var_g, K, RATE, T)
        fa = float(np.quantile(a, 1 - TAIL)) - K
        fr = float(np.quantile(resid, 1 - TAIL))
        out[rho] = dict(ratio=fa / fr, fmax_a=fa, fmax_r=fr, ca=ca,
                        min=float(resid.min()))
        if rho == 0.5:
            zg = (cg - cg_cf) / cg_se
            gate("B1 geometric basket vs closed form", abs(zg) < 3.5,
                 f"rho=0.5: MC {cg:.5f} +- {cg_se:.5f} vs CF {cg_cf:.5f} (z={zg:+.2f})")
            pay_p = np.maximum(K - a, 0.0)
            lhs = ca - disc * float(pay_p.mean())
            rhs = S0 - K * disc                       # E[A] = S0 e^{rT}
            se = disc * float((pay_a - pay_p).std(ddof=1)) / math.sqrt(len(a))
            zp = (lhs - rhs) / se
            gate("B2 arithmetic basket put-call parity", abs(zp) < 3.5,
                 f"C-P {lhs:.5f} vs S0-Ke^-rT {rhs:.5f} (z={zp:+.2f})")
        if rho == 0.999:
            zbs = (ca - bs_call(S0, K, RATE, SIGMA, T)) / ca_se
            gate("B3 rho->1 limit vs Black-Scholes", abs(zbs) < 3.5,
                 f"basket {ca:.5f} +- {ca_se:.5f} vs BS {bs_call(S0, K, RATE, SIGMA, T):.5f} "
                 f"(z={zbs:+.2f})")
    gate("B4 pathwise basket A>=G", all(v["min"] >= -1e-10 for v in out.values()),
         f"min residual {min(v['min'] for v in out.values()):.2e}")
    return out


# ---------------------------------------------------------------- 3. Heston
V0, KAPPA, THETA, XI, RHO_SV = 0.04, 2.0, 0.04, 0.3, -0.7
N_HESTON = 500_000


def heston_call_cf(s0, k, r, t, v0, kappa, theta, xi, rho) -> float:
    """Semi-analytic Heston call, little-trap formulation."""
    lnk = math.log(k)

    def phi(u):
        u = complex(u)
        b = kappa - rho * xi * 1j * u
        dd = np.sqrt(b * b + xi * xi * (1j * u + u * u))
        g = (b - dd) / (b + dd)
        e = np.exp(-dd * t)
        c1 = (b - dd) * t - 2.0 * np.log((1 - g * e) / (1 - g))
        return np.exp(1j * u * (math.log(s0) + r * t)
                      + kappa * theta / xi**2 * c1
                      + v0 / xi**2 * (b - dd) * (1 - e) / (1 - g * e))

    def integrand(u, j):
        if j == 1:
            val = phi(u - 1j) / phi(-1j)
        else:
            val = phi(u)
        return (np.exp(-1j * u * lnk) * val / (1j * u)).real

    p = []
    for j in (1, 2):
        iv, _ = integrate.quad(integrand, 1e-9, 250.0, args=(j,), limit=500)
        p.append(0.5 + iv / math.pi)
    return s0 * p[0] - k * math.exp(-r * t) * p[1]


def run_heston(rng: np.random.Generator) -> dict:
    # gate H1: CF pricer degenerates to Black-Scholes as xi -> 0
    bs_ref = bs_call(S0, K, RATE, math.sqrt(THETA), T)
    cf_lim = heston_call_cf(S0, K, RATE, T, THETA, 5.0, THETA, 1e-4, 0.0)
    gate("H1 Heston CF pricer BS limit", abs(cf_lim - bs_ref) < 1e-4,
         f"CF(xi->0) {cf_lim:.6f} vs BS(sqrt(theta)) {bs_ref:.6f}")
    # full-truncation Euler
    dt = T / N
    s = np.full(N_HESTON, S0)
    v = np.full(N_HESTON, V0)
    sum_s = np.zeros(N_HESTON)
    sum_log = np.zeros(N_HESTON)
    for _ in range(N):
        z1 = rng.standard_normal(N_HESTON)
        z2 = RHO_SV * z1 + math.sqrt(1 - RHO_SV**2) * rng.standard_normal(N_HESTON)
        vp = np.maximum(v, 0.0)
        s *= np.exp((RATE - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
        v += KAPPA * (THETA - vp) * dt + XI * np.sqrt(vp * dt) * z2
        sum_s += s
        sum_log += np.log(s)
    disc = math.exp(-RATE * T)
    pay_e = np.maximum(s - K, 0.0)
    ce_mc = disc * float(pay_e.mean())
    ce_se = disc * float(pay_e.std(ddof=1)) / math.sqrt(N_HESTON)
    ce_cf = heston_call_cf(S0, K, RATE, T, V0, KAPPA, THETA, XI, RHO_SV)
    zz = (ce_mc - ce_cf) / ce_se
    gate("H2 Euler European vs Heston CF", abs(zz) < 3.5,
         f"MC {ce_mc:.5f} +- {ce_se:.5f} vs CF {ce_cf:.5f} (z={zz:+.2f})")
    a = sum_s / N
    g = np.exp(sum_log / N)
    resid = a - g
    gate("H3 pathwise Heston A>=G", float(resid.min()) >= -1e-10,
         f"min residual {float(resid.min()):.2e}")
    fa = float(np.quantile(a, 1 - TAIL)) - K
    fr = float(np.quantile(resid, 1 - TAIL))
    return dict(fmax_a=fa, fmax_r=fr, ratio=fa / fr)


# ---------------------------------------------------------------- figures
def make_figures(lad: dict, bask: dict, hes: dict) -> None:
    ks = np.array(KS, dtype=float)
    fr = np.array([lad["rows"][k]["fmax_r"] for k in KS])
    ratio = np.array([lad["rows"][k]["ratio"] for k in KS])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax0.loglog(ks, fr, "o-", color="#16a34a", label=r"$f_{\max}(R_k)$ measured")
    ax0.loglog(ks, fr[0] * ks ** (-1.5), "--", color="gray",
               label=r"$k^{-3/2}$ reference")
    ax0.set_xlabel("number of blocks $k$")
    ax0.set_ylabel(r"residual payoff bound $f_{\max}(R_k)$ (\$)")
    ax0.set_title("(a) Residual bound vs block count")
    ax0.set_xticks(ks)
    ax0.set_xticklabels([str(k) for k in KS])
    ax0.legend(fontsize=9)
    ax0.grid(True, which="both", alpha=0.25)

    ax1.semilogy(ks, ratio, "s-", color="#7c3aed")
    for k, r_ in zip(KS, ratio):
        ax1.annotate(f"{r_:.0f}$\\times$", (k, r_), textcoords="offset points",
                     xytext=(4, 6), fontsize=9)
    ax1.set_xlabel("number of blocks $k$")
    ax1.set_ylabel(r"query ratio $f_{\max}(A)/f_{\max}(R_k)$")
    ax1.set_title("(b) Oracle-query reduction")
    ax1.set_xticks(ks)
    ax1.set_xticklabels([str(k) for k in KS])
    ax1.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "qcv_ladder.png", dpi=200, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = ["BS Asian\n$N=252$", r"basket $\rho=0.2$", r"basket $\rho=0.5$",
              r"basket $\rho=0.8$", "Heston Asian\n$\\xi=0.3$"]
    vals = [lad["rows"][1]["ratio"], bask[0.2]["ratio"], bask[0.5]["ratio"],
            bask[0.8]["ratio"], hes["ratio"]]
    colors = ["#2563eb", "#16a34a", "#16a34a", "#16a34a", "#c0392b"]
    ax.bar(range(len(vals)), vals, color=colors)
    for i, v_ in enumerate(vals):
        ax.text(i, v_ + 0.6, f"{v_:.1f}$\\times$", ha="center", fontsize=10)
    ax.set_xticks(range(len(vals)), labels, fontsize=9)
    ax.set_ylabel(r"query ratio $f_{\max}(A)/f_{\max}(R)$")
    ax.set_title("Geometric control variate across contracts and models ($k=1$)")
    ax.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(OUT / "qcv_generality.png", dpi=200, bbox_inches="tight")
    print(f"-> wrote {OUT}/qcv_ladder.png, {OUT}/qcv_generality.png")


def main() -> int:
    rng = np.random.default_rng(SEED)
    print("=== 1. block-geometric ladder (BS Asian, N=252) ===")
    lad = run_ladder(rng)
    print("=== 2. basket (d=3, equal weights) ===")
    bask = run_basket(rng)
    print("=== 3. Heston Asian ===")
    hes = run_heston(rng)

    lines = [
        "v4 QCV extensions: block ladder, basket, Heston",
        f"paths: {N_PATHS} (ladder, basket), {N_HESTON} (Heston); tail mass {TAIL:.4e}",
        f"seed {SEED}",
        "",
        f"BS Asian N=252: f_max(A) = {lad['fmax_a']:.2f}",
        "  k   f_max(R_k)   ratio    mean(A-B_k)   clip bias E[(R-cap)+]",
    ]
    for k in KS:
        r_ = lad["rows"][k]
        lines.append(f"  {k:2d}   {r_['fmax_r']:8.3f}   {r_['ratio']:6.1f}"
                     f"    {r_['mean']:.4f}        {r_['clip']:.2e}")
    lines += [
        "",
        f"control prices: C_G closed form = {lad['cg_cf']:.6f} (k=1); "
        f"k=2 block control by conditioning (1-D quadrature) = {lad['cb2_cond']:.6f}",
        f"crossover (KV-CV classical baseline, charge {QCV_CHARGE[1]}D / {QCV_CHARGE[2]}D):",
        f"  eps*_QCV,CV(k=1) = {lad['eps'][1]:.2e}   eps*_QCV,CV(k=2) = {lad['eps'][2]:.2e}",
        "",
        "basket d=3 (single maturity, equal weights):",
    ]
    for rho in (0.2, 0.5, 0.8):
        b = bask[rho]
        lines.append(f"  rho={rho}: f_max(A)={b['fmax_a']:.2f}  f_max(R)={b['fmax_r']:.3f}"
                     f"  ratio={b['ratio']:.1f}")
    lines += [
        "",
        f"Heston Asian (v0=theta={V0}, kappa={KAPPA}, xi={XI}, rho={RHO_SV}):",
        f"  f_max(A)={hes['fmax_a']:.2f}  f_max(R)={hes['fmax_r']:.3f}  ratio={hes['ratio']:.1f}",
        "",
        "validation gates:",
    ]
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in GATES]
    (OUT / "qcv_extensions.txt").write_text("\n".join(lines) + "\n")
    print(f"-> wrote {OUT}/qcv_extensions.txt")

    make_figures(lad, bask, hes)
    failed = [n for n, ok, _ in GATES if not ok]
    if failed:
        print(f"FAILED GATES: {failed}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
