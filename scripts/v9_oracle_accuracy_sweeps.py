"""Encoded-model accuracy sweeps for the finite-grid reference Asian oracle.

The MPS validation (scripts/validate_full_asian_oracle.py) shows the circuit
reproduces the exact encoded finite-grid model to ~1e-11, so
`enumerate_encoded_asian` gives the oracle's exact decoded price.  These
sweeps measure the remaining error, encoded model versus the continuous
normal-shock discretely monitored Asian price, decomposed by source.  The
oracle source is not modified; every knob is a constructor parameter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import roots_hermite
from scipy.stats import norm, qmc

from qc_option_pricing.quantum.asian_oracle import (
    AsianGridSpec,
    build_asian_model,
    enumerate_encoded_asian,
    estimate_asian_oracle_resources,
    gauss_hermite_normal_grid,
    residual_cap_from_bias_budget,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v9" / "oracle_accuracy_sweeps.json"
results: dict = {}


# ----------------------------------------------------------------------
# References
# ----------------------------------------------------------------------

def exact_two_date_price(s0, strike, rate, sigma, maturity):
    """Near-exact 2-date price: closed-form inner expectation + 1D quadrature.

    Conditional on z1 the payoff is increasing in z2, so the conditional
    expectation is a lognormal partial expectation with an explicit threshold.
    """
    dt = maturity / 2
    d = (rate - 0.5 * sigma * sigma) * dt
    s = sigma * math.sqrt(dt)

    def inner(z1):
        S1 = s0 * math.exp(d + s * z1)

        def f(z2):
            return 0.5 * S1 * (1.0 + math.exp(d + s * z2)) - strike

        if f(-40.0) >= 0.0:
            zstar = -40.0
        elif f(40.0) <= 0.0:
            return 0.0
        else:
            zstar = brentq(f, -40.0, 40.0, xtol=1e-14)
        a = 0.5 * S1 - strike
        b = 0.5 * S1 * math.exp(d)
        return a * norm.sf(zstar) + b * math.exp(0.5 * s * s) * norm.sf(zstar - s)

    val, _ = quad(lambda z1: inner(z1) * norm.pdf(z1), -12.0, 12.0,
                  limit=400, epsabs=1e-13, epsrel=1e-13)
    return math.exp(-rate * maturity) * val


def grid_price_exact(n, s0, strike, rate, sigma, maturity, pts, prob):
    """Discrete-shock model priced with exact float arithmetic (no rounding)."""
    pts = np.asarray(pts, dtype=float)
    prob = np.asarray(prob, dtype=float)
    dt = maturity / n
    drift = (rate - 0.5 * sigma * sigma) * dt
    dif = sigma * math.sqrt(dt)
    grids = np.meshgrid(*([pts] * n), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=1)
    pgrids = np.meshgrid(*([prob] * n), indexing="ij")
    P = np.ones(Z.shape[0])
    for g in pgrids:
        P *= g.ravel()
    logS = math.log(s0) + np.cumsum(drift + dif * Z, axis=1)
    A = np.exp(logS).mean(axis=1)
    payoff = np.maximum(A - strike, 0.0)
    return math.exp(-rate * maturity) * float(P @ payoff)


def true_price_gh_tensor(n, s0, strike, rate, sigma, maturity, m_nodes):
    nodes, weights = roots_hermite(m_nodes)
    pts = math.sqrt(2.0) * nodes
    prob = weights / math.sqrt(math.pi)
    return grid_price_exact(n, s0, strike, rate, sigma, maturity, pts, prob)


def true_price_qmc(n, s0, strike, rate, sigma, maturity, log2_paths=22, reps=8, seed=7):
    dt = maturity / n
    drift = (rate - 0.5 * sigma * sigma) * dt
    dif = sigma * math.sqrt(dt)
    disc = math.exp(-rate * maturity)
    estimates = []
    for r in range(reps):
        sob = qmc.Sobol(d=n, scramble=True, seed=seed + r)
        U = sob.random(1 << log2_paths)
        Z = norm.ppf(np.clip(U, 1e-15, 1 - 1e-15))
        logS = math.log(s0) + np.cumsum(drift + dif * Z, axis=1)
        A = np.exp(logS).mean(axis=1)
        estimates.append(disc * float(np.maximum(A - strike, 0.0).mean()))
    return float(np.mean(estimates)), float(np.std(estimates, ddof=1) / math.sqrt(reps))


# ----------------------------------------------------------------------
# Shock grids
# ----------------------------------------------------------------------

def stratified_grid(q):
    """Equal-probability conditional-mean quantizer (uniform p => Hadamard prep)."""
    k = 1 << q
    edges = norm.ppf(np.linspace(0.0, 1.0, k + 1))
    pdf = norm.pdf(edges)
    pts = (pdf[:-1] - pdf[1:]) * k
    return tuple(float(x) for x in pts), tuple(1.0 / k for _ in range(k))


def lloyd_max_grid(q, iters=4000):
    k = 1 << q
    pts = np.array(stratified_grid(q)[0])
    for _ in range(iters):
        edges = np.concatenate(([-np.inf], (pts[:-1] + pts[1:]) / 2.0, [np.inf]))
        cdf, pdf = norm.cdf(edges), norm.pdf(edges)
        pdf[~np.isfinite(edges)] = 0.0
        new_pts = (pdf[:-1] - pdf[1:]) / np.diff(cdf)
        if np.allclose(new_pts, pts, atol=1e-15):
            pts = new_pts
            break
        pts = new_pts
    edges = np.concatenate(([-np.inf], (pts[:-1] + pts[1:]) / 2.0, [np.inf]))
    prob = np.diff(norm.cdf(edges))
    prob = prob / prob.sum()
    return tuple(float(x) for x in pts), tuple(float(p) for p in prob)


def rounded_grid(pts, ss):
    """The grid the oracle actually uses after integer quantization."""
    return tuple(round(z * ss) / ss for z in pts)


def moment_rematched_probs(pts_rounded, k_moments=None):
    """Re-solve probabilities on the *rounded* points to match N(0,1) moments.

    Solves the Vandermonde system sum_i p_i x_i^m = E[Z^m], m=0..k-1.
    Returns None if any probability is negative (system infeasible).
    """
    x = np.asarray(pts_rounded, dtype=float)
    k = len(x) if k_moments is None else k_moments
    # E[Z^m] = (m-1)!! for even m, 0 for odd m
    target = [float(np.prod(np.arange(m - 1, 0, -2))) if m % 2 == 0 else 0.0 for m in range(k)]
    V = np.vander(x, k, increasing=True).T
    p = np.linalg.solve(V, np.array(target))
    if np.any(p < -1e-12):
        return None
    p = np.clip(p, 0.0, None)
    return tuple(float(v) for v in (p / p.sum()))


def encoded_price(spec):
    ref = enumerate_encoded_asian(spec)
    disc = math.exp(-spec.rate * spec.maturity)
    return disc * ref.clipped_raw_payoff_undiscounted, ref


def make_spec(n, s0, strike, rate, sigma, maturity, pts, prob, ss, ps, **kw):
    return AsianGridSpec(
        n_dates=n, shock_points=pts, shock_probabilities=prob,
        s0=s0, strike=strike, rate=rate, volatility=sigma, maturity=maturity,
        shock_scale=ss, price_scale=ps, **kw,
    )


# ======================================================================
# Instance A: n=2, S0=K=2, r=5%, sigma=30%, T=1   (circuit-simulable size)
# Instance B: n=4, S0=K=100, r=5%, sigma=20%, T=1 (repo's scaling params)
# ======================================================================
A = dict(n=2, s0=2.0, strike=2.0, rate=0.05, sigma=0.30, maturity=1.0)
B = dict(n=4, s0=100.0, strike=100.0, rate=0.05, sigma=0.20, maturity=1.0)

truth_A = exact_two_date_price(A["s0"], A["strike"], A["rate"], A["sigma"], A["maturity"])
tA_301 = true_price_gh_tensor(m_nodes=301, **A)
tA_qmc, tA_qmc_se = true_price_qmc(log2_paths=22, reps=8, **A)
print(f"Instance A truth: exact={truth_A:.10f}  GH301={tA_301:.8f}  "
      f"QMC={tA_qmc:.8f}+/-{tA_qmc_se:.1e}")

tB_qmc, tB_qmc_se = true_price_qmc(log2_paths=22, reps=8, **B)
truth_B = tB_qmc
print(f"Instance B truth: QMC={tB_qmc:.6f}+/-{tB_qmc_se:.1e}")
results["truth"] = {"A": {"exact_quadrature": truth_A, "gh301": tA_301,
                          "qmc": tA_qmc, "qmc_se": tA_qmc_se},
                    "B": {"qmc": tB_qmc, "qmc_se": tB_qmc_se}}

# ----------------------------------------------------------------------
# Sweep 1: pure quadrature error (exact arithmetic, no rounding at all)
# ----------------------------------------------------------------------
print("\n== Sweep 1: pure shock-quadrature error (exact arithmetic) ==")
grid_menu = [
    ("binary +/-1 (q=1, = GH 2pt)", (-1.0, 1.0), (0.5, 0.5)),
    ("Gauss-Hermite 4pt (q=2)", *gauss_hermite_normal_grid(2)),
    ("Gauss-Hermite 8pt (q=3)", *gauss_hermite_normal_grid(3)),
    ("Gauss-Hermite 16pt (q=4)", *gauss_hermite_normal_grid(4)),
    ("stratified 4pt (q=2)", *stratified_grid(2)),
    ("stratified 8pt (q=3)", *stratified_grid(3)),
    ("stratified 16pt (q=4)", *stratified_grid(4)),
    ("Lloyd-Max 4pt (q=2)", *lloyd_max_grid(2)),
    ("Lloyd-Max 8pt (q=3)", *lloyd_max_grid(3)),
    ("Lloyd-Max 16pt (q=4)", *lloyd_max_grid(4)),
]
rows = []
for name, pts, prob in grid_menu:
    eA = grid_price_exact(pts=pts, prob=prob, **A) - truth_A
    eB = grid_price_exact(pts=pts, prob=prob, **B) - truth_B
    rows.append({"grid": name, "err_A": eA, "err_B": eB})
    print(f"  {name:32s} err_A={eA:+.6f}   err_B={eB:+.6f}")
results["sweep1_quadrature"] = rows

# ----------------------------------------------------------------------
# Sweep 2: shock integer rounding (exact price arithmetic on rounded grid)
# plus the moment-rematched-probability fix
# ----------------------------------------------------------------------
print("\n== Sweep 2: shock rounding error at GH 4pt (exact price arithmetic) ==")
gh4_pts, gh4_prob = gauss_hermite_normal_grid(2)
quad_A = grid_price_exact(pts=gh4_pts, prob=gh4_prob, **A)
quad_B = grid_price_exact(pts=gh4_pts, prob=gh4_prob, **B)
rows = []
for ss in (1, 2, 4, 8, 16, 32, 64):
    rp = rounded_grid(gh4_pts, ss)
    if len(set(rp)) < len(rp):
        print(f"  shock_scale={ss:3d}  [points collapse; skipped]")
        continue
    plain_A = grid_price_exact(pts=rp, prob=gh4_prob, **A) - quad_A
    plain_B = grid_price_exact(pts=rp, prob=gh4_prob, **B) - quad_B
    mm = moment_rematched_probs(rp)
    mm_A = mm_B = None
    if mm is not None:
        mm_A = grid_price_exact(pts=rp, prob=mm, **A) - quad_A
        mm_B = grid_price_exact(pts=rp, prob=mm, **B) - quad_B
    rows.append({"shock_scale": ss, "plain_A": plain_A, "plain_B": plain_B,
                 "rematched_A": mm_A, "rematched_B": mm_B})
    mm_txt = ("rematched: " + (f"A={mm_A:+.6f} B={mm_B:+.6f}" if mm is not None else "infeasible"))
    print(f"  shock_scale={ss:3d}  plain: A={plain_A:+.6f} B={plain_B:+.6f}   {mm_txt}")
results["sweep2_shock_rounding"] = rows

# ----------------------------------------------------------------------
# Sweep 3: price fixed-point rounding (full encoded model) + Richardson
# ----------------------------------------------------------------------
print("\n== Sweep 3: price_scale (encoded, GH4, shock_scale=64) ==")
rows, prices_by_ps = [], {}
for ps in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    spec = make_spec(pts=gh4_pts, prob=gh4_prob, ss=64, ps=ps, **A)
    price, _ = encoded_price(spec)
    prices_by_ps[ps] = price
    rows.append({"price_scale": ps, "price": price, "error": price - truth_A})
    print(f"  price_scale={ps:4d}  price={price:.6f}  error={price-truth_A:+.6f}")
results["sweep3_price_scale"] = rows

print("  Richardson pairs (2*p(2s) - p(s)):")
rich = []
for ps in (4, 8, 16, 32, 64, 128):
    ext = 2.0 * prices_by_ps[2 * ps] - prices_by_ps[ps]
    rich.append({"pair": f"{ps}->{2*ps}", "price": ext, "error": ext - truth_A})
    print(f"    scales {ps:3d}+{2*ps:3d}: error={ext-truth_A:+.6f}")
results["sweep3_richardson"] = rich

# ----------------------------------------------------------------------
# Sweep 4: full-config ladder on instance B with resource costs
# ----------------------------------------------------------------------
print("\n== Sweep 4: encoded-config ladder, n=4 realistic instance ==")
gh8_pts, gh8_prob = gauss_hermite_normal_grid(3)
strat8 = stratified_grid(3)


def rematched_or_plain(pts, prob, ss):
    rp = rounded_grid(pts, ss)
    mm = moment_rematched_probs(rp)
    return (pts, mm) if mm is not None else (pts, prob)


ladder = [
    ("repo baseline: binary, ss=1, ps=1", (-1.0, 1.0), (0.5, 0.5), 1, 1),
    ("binary, ss=1, ps=64", (-1.0, 1.0), (0.5, 0.5), 1, 64),
    ("GH4, ss=16, ps=64", gh4_pts, gh4_prob, 16, 64),
    ("GH4+rematch, ss=16, ps=64", *rematched_or_plain(gh4_pts, gh4_prob, 16), 16, 64),
    ("GH8, ss=32, ps=64", gh8_pts, gh8_prob, 32, 64),
    ("GH8, ss=32, ps=256", gh8_pts, gh8_prob, 32, 256),
    ("GH8+rematch, ss=32, ps=256", *rematched_or_plain(gh8_pts, gh8_prob, 32), 32, 256),
    ("strat8, ss=32, ps=256", *strat8, 32, 256),
]
rows = []
for name, pts, prob, ss, ps in ladder:
    spec = make_spec(pts=pts, prob=prob, ss=ss, ps=ps, **B)
    price, _ = encoded_price(spec)
    est = estimate_asian_oracle_resources(spec, "qcv", residual_method="factorized_arithmetic")
    rows.append({"config": name, "price": price, "error": price - truth_B,
                 "qcv_qubits": est.total_qubits, "qcv_lookup_rows": est.lookup_rows})
    print(f"  {name:34s} price={price:9.6f}  error={price-truth_B:+9.6f}  "
          f"[qubits={est.total_qubits}, rows={est.lookup_rows:,}]")
results["sweep4_ladder_n4"] = rows

# ----------------------------------------------------------------------
# Error decomposition for two representative configs on instance B
# ----------------------------------------------------------------------
print("\n== Error decomposition (instance B) ==")
decomp = []
for name, pts, prob, ss, ps in [ladder[0], ladder[3]]:
    quad_price = grid_price_exact(pts=pts, prob=prob, **B)
    rp = rounded_grid(pts, ss)
    shock_rounded = grid_price_exact(pts=rp, prob=prob, **B)
    spec = make_spec(pts=pts, prob=prob, ss=ss, ps=ps, **B)
    enc, _ = encoded_price(spec)
    d = {"config": name,
         "quadrature": quad_price - truth_B,
         "shock_rounding": shock_rounded - quad_price,
         "price_rounding_and_cap": enc - shock_rounded,
         "total": enc - truth_B}
    decomp.append(d)
    print(f"  {name}")
    print(f"    quadrature      {d['quadrature']:+.6f}")
    print(f"    shock rounding  {d['shock_rounding']:+.6f}")
    print(f"    price rounding  {d['price_rounding_and_cap']:+.6f}")
    print(f"    total           {d['total']:+.6f}")
results["decomposition_B"] = decomp

# ----------------------------------------------------------------------
# Sweep 5: raw vs QCV estimation accuracy at fixed oracle budget
# ----------------------------------------------------------------------
print("\n== Sweep 5: raw vs QCV amplitude-estimation accuracy ==")
rows = []
for label, base, pts, prob, ss, ps in [
    ("A (n=2, S0=K=2)", A, gh4_pts, gh4_prob, 64, 256),
    ("B (n=4, S0=K=100)", B, gh4_pts, gh4_prob, 16, 64),
]:
    spec = make_spec(pts=pts, prob=prob, ss=ss, ps=ps, **base)
    model = build_asian_model(spec)
    ref = enumerate_encoded_asian(model)
    disc = math.exp(-base["rate"] * base["maturity"])
    B_raw, p_raw = model.raw_cap_dollars, ref.raw_objective_probability
    budget = 1e-4
    cap = residual_cap_from_bias_budget(spec, budget)
    spec_c = make_spec(pts=pts, prob=prob, ss=ss, ps=ps,
                       residual_payoff_cap=cap, **base)
    model_c = build_asian_model(spec_c)
    ref_c = enumerate_encoded_asian(model_c)
    B_res, p_res = model_c.residual_cap_dollars, ref_c.qcv_objective_probability
    clip_bias = disc * (ref_c.residual_payoff_undiscounted
                        - ref_c.clipped_residual_payoff_undiscounted)
    sd = lambda Bc, p: disc * Bc * math.sqrt(max(p * (1 - p), 0.0))
    row = {"instance": label,
           "raw": {"cap": B_raw, "p": p_raw, "shot_sd": sd(B_raw, p_raw)},
           "qcv_budgeted": {"cap": B_res, "p": p_res, "shot_sd": sd(B_res, p_res),
                             "clipping_bias": clip_bias, "budget": budget},
           "variance_ratio": (sd(B_raw, p_raw) / sd(B_res, p_res)) ** 2,
           "qae_cap_ratio": B_raw / B_res}
    rows.append(row)
    print(f"  {label}: raw cap=${B_raw:.4f} p={p_raw:.5f} sd=${row['raw']['shot_sd']:.4f}  "
          f"qcv cap=${B_res:.4f} p={p_res:.5f} sd=${row['qcv_budgeted']['shot_sd']:.4f}  "
          f"variance ratio {row['variance_ratio']:.0f}x, cap ratio {row['qae_cap_ratio']:.1f}x, "
          f"clip bias ${clip_bias:.2e}")
results["sweep5_qcv_caps"] = rows

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=1))
print(f"\nwrote {OUT.relative_to(ROOT)}")
