# Matched clipping-bias normalisations for the daily arithmetic Asian call,
# selected and evaluated on independent samples, with a citable JSON artifact.
#
# Supersedes scripts/v5_matched_bias_cutoff_ratio.py, which computed the same
# ratios but (i) printed to stdout only, (ii) used UNDISCOUNTED losses, and
# (iii) selected and evaluated the cutoffs on the same paths.
#
# For each clipping-bias budget b, this script finds the smallest
# normalisation B such that the DISCOUNTED expected truncation loss
#
#     exp(-r*T) * E[(X - B)^+]  <=  b
#
# on a seeded selection sample, separately for
#     X = (A_N - K)^+                       (raw arithmetic payoff), and
#     X = (A_N - K)^+ - (G_N - K)^+          (one-block Kemna--Vorst residual),
# and reports B_A, B_R, and B_A/B_R.  The realised discounted loss at each
# selected B is then recomputed on an independent evaluation sample with a
# different seed, so the selection bias of the in-sample convention is visible.
#
# The artifact also records two single numbers quoted in Section 6 at the base
# case: the cutoff ratio under the paper's original matched-exceedance rule
# (fraction 1.35e-3 of paths strictly above the cutoff on both arms; expected
# to reproduce 16.0) and the residual variance ratio at beta = 1 (expected
# near 522; invariant to discounting).
#
# Writes results/v19/matched_bias_normalisation.json.

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Model: daily arithmetic Asian call under continuous Black-Scholes, sampled
# exactly at the N daily fixings t_i = i*T/N, i = 1..N (same convention as
# scripts/v5_matched_bias_cutoff_ratio.py: S0 is not part of the average).
S0, K, RATE, SIGMA, T, N_FIX = 100.0, 100.0, 0.05, 0.20, 1.0, 252
DISC = math.exp(-RATE * T)

BUDGETS = [3e-3, 1e-3, 3e-4, 1e-4]
EXCEEDANCE_FRACTION = 1.35e-3  # the paper's original matched-exceedance rule


def simulate(seed, n_paths, chunk):
    """Return (payA, resid) at the daily fixings for n_paths GBM paths.

    payA  = (A_N - K)^+  with A_N the arithmetic mean of S at the fixings.
    resid = (A_N - K)^+ - (G_N - K)^+  with G_N the geometric mean.
    Both are undiscounted per-path payoffs in float64.
    """
    rng = np.random.default_rng(seed)
    dt = T / N_FIX
    drift = (RATE - 0.5 * SIGMA**2) * dt
    vol = SIGMA * math.sqrt(dt)
    payA_parts, resid_parts = [], []
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        z = rng.standard_normal((m, N_FIX))
        logs = math.log(S0) + np.cumsum(drift + vol * z, axis=1)
        A = np.exp(logs).mean(axis=1)
        G = np.exp(logs.mean(axis=1))
        payA = np.maximum(A - K, 0.0)
        payG = np.maximum(G - K, 0.0)
        payA_parts.append(payA)
        resid_parts.append(payA - payG)
        done += m
    return np.concatenate(payA_parts), np.concatenate(resid_parts)


def discounted_loss(x, B):
    """exp(-r*T) * mean((x - B)^+): the paper's clipping-bias convention."""
    exceed = x[x > B]
    return DISC * float((exceed - B).sum()) / x.size


def loss_with_se(x, B):
    """Discounted loss, its standard error, and the exceedance count at B."""
    per_path = DISC * np.maximum(x - B, 0.0)
    loss = float(per_path.mean())
    se = float(per_path.std(ddof=1)) / math.sqrt(x.size)
    return loss, se, int((x > B).sum())


def find_B(x, budget):
    """Smallest B with discounted_loss(x, B) <= budget, by 80-step bisection.

    The discounted loss is continuous and strictly decreasing in B wherever it
    is positive, so bisection on [0, max(x)+1] brackets the root; hi always
    satisfies the budget, and after 80 halvings it equals the infimum to
    double precision.
    """
    lo, hi = 0.0, float(x.max()) + 1.0
    if discounted_loss(x, lo) <= budget:
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if discounted_loss(x, mid) > budget:
            lo = mid
        else:
            hi = mid
    return hi


def matched_exceedance_B(x, fraction):
    """B such that exactly round(fraction * n) paths lie strictly above it.

    B is the k-th largest sample value (k = round(fraction * n)); with a
    continuous payoff distribution the values are distinct, so exactly k
    paths exceed it.  This is the paper's original cutoff rule.
    """
    n = x.size
    k = int(round(fraction * n))
    return float(np.partition(x, n - 1 - k)[n - 1 - k]), k


def arm_record(B, sel, ev):
    """In- and out-of-sample discounted losses at B for one arm."""
    loss_in, se_in, n_in = loss_with_se(sel, B)
    loss_out, se_out, n_out = loss_with_se(ev, B)
    return {
        "B": B,
        "loss_in_sample": loss_in,
        "loss_in_sample_se": se_in,
        "n_selection_paths_above_B": n_in,
        "loss_out_of_sample": loss_out,
        "loss_out_of_sample_se": se_out,
        "n_evaluation_paths_above_B": n_out,
    }


def paired_difference(ev_payA, ev_resid, B_A, B_R):
    """Out-of-sample raw-minus-residual loss difference with a paired SE.

    Both arms are evaluated on the same paths, so the SE of the difference
    comes from the per-path differences, not from adding the two arm SEs.
    """
    d = DISC * (np.maximum(ev_payA - B_A, 0.0) - np.maximum(ev_resid - B_R, 0.0))
    return float(d.mean()), float(d.std(ddof=1)) / math.sqrt(d.size)


def git_info(repo_root):
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True
        )
    rev = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "rev": rev.stdout.strip() if rev.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-select", type=int, default=2_000_000,
                        help="paths in the selection sample (default 2e6)")
    parser.add_argument("--n-eval", type=int, default=2_000_000,
                        help="paths in the independent evaluation sample (default 2e6)")
    parser.add_argument("--seed-select", type=int, default=2026_0713,
                        help="selection-sample seed (default 20260713, the v5 seed)")
    parser.add_argument("--seed-eval", type=int, default=2026_0727,
                        help="evaluation-sample seed (default 20260727)")
    parser.add_argument("--chunk", type=int, default=250_000,
                        help="paths per simulation chunk (default 250000)")
    parser.add_argument("--out", type=Path,
                        default=repo_root / "results" / "v19" / "matched_bias_normalisation.json")
    args = parser.parse_args()
    assert args.seed_select != args.seed_eval, "evaluation sample must be independent"

    created_at_start = datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.time()

    sel_payA, sel_resid = simulate(args.seed_select, args.n_select, args.chunk)
    ev_payA, ev_resid = simulate(args.seed_eval, args.n_eval, args.chunk)

    # --- matched-bias budgets: select on the selection sample, evaluate on both
    budget_rows = []
    for b in BUDGETS:
        B_A = find_B(sel_payA, b)
        B_R = find_B(sel_resid, b)
        raw = arm_record(B_A, sel_payA, ev_payA)
        res = arm_record(B_R, sel_resid, ev_resid)
        # tolerance covers summation-order noise: the bisection criterion uses
        # sum()/n while the recorded loss uses np.mean over the same values
        tol = b * (1.0 + 1e-12)
        assert raw["loss_in_sample"] <= tol and res["loss_in_sample"] <= tol
        diff, diff_se = paired_difference(ev_payA, ev_resid, B_A, B_R)
        budget_rows.append({
            "budget": b,
            "raw_payoff": raw,
            "residual": res,
            "ratio_B_raw_over_B_residual": B_A / B_R,
            "out_of_sample_loss_difference": {
                "raw_minus_residual": diff,
                "paired_se": diff_se,
            },
        })

    # B must grow as the budget shrinks (BUDGETS is ordered decreasing).
    for arm in ("raw_payoff", "residual"):
        Bs = [row[arm]["B"] for row in budget_rows]
        assert all(b1 < b2 for b1, b2 in zip(Bs, Bs[1:])), \
            f"{arm} normalisations not monotone in the budget: {Bs}"

    # --- the paper's original matched-exceedance rule, on the selection sample
    B_A_q, k = matched_exceedance_B(sel_payA, EXCEEDANCE_FRACTION)
    B_R_q, _ = matched_exceedance_B(sel_resid, EXCEEDANCE_FRACTION)
    matched_exceedance = {
        "rule": "cutoff set so the fraction of selection paths strictly above it "
                "is 1.35e-3 on both arms (the paper's original convention; "
                "selection sample = the v5 seed, so this reproduces the quoted 16.0)",
        "exceedance_fraction": EXCEEDANCE_FRACTION,
        "target_paths_above": k,
        "raw_payoff": arm_record(B_A_q, sel_payA, ev_payA),
        "residual": arm_record(B_R_q, sel_resid, ev_resid),
        "ratio_B_raw_over_B_residual": B_A_q / B_R_q,
    }

    # --- residual variance ratio at beta = 1 (invariant to discounting)
    variance_ratio = {
        "definition": "Var[(A_N-K)^+] / Var[(A_N-K)^+ - (G_N-K)^+], ddof=1; "
                      "the discount factor cancels",
        "selection_sample": float(sel_payA.var(ddof=1) / sel_resid.var(ddof=1)),
        "evaluation_sample": float(ev_payA.var(ddof=1) / ev_resid.var(ddof=1)),
    }

    runtime = time.time() - t0
    script_path = Path(__file__).resolve()
    artifact = {
        "schema_version": "matched-bias-normalisation-v1",
        "created_at_start": created_at_start,
        "command": "scripts/matched_bias_normalisation.py",
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "supersedes": "scripts/v5_matched_bias_cutoff_ratio.py "
                      "(stdout only; undiscounted losses; selected and evaluated "
                      "on the same paths)",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "git": git_info(repo_root),
        "model": {
            "instrument": "daily arithmetic-average Asian call, continuous "
                          "Black-Scholes sampled exactly at the daily fixings "
                          "t_i = i*T/N, i = 1..N (S0 not in the average)",
            "s0": S0, "strike": K, "rate": RATE, "sigma": SIGMA,
            "maturity": T, "n_fixings": N_FIX,
            "raw_payoff": "(A_N - K)^+, A_N the arithmetic mean at the fixings",
            "residual": "(A_N - K)^+ - (G_N - K)^+, G_N the geometric mean; "
                        "nonnegative path by path",
        },
        "convention": {
            "clipping_bias": "exp(-r*T) * E[(X - B)^+], the DISCOUNTED expected "
                             "truncation loss, matching the paper's definition; "
                             "the superseded v3/v5 scripts omitted exp(-r*T)",
            "discount_factor": DISC,
            "selection": "B = smallest value with discounted loss <= budget on "
                         "the selection sample, by 80-step bisection",
            "evaluation": "realised discounted loss at the selected B recomputed "
                          "on an independent evaluation sample (different seed)",
        },
        "sampling": {
            "rng": "numpy.random.default_rng (PCG64), standard_normal",
            "n_paths_selection": args.n_select,
            "seed_selection": args.seed_select,
            "n_paths_evaluation": args.n_eval,
            "seed_evaluation": args.seed_eval,
            "chunk_size": args.chunk,
        },
        "budgets": budget_rows,
        "matched_exceedance_reproduction": matched_exceedance,
        "variance_ratio_beta_one": variance_ratio,
        "source_hashes": {
            "scripts/matched_bias_normalisation.py":
                hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
        "runtime_seconds": runtime,
        "created_at_end": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(artifact, fh, indent=1)
        fh.write("\n")

    # human-readable summary
    print(f"{'budget':>8} {'B_A':>8} {'B_R':>7} {'ratio':>6} "
          f"{'loss_A in':>10} {'loss_A out':>11} {'loss_R in':>10} {'loss_R out':>11} "
          f"{'nA>':>6} {'nR>':>6}")
    for row in budget_rows:
        a, r = row["raw_payoff"], row["residual"]
        print(f"{row['budget']:>8.0e} {a['B']:>8.2f} {r['B']:>7.3f} "
              f"{row['ratio_B_raw_over_B_residual']:>6.1f} "
              f"{a['loss_in_sample']:>10.3e} {a['loss_out_of_sample']:>11.3e} "
              f"{r['loss_in_sample']:>10.3e} {r['loss_out_of_sample']:>11.3e} "
              f"{a['n_evaluation_paths_above_B']:>6} {r['n_evaluation_paths_above_B']:>6}")
    me = matched_exceedance
    print(f"\nmatched exceedance 1.35e-3: B_A={me['raw_payoff']['B']:.2f} "
          f"B_R={me['residual']['B']:.3f} ratio={me['ratio_B_raw_over_B_residual']:.1f}")
    print(f"variance ratio beta=1: selection={variance_ratio['selection_sample']:.1f} "
          f"evaluation={variance_ratio['evaluation_sample']:.1f}")
    print(f"runtime {runtime:.1f} s; artifact -> {args.out}")


if __name__ == "__main__":
    main()
