#!/usr/bin/env python3
"""v3: European statevector error decomposition, now with a truncation-window
sweep so the manuscript's truncation claim is backed by data.

(a) error vs rescaling parameter c at fixed n (n=5, n=7 overlap -> discretization
    is not dominant);
(b) error vs n at c=0.10, w=3 sigma (plateau ~ $0.009);
(c) error vs truncation half-width w in {2, 2.5, 3, 3.5, 4} sigma at n=7, c=0.10.

Outputs: results/v3/error_decomposition.png, results/v3/error_decomposition.txt
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
from qiskit.quantum_info import Statevector

from qc_option_pricing.classical import european_call
from qc_option_pricing.quantum.european_ae import build_european_call_circuit

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
OUT = _REPO / "results" / "v3"
OUT.mkdir(parents=True, exist_ok=True)


def encoded_error(n: int, c: float, bs: float, w: float = 3.0) -> float:
    qc, obj, post = build_european_call_circuit(S0, K, R, SIGMA, T, n,
                                                c_approx=c, n_stddevs=w)
    sv = Statevector(qc)
    price = math.exp(-R * T) * post(sv.probabilities([obj])[1])
    return price - bs


def main() -> int:
    bs = european_call(S0, K, R, SIGMA, T)
    lines = [f"Black-Scholes price: {bs:.6f}", ""]

    cs = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25]
    err_c = {n: [encoded_error(n, c, bs) for c in cs] for n in (5, 7)}
    for n in (5, 7):
        lines.append(f"(a) n={n}: " + ", ".join(
            f"c={c:.3f}->{e:+.4f}" for c, e in zip(cs, err_c[n])))

    ns = [3, 4, 5, 6, 7, 8]
    err_n = [encoded_error(n, 0.10, bs) for n in ns]
    lines.append("(b) c=0.10, w=3: " + ", ".join(
        f"n={n}->{e:+.6f}" for n, e in zip(ns, err_n)))

    ws = [2.0, 2.5, 3.0, 3.5, 4.0]
    err_w = [encoded_error(7, 0.10, bs, w=w) for w in ws]
    lines.append("(c) n=7, c=0.10: " + ", ".join(
        f"w={w:.1f}->{e:+.6f}" for w, e in zip(ws, err_w)))

    (OUT / "error_decomposition.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14.5, 4.0))
    for n, color, style in ((5, "#2E75B6", "s--"), (7, "#c0392b", "o-")):
        ax0.plot(cs, err_c[n], style, color=color, label=f"$n={n}$", ms=5)
    ax0.axhline(0, color="grey", lw=0.7)
    ax0.axvline(0.10, color="grey", ls=":", lw=0.8)
    ax0.set_xlabel("rescaling parameter $c$")
    ax0.set_ylabel("pricing error (\\$)")
    ax0.set_title("(a) error vs $c$  ($w=3\\sigma$)")
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    ax1.semilogy(ns, [abs(e) for e in err_n], "s-", color="#1f3864", ms=5)
    ax1.axhline(0.009, color="grey", ls="--", lw=0.8, label="plateau $\\approx\\$0.009$")
    ax1.set_xlabel("distribution qubits $n$")
    ax1.set_ylabel("|pricing error| (\\$)")
    ax1.set_title("(b) error vs $n$  ($c=0.10$, $w=3\\sigma$)")
    ax1.legend(fontsize=9); ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(ws, err_w, "o-", color="#16a34a", ms=5)
    ax2.axhline(0, color="grey", lw=0.7)
    ax2.set_xlabel("truncation half-width $w$ ($\\times\\sigma$)")
    ax2.set_ylabel("pricing error (\\$)")
    ax2.set_title("(c) error vs truncation  ($n=7$, $c=0.10$)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("European-call statevector error decomposition", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "error_decomposition.png", dpi=200, bbox_inches="tight")
    print(f"-> wrote {OUT}/error_decomposition.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
