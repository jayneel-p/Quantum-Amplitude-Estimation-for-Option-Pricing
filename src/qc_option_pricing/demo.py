"""Console demo: classical Monte Carlo vs Black–Scholes and Asian MC."""

from __future__ import annotations

import sys
from pathlib import Path

# PyCharm / "Run file" puts this directory on sys.path, not `src`, so the package name
# is not resolvable until we add the parent of this folder (`src`) explicitly.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qc_option_pricing.classical import arithmetic_asian_call_mc, convergence_curve, european_call


def main() -> None:
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    analytic = european_call(s0, k, r, sigma, t)
    ns = [500, 2000, 8000, 32000]
    _, estimates, stderrs, _ = convergence_curve(s0, k, r, sigma, t, ns)
    print("European call — Black–Scholes:", f"{analytic:.4f}")
    print("Paths | MC estimate | stderr")
    for n, y, se in zip(ns, estimates, stderrs):
        print(f"{n:6d} | {y:11.4f} | {se:.4f}")

    asian, asian_se = arithmetic_asian_call_mc(s0, k, r, sigma, t, n_steps=252, n_paths=50_000)
    print()
    print("Arithmetic Asian (daily steps, same params), MC:", f"{asian:.4f} ± {asian_se:.4f}")


if __name__ == "__main__":
    main()
