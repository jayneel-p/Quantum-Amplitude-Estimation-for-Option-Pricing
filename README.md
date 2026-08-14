# Quantum-Amplitude-Estimation-for-Option-Pricing

Classical MC + Kemna–Vorst; Qiskit European AE (Sta20-style). `pip install -e ".[quantum]"` for Qiskit.

The original finite-grid arithmetic-Asian and geometric-control residual
reference oracles are in `qc_option_pricing.quantum.asian_oracle`. They use
QROM price and payoff maps, so they validate finite-grid semantics but are not
a scalable arithmetic construction.

## Wang--Kan weak-Euler geometric control

The deterministic same-model geometric control for the 256-step Wang--Kan
binary weak-Euler model is implemented in
`qc_option_pricing.classical.heston_weak_euler_geometric`.  It uses a
one-dimensional backward transform recursion in variance and an exponentially
damped payoff inversion suitable for the model's atomic distribution.  It
must not be confused with the separate continuous-Heston geometric benchmark.

Reproduce the exact-enumeration tests, production refinement table, and
same-model restored prices with:

```bash
.venv/bin/python -m unittest tests.test_heston_weak_euler_geometric -v
```

The authoritative output is
`results/v11/wang_kan_exact_control.json`, which records the refinement table,
the independent Monte Carlo comparison, and the source hashes of the modules
that produced it.

The complete reversible arithmetic QCV oracle is in
`qc_option_pricing.quantum.arithmetic_asian_oracle`. For a declared uniform
finite shock model it prepares the shocks, evolves both arithmetic and
geometric paths with directed fixed-point multiplication, computes the
nonnegative residual, encodes it with a uniform threshold comparator, and
uncomputes every work register. Its circuit contains only H, X, CX, and
Toffoli primitives—no QROM or arbitrary payoff rotations.

Reproduce the reference-oracle report with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The generated report is `results/v5/full_asian_oracle_validation.md`. These are
executable reference QROM oracles; the report explains why the present QCV
residual lookup must be replaced by reversible arithmetic before making a
credible 252-fixing resource-advantage claim.

Validate the arithmetic oracle in a clean environment:

```bash
python -m pip install -e ".[quantum]"
python -m unittest tests.test_arithmetic_asian_oracle -v
```

## General blocked-control oracle

The saved one-block production oracle remains in
`qc_option_pricing.quantum.arithmetic_asian_oracle`.  The separate
`qc_option_pricing.quantum.blocked_asian_oracle` module constructs the
single-amplitude residual oracle `H - C_k` for any equal block count `k` that
divides the number of fixing dates.  Its convenience interface accepts the
Black--Scholes contract and encoding parameters directly:

```python
from qc_option_pricing.quantum import (
    build_black_scholes_blocked_model,
    build_blocked_asian_oracle,
    estimate_blocked_asian_resources,
)

model = build_black_scholes_blocked_model(
    n_dates=252,
    block_count=6,
    s0=100.0,
    strike=100.0,
    rate=0.05,
    volatility=0.20,
    maturity=1.0,
    price_scale=16_384,
    multiplier_fraction_bits=30,
    residual_cap_dollars=0.10248069157676092,
)
resources = estimate_blocked_asian_resources(model)
oracle = build_blocked_asian_oracle(model)  # materialises the full A circuit
```

The local frontier runner accepts the same model parameters and a list of
block counts.  This command reproduces the reported `k=1,2,3,4,6,12`
analysis and materialises the circuits through `k=6`:

```bash
.venv/bin/python scripts/v25_general_k_frontier.py \
  --n-dates 252 --block-counts 1 2 3 4 6 12 \
  --transpile-blocks 1 2 3 4 6 \
  --s0 100 --strike 100 --rate 0.05 --volatility 0.20 --maturity 1
```

The machine-readable outputs are `results/general_k_frontier.json` and
`results/general_k_frontier.csv`.  They record the clipping checks, directed
rounding certificates, control-price errors, gate-by-gate circuit checks,
resource counts, source hashes, and limitations.  These are logical resource
results under an amplitude-scale query model, not physical runtime estimates.

The output is `results/v8/arithmetic_asian_oracle_validation.json`. The
two-date circuit is simulated and compared with exhaustive path enumeration;
the 252-date circuit is a complete source-level construction with a
compositional Clifford+T ledger, not a materialized or simulated 252-date
circuit. It therefore does not by itself validate physical runtime or quantum
advantage.

## Accuracy validation and calibration

The arithmetic-oracle validator also tests accuracy against a predeclared
$0.01 absolute-price target for one stated daily Black--Scholes benchmark
(`S0=K=100`, `r=0.05`, `sigma=0.20`, `T=1`, 252 fixings). It keeps the three
questions separate:

1. **Finite-oracle correctness.** Small circuits are checked by exact path
   enumeration and statevector/MPS simulation, including clean uncomputation.
2. **Continuous benchmark price.** The geometric control is calculated from
   an independent closed-form expression and the arithmetic price is estimated
   with a seeded, one-million-path, unit-control Monte Carlo calculation. The
   report records both standard errors.
3. **Encoding bias.** Common binary-shock paths compare the encoded directed
   fixed-point payoff with the exact payoff on those same paths. This separates
   shock discretization from multiplication rounding and residual clipping.

The original 252-date encoding (`price_scale=1024`, 18 multiplier fraction
bits) **fails** this benchmark: its estimated price error is `$0.05027 ±
$0.00102` (one standard error). The binary shock grid is not the dominant
problem; its isolated error is `$0.00258 ± $0.00086`.

For this one benchmark, the report identifies an accuracy-qualified refinement:
`price_scale=16384` and 30 multiplier fraction bits. Its estimated error is
`$0.00377 ± $0.00086`; the two-sided 95% bound remains below `$0.01`. It
requires approximately 29,319 logical qubits and `2.10e8` T gates per state
preparation `A` under the report's standard 7-T Toffoli accounting. The
refinement restores its finite geometric-control scalar using a paired
classical calibration, so that calibration and its recorded uncertainty must
be rerun whenever the contract, shock model, or precision changes.

This is a supported result for the stated benchmark—not a uniform error bound,
a full 252-date circuit simulation, a hardware-runtime estimate, or evidence
of quantum advantage. The JSON report is the authoritative record of inputs,
seeds, confidence calculations, evidence states, and limitations.
