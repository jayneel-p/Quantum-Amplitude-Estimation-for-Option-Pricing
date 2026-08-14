# Finite-grid Asian option oracle audit

**Audit date:** 2026-07-15  
**Implementation:** `src/qc_option_pricing/quantum/asian_oracle.py`  
**Reproducible circuit report:** `results/v6/full_asian_oracle_validation.md`

## Result

We built and simulated the two-fixing raw and QCV circuits. Their objective probabilities agree with independent enumeration within `1.4e-11`. The measured dirty-work probability is zero. After `A` followed by `A^-1`, the largest measured nonzero probability is below `8e-13`. The cap tests cover no clipping, partial clipping, and saturation. The integer AM-GM proof shows that the implemented QCV residual is nonnegative on every reachable path.

For 252 fixings, we generated the integer domains and counted lookup rows for one shock bit per date and whole-dollar prices. We did not build, transpile, synthesize, or simulate that circuit. The QCV count is 23,883,377,236 lookup rows before gate synthesis, including an 11,941,558,586-row residual table applied once forward and once in reverse. We did not bound the difference between this finite model and the continuous GBM price.

The tests establish the following:

| Claim | Status | Evidence |
|---|---|---|
| Exact semantics for the encoded finite model | **PASS** | Independent path enumeration and circuit simulation |
| Raw objective amplitude | **PASS** | `0.740000000006549` vs exact `0.74` |
| QCV residual amplitude | **PASS** | `0.440000000013873` vs exact `0.44` |
| Clean reversible A-operator | **PASS** | Zero measured dirty-work probability; round-trip leakage below `8e-13` |
| Cap and residual-cap semantics | **PASS** | No-clip, partial, and saturating cases |
| QCV nonnegativity and pathwise decomposition | **PASS** | Proof plus exhaustive/randomized stress tests |
| 252-date register sizing and structural count | **PASS** | Deterministic table construction and overflow tests |
| Executable 252-date circuit | **Not run** | Only integer-domain and lookup-row counts were generated |
| Accuracy for the continuous 252-dimensional GBM target | **Not measured** | No refinement study jointly changes shock bits and price precision |
| Clifford+T / physical resources | **Not counted** | State preparation, exponentials, rotations, reflections, and QAE were not synthesized |
| Runtime comparison | **Not made** | No physical quantum runtime was computed |

## What was reviewed

The audit traced every forward and inverse operation in the implementation, ran independent differential tests, checked all cap regimes, checked the 252-date integer ranges and resource formulas, and reviewed the full text of the Asian-specific quantum pricing papers located by title/keyword search and citation chaining through 2026-07-15. “End-to-end oracle” is used here for a construction that addresses stochastic-state preparation, path/payoff computation, amplitude encoding, and an expectation-estimation interface. PDE algorithms are listed separately because they do not implement that Monte Carlo A-operator.

This is a bounded, reproducible literature claim: it covers the published and publicly accessible Asian-specific quantum implementations found by the search, plus the main circuit-level option-pricing comparators. It cannot prove that no unpublished, unindexed, or inaccessible manuscript exists.

The Pracht paper's bibliographic record was verified on SSRN; because SSRN's automated PDF endpoint returned an access-challenge page, its full manuscript was checked through a publicly indexed text copy and cross-checked against the official title, authors, date, page count and DOI.

## Implementation validation

### Encoded stochastic model

For fixing `i`, the implementation prepares an independent finite shock register and encodes the integer innovation

`z_i = round(shock_scale * shock_point_i)`.

It accumulates the signed prefix `Z_i = sum_{j<=i} z_j`, then looks up

`s_i = ceil(price_scale * S0 * exp((r-sigma^2/2)t_i + sigma*sqrt(dt)*Z_i/shock_scale))`.

The arithmetic payoff numerator is

`P_A = max(sum_i s_i - N*K_integer, 0)`.

The controlled rotation uses angle `2 asin(sqrt(min(P_A/B, 1)))`, so the objective-one probability is exactly `min(P_A/B, 1)` apart from floating-point gate/simulator error. All prefix, price, total, and addend registers are then uncomputed.

This is materially better than loading a precomputed amplitude for every full path: the stochastic input is a tensor product of one small state preparation per fixing. However, the exponential is still represented by QROM over every integer prefix value, so it is a reference implementation rather than an arithmetic exponential circuit.

### QCV path identity

The discrete geometric value uses

`g = floor(price_scale * G)`,

where the weighted shock sum implements the exact identity

`(1/N) sum_i log(S_i) = log(S0) + (r-sigma^2/2)dt(N+1)/2 + sigma*sqrt(dt)/(N*shock_scale) * sum_j (N-j+1)z_j`.

For positive path prices, AM-GM and directed rounding give

`sum_i s_i >= price_scale * sum_i S_i >= N*price_scale*G >= N*g`.

Because `x -> max(x-NK,0)` is monotone,

`P_R = max(sum_i s_i-NK,0) - N*max(g-K,0) >= 0`.

Thus the residual is an unsigned integer on every reachable path and, when its cap does not bind,

`arithmetic call = geometric call + residual`

holds path by path and in expectation. This proof is stronger than a numerical observation and is correctly reflected by the circuit.

### Reversible-circuit results

For the two-fixing binary reference (`P(-1)=0.4`, `P(+1)=0.6`, `S0=2`, `K=1`, `sigma=0.3`):

| Quantity | Independent exact result | Circuit result | Absolute error |
|---|---:|---:|---:|
| Raw objective probability | 0.74 | 0.740000000006549 | 6.55e-12 |
| QCV objective probability | 0.44 | 0.440000000013873 | 1.39e-11 |
| Arithmetic payoff | 1.48 | 1.48 after decoding | numerical precision |
| Geometric control | 0.60 | 0.60 | numerical precision |
| Residual payoff | 0.88 | 0.88 | numerical precision |

The raw circuit uses 15 qubits, 48 high-level QROM rows, 1,008 decomposed CX gates, and depth 1,585. The QCV circuit uses 24 qubits, 108 actual high-level QROM rows, 8,516 CX gates, and depth 12,950. These are executable-reference counts, not fault-tolerant resource counts.

The separate high-width reference harness also completed **23/23 checks**. On a four-node Gauss-Hermite, two-fixing grid, the 27-qubit raw circuit produced `0.080946404516` vs exact `0.080946404281`; the 43-qubit QCV circuit produced `0.039779310213` vs exact `0.039779310231`. With a genuinely partial `$0.75` residual cap, the 43-qubit QCV circuit produced `0.409722222201` vs exact `0.409722222222`. Both QCV cases had zero measured dirty-work probability; their MPS simulations took 1,761 and 1,507 seconds respectively.

The test suite covers:

- independent formula vs compiled tables;
- raw and QCV objective amplitudes;
- work-register cleanup;
- `A A^-1` round trip;
- explicit raw cap;
- independent residual cap;
- no-clip, partial-clip, and saturating residual regimes;
- exact minimal cap selected from a clipping-bias budget;
- Gauss-Hermite normal moments;
- 250 randomized AM-GM/decomposition trials;
- invalid/non-finite input rejection;
- estimator-to-built-circuit structural agreement; and
- 252-date signed/unsigned range containment and locked resource totals.

### Defects found and corrected

1. The specification accepted `NaN`/infinite model scalars and non-integer fixing/scaling values. These are now rejected before table construction.
2. The independent test reference previously reused the raw cap incorrectly when checking a separately configured residual cap. It now follows the implementation's two-cap semantics.
3. A purported “partial” residual-cap test used a cap of `$0.75` on support `{1,2,3}` encoded units, where the cap numerator was `3` and therefore clipped nothing. The cap is now `$0.50` (numerator `2`), which genuinely leaves one support value below the cap and one above it.
4. Saturating probabilities numerically within about `1e-13` of one were classified as interior by a strict floating-point comparison. The validator now uses a `1e-9` boundary tolerance.

## The 252-fixing result

The structural benchmark uses 252 independent one-qubit shocks `(-1,+1)`, `shock_scale=1`, `price_scale=1`, `S0=K=100`, `r=5%`, `sigma=20%`, and `T=1`.

| Route | Logical qubits before synthesis | High-level lookup rows |
|---|---:|---:|
| Raw | 307 | 258,048 |
| QCV | 357 | 23,883,377,236 (upper bound) |

The QCV residual rectangle alone has 11,941,558,586 `(arithmetic total, weighted shock)` entries and must be computed and uncomputed. The modest qubit count therefore does not imply feasibility: circuit depth and non-Clifford synthesis cost are the blockers.

The benchmark also has intentionally weak numerical resolution. One shock bit per day and whole-dollar prices do not establish a useful continuous-normal or market-pricing approximation. Increasing shock and price precision expands the integer domains and QROMs sharply. Building the Python table is not equivalent to constructing an efficient quantum exponential or payoff oracle.

## Asian-specific literature: full-text assessment

### Monte Carlo / amplitude-oracle constructions

| Work | What is implemented or specified | Scale actually evaluated | Validation value | Limitation relevant here |
|---|---|---|---|---|
| [Rebentrost, Gupt & Bromley (2018)](https://doi.org/10.1103/PhysRevA.98.022321) | Independent Gaussian increment states, sequential BSM path arithmetic, average/payoff register, amplitude rotation and AE interface | Asymptotic construction; numerical experiment injects a known European price into single-qubit phase estimation | Establishes the canonical path-oracle decomposition | No compiled Asian circuit, Asian price experiment, or elementary gate/resource count; efficient Gaussian loading/arithmetic are assumptions |
| [Stamatopoulos et al. (2020)](https://quantum-journal.org/papers/q-2020-07-06-291/) | Multivariate lognormal price-state loader, reversible weighted sum, comparator and sine-linearized payoff amplitude | Asian method illustrated at two time points; hardware experiment is European only | Specifies the weighted-sum, comparison, and payoff-loading components used in later work | Joint distribution loading is not costed at daily path dimension; the Asian circuit is not run with AE or on hardware |
| [Pracht et al. (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4137397) | Full EUR/PLN joint path table, generic state preparation, QFT/Draper averaging, comparator, payoff encoding and QAE | 4 dates, 3 qubits/date, `8^4=4096` paths | A genuinely integrated small Asian circuit and numerical comparison | State preparation enumerates all paths; practical contracts use roughly hundreds of observations; QAE errors become very large for deep-OTM strikes; the claimed local-volatility extensibility is not demonstrated by a scalable loader |
| [Wolf, Horsky & Koppe (2022)](https://www.mdpi.com/2227-9091/10/12/221) | Path-probability state, amplitude arithmetic payoff and indicator for a floating-strike Asian on a valuation tree | Simulations up to about 5 time steps | End-to-end module composition and explicit small-circuit scaling | Uniformly controlled rotations/state loading grow exponentially with path bits; AE was not run because the simulated circuit was already too deep |
| [Wang & Kan (2024)](https://quantum-journal.org/papers/q-2024-10-23-1504/) | Fault-tolerant Heston path simulation (`U1`), Asian payoff (`U2`), amplitude loading (`U3`), QAE iterate and T-resource accounting | Asian instances with 256 steps; barriers with 1024 | The weak-Euler Asian table gives `Tcount(Q)=3.2e7`, `N_oracle=7363`, `2.4e11` total T gates, `1.2e11` T-depth, and about `2.2e4` logical qubits | The target is the numerical scheme, not a proved continuous-Heston price; payoff normalization `Z` is estimated from classical samples and larger payoffs are ignored, but that clipping loss is absent from the displayed aggregate error budget; fixed-point error is empirical |
| [Prakash et al. (2024)](https://arxiv.org/abs/2402.10132) | Karhunen-Loève/semi-digital Brownian construction, exponentiation, time-domain subsampling, payoff rotation and nested AE | Theorems and pseudocode, no compiled instance | Removes polynomial dependence on monitoring count; best stated method is polylogarithmic in `T` and about `O~(epsilon^-3)` | Not an executable resource estimate; Gaussian truncation, `Gmax` normalization, arithmetic precision, and constants are not jointly instantiated; worse epsilon scaling than ordinary QAE is traded for improved `T` dependence |

None of the papers reviewed here reports all five items together: a daily Asian instance, a compiled reversible `A` operator, a bound to the continuous-model price, a clipping-loss bound, and physical resources. Wang–Kan reports the most detailed Asian T-resource table. Pracht and Wolf build small path-dependent circuits. Prakash analyzes how the monitoring-date dependence can be reduced without compiling an instance.

### Asian PDE approaches (not amplitude oracles)

| Work | Method | What was validated | Why it does not validate this A-operator |
|---|---|---|---|
| [Fontanela, Jacquier & Oumgari (2021)](https://doi.org/10.1137/21M1397878) | Variational imaginary-time evolution of the reduced Asian pricing PDE | 4-qubit, 16-point spatial grid; 25 parameterized `Ry` gates; 500 classical time steps; comparison with a classical PDE solution | No stochastic path state, payoff-amplitude oracle, or AE. The authors explicitly assume small initial ansatz error does not propagate strongly and leave scalable ansatz selection/optimization open |
| [Rendon, Kshirsagar & Tran (2025)](https://arxiv.org/abs/2501.15614) | Preconditioned linear-system/QSVT solution of an Asian PDE plus probability-integral extraction | Analytical discretization, block encodings and asymptotic complexity | No executed end-to-end circuit or concrete gate table. The polylogarithmic epsilon claim relies on conditioning, smoothness/interpolation, block-encoding, inversion, and state-extraction assumptions with substantial hidden constants |

### Other end-to-end option-oracle comparators

The following are important for standards of evidence but do not implement the same Asian contract:

- [Chakrabarti et al. (2021)](https://quantum-journal.org/papers/q-2021-06-01-463/) gives explicit truncation/discretization/AE budgeting and fault-tolerant resources for an autocallable and a TARF. Its roughly `1e10`-T results cannot be linearly transferred to 252-date Asians, but its accounting discipline is the right benchmark.
- [Kaneko et al. (2022)](https://doi.org/10.1140/epjqt/s40507-022-00125-2) constructs local-volatility time evolution and payoff-oracle machinery. It informs path preparation, not this GBM QROM's validation.
- [Stamatopoulos & Zeng (2024)](https://quantum-journal.org/papers/q-2024-04-30-1322/) uses quantum signal processing for payoff loading. It can reduce payoff-loading cost but does not remove Asian path simulation or exponential evaluation.
- [Cibrario et al. (2024)](https://arxiv.org/abs/2402.05574) demonstrates an end-to-end IQAE rainbow-option circuit and compares exponential amplitude loaders. It is path-independent and small-scale, but is relevant to replacing QROM exponentials.
- [Manzano et al. (2025)](https://doi.org/10.1140/epjqt/s40507-025-00328-3) implements direct signed-payoff encoding with modified real QAE, including a small cliquet example. Its query scaling assumes the path/payoff oracle and does not make that oracle cheap.
- The 2026 “end-to-end” multidimensional integration and PDE papers found in the search address path-independent multi-asset integration or European PDEs, not an arithmetic-Asian amplitude oracle, so they were not used as direct validation precedents.

## Comparison with the repository code

The repository implementation does three things that the smallest table-loaded demonstrations do not do together:

1. It prepares independent innovations instead of a single exponentially large path-amplitude vector.
2. It provides an executable reversible reference with explicit uncomputation and objective decoding.
3. Its directed-rounding QCV construction has an exact nonnegative residual and pathwise control identity.

Price exponentials remain QROMs, and the QCV residual is a two-dimensional rectangular QROM. At 252 dates, the counted residual table has 11,941,558,586 rows and is applied twice. We did not replace either table with reversible arithmetic.

## Work not done

The audit did not do the following:

1. No market data or model calibration was used. The tests use the parameters written in the circuit specification.
2. The prefix-to-price QROM was not replaced by a synthesized fixed-point exponential. The QCV residual table was not replaced by comparator and subtraction circuits.
3. No single price tolerance was allocated across shock quadrature, shock rounding, price rounding, clipping, control-price error, AE error, rotation approximation, and gate synthesis.
4. No refinement run jointly varied shock bits, price precision, and monitoring dates against a continuous-model reference.
5. `A`, `A^-1`, the reflections, and an AE schedule were not compiled to Clifford+T or converted to physical qubits and runtime.
6. No total-runtime comparison with an optimized classical implementation was made.

The two-date circuits implement the stated finite-grid probabilities and pass the listed reversibility tests. The 252-date result is a register and lookup-row count for a coarse finite model. It is not an executed circuit, a continuous-model accuracy result, or a runtime result.
