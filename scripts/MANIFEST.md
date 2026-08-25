# Script manifest

Every top-level script here either produces a number or figure used by the manuscript or is the
exact historical generator named in a released provenance record. Superseded scripts without a
published provenance dependency are in `_retired/`, and the Heston follow-up's scripts moved to
`heston-track/scripts/`.

Top-level publication scripts are tracked in the public repository. Retire rather than delete so
released result hashes remain traceable.

## Figures the paper includes

| Script | Figure |
|---|---|
| `generate_classical_results.py` | `asian_methods_comparison.png`, plus `asian_table.txt` and `geo_crosscheck.txt` |
| `generate_gbm_paths_figure.py` | `gbm_sample_paths.png` |
| `reproduce_stamatopoulos_fig3.py` | `stamatopoulos_fig3_reproduction.png` |
| `generate_vr_heatmap_figure.py` | `vr_heatmap.png`, `vr_heatmap_data.json` |
| `v3_rqmc.py` | `rqmc_convergence.png`, `v3/rqmc_replicates.json` |
| `v3_error_decomposition.py` | `error_decomposition.png`, `v3/error_decomposition.txt` |
| `v3_iae_scaling.py` | `iae_scaling.png`, `v3/iae_scaling.json` |
| `v4_qcv_extensions.py` | `qcv_ladder.png`, `v4/qcv_extensions.txt`, and the basket check. Every cutoff is a `np.quantile` at a matched exceedance fraction of `norm.cdf(-3)` = 1.3499e-3, **not** a matched dollar bias budget, and the clipped variable is `A - B_k` rather than the residual payoff. Its k=1 ratio is 15.9; the 20.5 of `v19/` is the budget-matched rule and the two must not be mixed |
| `figure_range_vs_variance.py` | `range_vs_variance.png`, `range_vs_variance.json` |
| `figure_principle_diagnostics.py` | `principle_beta_sweep.png`, `principle_grid_scatter.png`, CSV source tables, and `principle_diagnostics.json` — exactly compares the probability-weighted variance- and finite-grid range-optimal coefficients on the 12-date four-point encoding grid, then joins the existing 25-cell variance and matched-exceedance range experiments at $\beta=1$; neither figure is an executed query-law measurement |
| `figure_oracle_accuracy.py` | `oracle_accuracy.png`, `oracle_accuracy_figure_data.json` — reads `v20/collapsed_resource_evidence.json`, collapsed leg at the manuscript cap. The 18- and 24-bit axis resource counts are absent from that artifact and are recomputed here, with the 30-bit row asserted against its ledger |

## Numbers in Section 6

| Script | What it backs |
|---|---|
| `matched_bias_normalisation.py` | `v19/` — normalisation curve, the matched-exceedance reproduction, the variance ratio |
| `v21_collapsed_resource_evidence.py` | `v20/collapsed_resource_evidence.json` — precision decision, retuned normalisation, resource ledger, formula-vs-built-circuit, scaling |
| `v22_collapsed_validation_ladder.py` | `v20/collapsed_validation_ladder.json` — the five validation rungs |
| `v20_three_way_price_validation.py` | `v20/three_way_price_validation.json` — the three-way price table, both agreement gaps as measured, and the aer MPS cross-check under `--mps-crosscheck` |
| `v4_linetsky_benchmark.py` | `v4/linetsky_benchmark.txt` — the seven spectral benchmarks |
| `v4_discretization_bias.py` | `v4/discretization_bias.txt` — the paired grid-bias table |
| `v4_linf_heatmap.py` | `v4/linf_heatmap.txt` — the strike and volatility normalisation grid |
| `v4_asian_exact12.py` | `v4/asian_exact12.txt` — twelve-date grid and the blocked ratios |
| `v4_iae_qcv_measured.py` | `v4/iae_qcv_measured.txt` — toy-grid IAE and the 2,048 floor |
| `v4_independent_verification.py` | `v4/independent_verification.txt` — the independent-circuit agreement |
| `v3_quantum_cv_toy.py` | `v3/quantum_cv_toy.txt` — the two- and three-date grids |
| `v3_weighted_sum_oracle.py` | `v3/weighted_sum_table.txt` — the affinity check against the published architecture |
| `validate_factorized_asian_oracle.py` | `v7/` — the lookup-table row and rotation counts |
| `validate_arithmetic_asian_oracle.py` | `v8/` — the per-date oracle, now superseded but still the audit baseline |
| `v13_arithmetic_oracle_scaleup.py` | `v9/arithmetic_oracle_scaleup.json` — the small-N scale-up |
| `v17_raw_vs_residual_oracle.py` | `v17/` — the original raw recount, superseded by the built comparison in v20 |
| `validate_full_asian_oracle.py` | `v5/full_asian_oracle_validation.md`, referenced by README.md:45 |

## Known gaps

The three-way price table and the oracle-accuracy figure both traced to nothing on 2026-07-27 and
now write artifacts; see the rows above.

`v4_qcv_extensions.py` still selects and evaluates its cutoffs on the same 2e6 paths, so the
clipping losses in `v4/qcv_extensions.txt` are in-sample. Re-measured out of sample against seed
20260727 they hold: every arm, including the raw one, lands within 1.1 standard errors of its
in-sample value. It also labels panel (b) with `f"{ratio:.0f}"`, so the k=1 ratio prints as 16
where the text says 15.9.
