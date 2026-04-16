# Changelog

## 2026-04-12

- **Quantum European AE (`european_ae.py`):** `QuantumPricingResult.circuit_depth` now uses the **transpiled** A-operator depth (basis `cx` + `u`, `optimization_level=1`), matching `scripts/generate_quantum_results.py` resource counts. Previously `qc.decompose().depth()` reflected only macro-instruction layers (~2) and was misleading for reporting.

- **Quantum circuit PNGs (`generate_quantum_results.py`):** Figures are **Stamatopoulos-style schematics** (`qc_option_pricing.visualization.sta20_a_operator`): serif layout, black wires, rounded boxes, abbreviated **Fig. 2 (right)** notation ($|i\rangle_n$, $P_X^S$, controlled $R_y[\tilde f(i)]$). Full transpiled CX+U diagrams stay in the console line and `quantum_resource_table.txt` for gate accounting.
