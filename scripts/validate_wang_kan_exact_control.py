"""Validate the same-model geometric control for Wang--Kan weak Euler.

Unlike ``validate_wang_kan_cv_restoration.py``, this runner never combines a
continuous-Heston control with a weak-Euler residual.  Both terms in the
restoration identity are expectations under the exact binary weak-Euler model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qc_option_pricing.classical.heston_weak_euler import HestonWeakEulerSpec
from qc_option_pricing.classical.heston_weak_euler_geometric import (
    weak_euler_geometric_asian_price,
)


ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY = ROOT / "results" / "v10" / "wang_kan_cv_feasibility.json"
CONTINUOUS = ROOT / "results" / "v10" / "wang_kan_cv_restoration.json"
OUT = ROOT / "results" / "v11" / "wang_kan_exact_control.json"

INSTANCES = {
    "call_instance_1": HestonWeakEulerSpec(
        s0=100.0, v0=0.1, rate=0.03, rho=-0.1, kappa=2.0, theta=0.12,
        xi=0.3, maturity=1.0, n_steps=256, strike=90.0, option_type="call",
    ),
    "put_instance_2": HestonWeakEulerSpec(
        s0=100.0, v0=0.05, rate=0.05, rho=-0.1, kappa=2.0, theta=0.04,
        xi=0.2, maturity=1.0, n_steps=256, strike=110.0, option_type="put",
    ),
}

REFINEMENT = (
    (64, 100.0, 256, 1.5),
    (64, 200.0, 512, 1.5),
    (64, 400.0, 1024, 1.5),
    (128, 200.0, 512, 1.5),
    (128, 400.0, 1024, 1.5),
    (256, 400.0, 1024, 1.5),
    (256, 400.0, 1024, 1.25),
    (256, 400.0, 1024, 2.0),
    (512, 400.0, 1024, 1.5),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    feasibility = json.loads(FEASIBILITY.read_text())
    continuous = json.loads(CONTINUOUS.read_text())
    result: dict = {
        "schema_version": "wang-kan-exact-control-v1",
        "created_at_start": started,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "estimand": (
            "discounted arithmetic Asian payoff under Wang-Kan's exact "
            "binary weak-Euler finite model with 256 fixings"
        ),
        "identity": {
            "call": "V_weak = mu_G_weak + E_weak[D_call]",
            "put": "V_weak = mu_G_weak - E_weak[D_put]",
            "model_mismatch": 0.0,
        },
        "method": (
            "one-dimensional backward transform recursion in variance; "
            "binary b shock summed analytically; cubic interpolation in "
            "sqrt(v); exponentially damped payoff Fourier inversion"
        ),
        "source_hashes": {
            "src/qc_option_pricing/classical/heston_weak_euler.py": sha256(
                ROOT / "src/qc_option_pricing/classical/heston_weak_euler.py"),
            "src/qc_option_pricing/classical/heston_weak_euler_geometric.py": sha256(
                ROOT / "src/qc_option_pricing/classical/heston_weak_euler_geometric.py"),
            "tests/test_heston_weak_euler_geometric.py": sha256(
                ROOT / "tests/test_heston_weak_euler_geometric.py"),
            "scripts/validate_wang_kan_exact_control.py": sha256(Path(__file__)),
            "results/v10/wang_kan_cv_feasibility.json": sha256(FEASIBILITY),
            "results/v10/wang_kan_cv_restoration.json": sha256(CONTINUOUS),
        },
        "instances": {},
    }

    for name, spec in INSTANCES.items():
        rows = []
        for nodes, u_max, order, damping in REFINEMENT:
            priced = weak_euler_geometric_asian_price(
                spec, variance_nodes=nodes, u_max=u_max,
                quadrature_order=order, damping=damping,
            )
            rows.append(priced)
            print(name, nodes, u_max, order, damping,
                  f"{priced['price']:.12f}")

        selected = rows[-1]
        previous_grid = rows[5]
        same_grid_cutoffs = rows[1:3]
        damping_rows = rows[5:8]
        interpolation_change = abs(selected["price"] - previous_grid["price"])
        cutoff_change = abs(same_grid_cutoffs[-1]["price"]
                            - same_grid_cutoffs[-2]["price"])
        damping_spread = max(row["price"] for row in damping_rows) - min(
            row["price"] for row in damping_rows
        )

        old = feasibility["production"]["instances"][name]
        pooled = old["pooled"]
        residual = pooled["residual_mean_discounted"]
        residual_se = pooled["residual_se"]
        sign = 1.0 if spec.option_type == "call" else -1.0
        restored = selected["price"] + sign * residual
        raw = pooled["arithmetic_price_discounted"]
        raw_se = pooled["arithmetic_price_se"]

        geometric_replicates = np.asarray(
            [row["geometric_price_discounted"] for row in old["replicates"]],
            dtype=float,
        )
        geometric_mc = float(geometric_replicates.mean())
        geometric_mc_se = float(
            geometric_replicates.std(ddof=1) / math.sqrt(geometric_replicates.size)
        )
        geometric_z = ((geometric_mc - selected["price"])
                       / geometric_mc_se)

        continuous_control = continuous["instances"][name][
            "mu_g_exact_discrete_heston"
        ]
        result["instances"][name] = {
            "spec": {
                "s0": spec.s0, "v0": spec.v0, "rate": spec.rate,
                "rho": spec.rho, "kappa": spec.kappa,
                "theta": spec.theta, "xi": spec.xi,
                "maturity": spec.maturity, "n_steps": spec.n_steps,
                "strike": spec.strike, "option_type": spec.option_type,
            },
            "refinement": rows,
            "selected_geometric_control_price": selected["price"],
            "interpolation_refinement_change": interpolation_change,
            "fourier_refinement_change": cutoff_change,
            "damping_parameter_spread": damping_spread,
            "numerical_status": (
                "refinement-supported, not a formal interval certificate"
            ),
            "independent_geometric_mc": {
                "replicates": int(geometric_replicates.size),
                "mean": geometric_mc,
                "se_across_replicates": geometric_mc_se,
                "difference_in_se": geometric_z,
            },
            "residual_mean_discounted": residual,
            "residual_se": residual_se,
            "restored_weak_euler_arithmetic_price": restored,
            "restored_statistical_se": residual_se,
            "raw_weak_euler_mc_price": raw,
            "raw_weak_euler_mc_se": raw_se,
            "restored_minus_raw_mc": restored - raw,
            "restored_minus_raw_in_raw_mc_se": (restored - raw) / raw_se,
            "continuous_heston_geometric_control_for_comparison_only": (
                continuous_control
            ),
            "weak_minus_continuous_geometric_control": (
                selected["price"] - continuous_control
            ),
            "error_budget": {
                "model_mismatch_within_declared_weak_euler_estimand": 0.0,
                "control_interpolation_refinement_change": interpolation_change,
                "control_fourier_refinement_change": cutoff_change,
                "control_damping_parameter_spread": damping_spread,
                "residual_classical_validation_se": residual_se,
                "amplitude_estimation": "not executed; allocate separately",
                "fixed_point_arithmetic": "unmeasured at production precision",
                "clipping": "empirical estimate only; not certified",
            },
        }

    result["created_at_end"] = datetime.now(
        timezone.utc).astimezone().isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=1) + "\n")
    temporary.replace(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
