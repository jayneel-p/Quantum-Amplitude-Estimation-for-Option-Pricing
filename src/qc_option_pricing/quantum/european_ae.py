"""European call via log-normal loading, linear payoff map, and Qiskit amplitude estimation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_algorithms import (
    AmplitudeEstimation,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
    EstimationProblem,
)
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")

_A_OPERATOR_PM = generate_preset_pass_manager(
    optimization_level=1,
    basis_gates=["cx", "u"],
)


def _a_operator_transpiled_depth(qc: QuantumCircuit) -> int:
    """Depth of the state-preparation / A-operator block after transpile to CX+U."""
    return int(_A_OPERATOR_PM.run(qc).depth())


@dataclass
class QuantumPricingResult:
    price: float
    undiscounted_payoff: float
    raw_amplitude: float
    n_qubits_distribution: int
    n_qubits_total: int
    circuit_depth: int
    n_oracle_queries: int
    ae_method: str
    sampler_seed: int | None
    sampler_default_shots: int
    powers: tuple[int, ...]
    ratios: tuple[float, ...]
    round_shots: tuple[int, ...]
    round_oracle_queries: tuple[int, ...]
    confidence_interval: tuple[float, float] | None
    confidence_interval_processed: tuple[float, float] | None
    estimate_intervals: tuple[tuple[float, float], ...]


def build_european_call_circuit(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_qubits: int,
    c_approx: float = 0.10,
    n_stddevs: float = 3.0,
) -> tuple[QuantumCircuit, int, "callable"]:
    """Return (circuit, objective_qubit_index, post_processing_fn)."""
    mu_log = math.log(s0) + (r - 0.5 * sigma ** 2) * t
    sigma_log = sigma * math.sqrt(t)

    low = max(1e-4, np.exp(mu_log - n_stddevs * sigma_log))
    high = np.exp(mu_log + n_stddevs * sigma_log)

    uncertainty_model = LogNormalDistribution(
        num_qubits=n_qubits,
        mu=mu_log,
        sigma=sigma_log ** 2,
        bounds=(low, high),
    )

    f_max = high - k

    european_call_objective = LinearAmplitudeFunction(
        num_state_qubits=n_qubits,
        slope=[0, 1],
        offset=[0, 0],
        domain=(low, high),
        image=(0, f_max),
        breakpoints=[low, k],
        rescaling_factor=c_approx,
    )

    n_total = european_call_objective.num_qubits
    qc = QuantumCircuit(n_total)
    qc.append(uncertainty_model, range(n_qubits))
    qc.append(european_call_objective, range(n_total))

    objective_qubit = n_qubits

    return qc, objective_qubit, european_call_objective.post_processing


def price_european_call_quantum(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_qubits: int = 5,
    ae_method: str = "iae",
    n_eval_qubits: int = 5,
    epsilon: float = 0.01,
    alpha: float = 0.05,
    shots: int = 1024,
    c_approx: float = 0.10,
    seed: int | None = None,
) -> QuantumPricingResult:
    """Price a European call and retain the stochastic AE schedule.

    ``ae_method`` is ``'ae'``, ``'iae'``, or ``'mlae'``.  For IAE use
    ``epsilon``/``alpha``; for AE/MLAE use ``n_eval_qubits``.  ``shots`` and
    ``seed`` are passed directly to :class:`StatevectorSampler`, so independent
    trials can be exactly reproduced.
    """
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    qc, obj_qubit, post_processing = build_european_call_circuit(
        s0, k, r, sigma, t, n_qubits, c_approx=c_approx
    )

    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=[obj_qubit],
        post_processing=post_processing,
    )

    sampler = StatevectorSampler(default_shots=shots, seed=seed)

    if ae_method == "ae":
        ae = AmplitudeEstimation(
            num_eval_qubits=n_eval_qubits,
            sampler=sampler,
        )
    elif ae_method == "iae":
        ae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon,
            alpha=alpha,
            sampler=sampler,
        )
    elif ae_method == "mlae":
        ae = MaximumLikelihoodAmplitudeEstimation(
            evaluation_schedule=list(range(0, n_eval_qubits)),
            sampler=sampler,
        )
    else:
        raise ValueError(f"Unknown ae_method: {ae_method!r}")

    result = ae.estimate(problem)

    raw_amplitude = result.estimation
    undiscounted_payoff = result.estimation_processed
    discount = math.exp(-r * t)
    price = discount * undiscounted_payoff

    depth = _a_operator_transpiled_depth(qc)

    if ae_method == "ae":
        n_queries = 2 ** n_eval_qubits
        total_qubits = qc.num_qubits + n_eval_qubits
    elif ae_method == "iae":
        n_queries = getattr(result, "num_oracle_queries", 0)
        total_qubits = qc.num_qubits + 1
    else:
        n_queries = sum(2 ** j for j in range(n_eval_qubits))
        total_qubits = qc.num_qubits + 1

    powers = tuple(int(power) for power in (getattr(result, "powers", None) or ()))
    ratios = tuple(float(ratio) for ratio in (getattr(result, "ratios", None) or ()))
    round_shots = tuple(shots for _ in powers)
    round_oracle_queries = tuple(shots * power for power in powers)
    confidence_interval_value = getattr(result, "confidence_interval", None)
    confidence_interval = (
        tuple(float(value) for value in confidence_interval_value)
        if confidence_interval_value is not None
        else None
    )
    processed_interval_value = getattr(result, "confidence_interval_processed", None)
    confidence_interval_processed = (
        tuple(float(value) for value in processed_interval_value)
        if processed_interval_value is not None
        else None
    )
    estimate_intervals = tuple(
        tuple(float(value) for value in interval)
        for interval in (getattr(result, "estimate_intervals", None) or ())
    )

    return QuantumPricingResult(
        price=price,
        undiscounted_payoff=undiscounted_payoff,
        raw_amplitude=raw_amplitude,
        n_qubits_distribution=n_qubits,
        n_qubits_total=total_qubits,
        circuit_depth=depth,
        n_oracle_queries=n_queries,
        ae_method=ae_method,
        sampler_seed=seed,
        sampler_default_shots=shots,
        powers=powers,
        ratios=ratios,
        round_shots=round_shots,
        round_oracle_queries=round_oracle_queries,
        confidence_interval=confidence_interval,
        confidence_interval_processed=confidence_interval_processed,
        estimate_intervals=estimate_intervals,
    )


def european_call_quantum_convergence(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    qubit_range: list[int] | None = None,
    ae_method: str = "iae",
    epsilon: float = 0.01,
    alpha: float = 0.05,
) -> list[QuantumPricingResult]:
    """Run pricing for each n in qubit_range (default 3..7)."""
    if qubit_range is None:
        qubit_range = [3, 4, 5, 6, 7]

    results = []
    for nq in qubit_range:
        print(f"  n_qubits={nq}, ae_method={ae_method} ... ", end="", flush=True)
        try:
            res = price_european_call_quantum(
                s0, k, r, sigma, t,
                n_qubits=nq,
                ae_method=ae_method,
                epsilon=epsilon,
                alpha=alpha,
            )
            results.append(res)
            print(f"price={res.price:.4f}, depth={res.circuit_depth}, "
                  f"queries={res.n_oracle_queries}")
        except Exception as e:
            print(f"FAILED: {e}")
    return results
