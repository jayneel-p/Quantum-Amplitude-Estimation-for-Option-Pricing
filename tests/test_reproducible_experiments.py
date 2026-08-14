"""Regression tests for seeded, machine-readable experiment paths."""

from __future__ import annotations

import unittest

from qc_option_pricing.quantum.european_ae import price_european_call_quantum


class TestReproducibleExperiments(unittest.TestCase):
    def test_iae_seed_and_schedule_are_retained(self) -> None:
        result = price_european_call_quantum(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            n_qubits=4,
            ae_method="iae",
            epsilon=0.03,
            alpha=0.32,
            shots=256,
            seed=20_260_716,
        )
        self.assertEqual(result.sampler_seed, 20_260_716)
        self.assertEqual(result.sampler_default_shots, 256)
        self.assertEqual(result.round_shots, tuple(256 for _ in result.powers))
        self.assertEqual(sum(result.round_oracle_queries), result.n_oracle_queries)
        self.assertIsNotNone(result.confidence_interval)


if __name__ == "__main__":
    unittest.main()
