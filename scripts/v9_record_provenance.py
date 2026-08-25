"""Stamp the v9 oracle result files with run provenance.

The v9 experiments were run before the runners recorded provenance, and the
nine-hour champion simulation cannot be cheaply rerun.  This script writes a
``provenance`` block into each ``results/v9`` result file and a checksum
manifest ``results/v9/PROVENANCE.json``.

Field sources are stated explicitly.  Commands, seeds, and simulator
settings are read from the runner scripts.  Start and end times come from
the creation and final-modification timestamps of the background task logs
that captured each run's stdout; they agree with the ``sim_seconds`` values
recorded inside the payloads (for example the champion run: wall
``14:09:13`` to ``23:24:21`` is 33,308 seconds against 33,305 recorded
simulation seconds).  Foreground runs did not keep a separate log, so only
the file-modification end time is known and start times are null.  Package
versions are measured from the environment, which is unchanged since the
runs.  Source hashes are computed at stamp time; the runners were revised
during the session as later parts were added, the recorded parts' code
paths are unchanged in the hashed revisions, and the intermediate revisions
were not retained.

Idempotent: result-file bytes are stable across re-runs of this script.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy
import qiskit
import qiskit_aer
import scipy

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "results" / "v9"

MPS_SETTINGS = {
    "simulator": "qiskit_aer.AerSimulator",
    "method": "matrix_product_state",
    "shots": None,
    "probabilities": "analytic, save_probabilities_dict",
    "decompose_reps": 8,
    "other_options": "Aer defaults",
}
SV_SETTINGS = {**MPS_SETTINGS, "method": "statevector"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
    }


def source_hashes(runner: str) -> dict:
    return {
        "oracle_source_sha256": sha256(
            ROOT / "src" / "qc_option_pricing" / "quantum" / "asian_oracle.py"
        ),
        "runner_sha256": sha256(ROOT / "scripts" / runner),
        "runner_file": f"scripts/{runner}",
        "validation_test_sha256": sha256(ROOT / "tests" / "test_asian_oracle.py"),
        "validation_test_file": "tests/test_asian_oracle.py",
        "runner_revision_note": (
            "runner revised during the session as later parts were added; "
            "recorded parts' code paths are unchanged in this hashed "
            "revision; intermediate revisions were not retained"
        ),
    }


STAMPS = {
    "oracle_accuracy_sweeps.json": {
        "command": ".venv/bin/python scripts/v9_oracle_accuracy_sweeps.py",
        "runner": "v9_oracle_accuracy_sweeps.py",
        "runs": [{
            "parts": "all sweeps, single foreground run",
            "start": None,
            "end": "2026-07-17T13:47:05-06:00",
            "time_source": "result file modification time; foreground run, start not logged",
        }],
        "simulator_settings": "no circuit simulation; exhaustive enumeration and quadrature",
        "seeds": {"sobol_scramble_seeds": [7, 8, 9, 10, 11, 12, 13, 14]},
    },
    "oracle_circuit_check.json": {
        "command": ".venv/bin/python scripts/v9_oracle_circuit_check.py",
        "runner": "v9_oracle_circuit_check.py",
        "runs": [{
            "parts": "all six circuits, single run",
            "start": "2026-07-17T13:46:51-06:00",
            "end": "2026-07-17T13:53:58-06:00",
            "time_source": "background task log creation and final modification times",
        }],
        "simulator_settings": MPS_SETTINGS,
        "seeds": {"note": "deterministic; analytic simulation, closed-form reference, no RNG"},
    },
    "oracle_champion.json": {
        "command": ".venv/bin/python scripts/v9_oracle_champion.py --part <part>",
        "runner": "v9_oracle_champion.py",
        "runs": [
            {"parts": "champion_ps32",
             "command": ".venv/bin/python scripts/v9_oracle_champion.py --part champion_ps32",
             "start": "2026-07-17T14:09:13-06:00",
             "end": "2026-07-17T23:24:21-06:00",
             "time_source": "background task log creation and final modification times; "
                            "wall 33,308 s against 33,305 recorded simulation seconds"},
            {"parts": "capped",
             "command": ".venv/bin/python scripts/v9_oracle_champion.py --part capped",
             "start": "2026-07-17T14:16:46-06:00",
             "end": "2026-07-17T14:22:56-06:00",
             "time_source": "background task log creation and final modification times"},
            {"parts": "roundtrip_raw_ps8, roundtrip_raw_ps32",
             "command": ".venv/bin/python scripts/v9_oracle_champion.py --part roundtrip",
             "start": "2026-07-17T23:32:06-06:00",
             "end": "2026-07-17T23:37:00-06:00",
             "time_source": "background task log creation and final modification times",
             "note": "two earlier runs of this part used the module's dense roundtrip "
                     "helper, were killed without output, and produced no results; the "
                     "recorded run uses the per-qubit union bound"},
            {"parts": "statevector_ps1, statevector_ps2",
             "command": ".venv/bin/python scripts/v9_oracle_champion.py --part statevector",
             "start": "2026-07-17T23:37:13-06:00",
             "end": "2026-07-18T00:09:03-06:00",
             "time_source": "background task log creation and final modification times"},
        ],
        "simulator_settings": {
            "champion_and_capped_and_roundtrip": MPS_SETTINGS,
            "statevector_parts": SV_SETTINGS,
        },
        "seeds": {"note": "deterministic; analytic simulation, closed-form reference, no RNG"},
    },
    "oracle_robustness.json": {
        "command": ".venv/bin/python scripts/v9_oracle_robustness.py --part <part>",
        "runner": "v9_oracle_robustness.py",
        "runs": [{
            "parts": "grid, grid16, dates, dates16, quadrature_ladder; separate foreground runs",
            "start": None,
            "end": "2026-07-17T14:17:40-06:00",
            "time_source": "result file modification time marks the final merge; "
                           "foreground runs, per-part wall times not logged",
        }],
        "simulator_settings": "no circuit simulation; exhaustive enumeration and quadrature",
        "seeds": {"sobol_scramble_seeds": [7, 8, 9, 10]},
    },
}


def main() -> None:
    env = environment()
    for name, stamp in STAMPS.items():
        path = V9 / name
        data = json.loads(path.read_text())
        data["provenance"] = {
            "command": stamp["command"],
            "runs": stamp["runs"],
            "environment": env,
            "simulator_settings": stamp["simulator_settings"],
            "seeds": stamp["seeds"],
            **source_hashes(stamp["runner"]),
            "recorded_after_run": True,
            "recording_script": "scripts/v9_record_provenance.py",
        }
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"stamped {name}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "generated_by": "scripts/v9_record_provenance.py",
        "environment": env,
        "source_files": {
            "src/qc_option_pricing/quantum/asian_oracle.py": sha256(
                ROOT / "src" / "qc_option_pricing" / "quantum" / "asian_oracle.py"),
            "tests/test_asian_oracle.py": sha256(ROOT / "tests" / "test_asian_oracle.py"),
            **{f"scripts/{s}": sha256(ROOT / "scripts" / s) for s in (
                "v9_oracle_accuracy_sweeps.py", "v9_oracle_circuit_check.py",
                "v9_oracle_champion.py", "v9_oracle_robustness.py",
                "v9_record_provenance.py")},
        },
        "result_files": {f"results/v9/{name}": sha256(V9 / name) for name in STAMPS},
    }
    (V9 / "PROVENANCE.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote results/v9/PROVENANCE.json")


if __name__ == "__main__":
    main()
