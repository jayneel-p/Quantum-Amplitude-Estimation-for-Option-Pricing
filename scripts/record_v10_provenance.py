"""Checksum manifest for the v10 Wang-Kan control-variate artifacts.

Unlike the v9 after-the-fact stamping, every v10 result file already carries
during-run provenance (command, start/end times, environment, source hashes,
seeds).  This manifest adds final SHA-256 hashes of the result files
themselves plus the source, test, and runner files.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "results" / "v10"

FILES = [
    "src/qc_option_pricing/classical/heston_weak_euler.py",
    "src/qc_option_pricing/quantum/wang_kan_resources.py",
    "tests/test_heston_weak_euler.py",
    "tests/test_wang_kan_resources.py",
    "scripts/validate_wang_kan_cv_feasibility.py",
    "scripts/validate_wang_kan_cv_resources.py",
    "scripts/validate_wang_kan_cv_small_oracle.py",
    "scripts/record_v10_provenance.py",
    "src/qc_option_pricing/classical/heston_geometric_asian.py",
    "tests/test_heston_geometric_asian.py",
    "scripts/validate_wang_kan_cv_restoration.py",
]
RESULTS = [
    "results/v10/wang_kan_cv_feasibility.json",
    "results/v10/wang_kan_cv_resources.json",
    "results/v10/wang_kan_cv_small_oracle.json",
    "results/v10/wang_kan_cv_restoration.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import numpy, scipy, qiskit, qiskit_aer

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "generated_by": "scripts/record_v10_provenance.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
        },
        "source_files": {f: sha256(ROOT / f) for f in FILES},
        "result_files": {f: sha256(ROOT / f) for f in RESULTS},
    }
    (V10 / "PROVENANCE.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote results/v10/PROVENANCE.json")


if __name__ == "__main__":
    main()
