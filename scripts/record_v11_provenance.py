"""Checksum manifest for the same-model Wang--Kan control artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/v11/PROVENANCE.json"
SOURCE_FILES = (
    "src/qc_option_pricing/classical/heston_weak_euler.py",
    "src/qc_option_pricing/classical/heston_weak_euler_geometric.py",
    "tests/test_heston_weak_euler_geometric.py",
    "scripts/validate_wang_kan_exact_control.py",
    "scripts/record_v11_provenance.py",
    "audit/wang-kan-exact-control-2026-07-18/baseline.md",
    "audit/wang-kan-exact-control-2026-07-18/derivation.md",
    "audit/wang-kan-exact-control-2026-07-18/claim-ledger.md",
    "audit/wang-kan-exact-control-2026-07-18/decision.md",
)
RESULT_FILES = (
    "results/v11/wang_kan_exact_control.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import numpy
    import scipy

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "generated_by": "scripts/record_v11_provenance.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "source_files": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "result_files": {name: sha256(ROOT / name) for name in RESULT_FILES},
    }
    OUTPUT.write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote results/v11/PROVENANCE.json")


if __name__ == "__main__":
    main()
