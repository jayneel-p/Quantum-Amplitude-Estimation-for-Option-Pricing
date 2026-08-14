#!/usr/bin/env python3
"""
Run this from PyCharm (project root) without configuring source roots.

Prefer for development: pip install -e .  then use scripts/run_classical_demo.py or
console entry qc-option-pricing-demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qc_option_pricing.demo import main

if __name__ == "__main__":
    main()
