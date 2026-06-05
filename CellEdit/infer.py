#!/usr/bin/env python3
from __future__ import annotations

"""Inference entry point for CellEdit FlowEdit-style counterfactual editing."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))

from celledit.flowedit.edit import main


if __name__ == "__main__":
    main()
