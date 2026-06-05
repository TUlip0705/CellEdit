#!/usr/bin/env python3
from __future__ import annotations

"""Training entry point for CellEdit latent flow models."""

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))


if __name__ == "__main__":
    runpy.run_path(
        str(ROOT / "external" / "latent_flow2" / "train" / "train_latent_sit.py"),
        run_name="__main__",
    )
