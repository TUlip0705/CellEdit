from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def make_time_grid(steps: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    s = int(steps)
    if s <= 0:
        raise ValueError(f"steps must be >0, got {steps}")
    t = torch.linspace(1.0, 0.0, steps=s + 1, device=device, dtype=dtype)
    return t


def step_dt(t_i: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    """
    Positive dt for an update where t decreases (t_i > t_next).
    """
    return t_i - t_next
