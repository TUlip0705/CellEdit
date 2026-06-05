from __future__ import annotations

import torch


def sample_time_uniform(batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.rand((int(batch_size),), device=device, dtype=dtype)


def flow_matching_linear_path(z_data: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Linear path flow-matching (v-pred):

      z_t = (1-t) * z_data + t * z_noise
      v*  = z_noise - z_data

    Returns:
      z_t, v_target, z_noise
    """
    if t.ndim != 1:
        t = t.view(-1)
    while t.ndim < z_data.ndim:
        t = t.view(*t.shape, 1)
    z_noise = torch.randn_like(z_data)
    z_t = (1.0 - t) * z_data + t * z_noise
    v_target = z_noise - z_data
    return z_t, v_target, z_noise


def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float()
    y = y.float()
    return torch.mean((x - y) ** 2)
