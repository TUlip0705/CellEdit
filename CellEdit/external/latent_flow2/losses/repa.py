from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_repa_loss(z_teacher: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
    """
    zs, zs_tilde: [B,T,D]
    """
    if z_teacher.ndim != 3 or z_pred.ndim != 3:
        raise ValueError(f"Expected [B,T,D], got {z_teacher.shape} and {z_pred.shape}")
    if z_teacher.shape != z_pred.shape:
        raise ValueError(f"Shape mismatch: {z_teacher.shape} vs {z_pred.shape}")
    # keep this loss numerically stable under AMP
    z_teacher = z_teacher.float()
    z_pred = z_pred.float()
    z_teacher = F.normalize(z_teacher, dim=-1)
    z_pred = F.normalize(z_pred, dim=-1)
    cos = (z_teacher * z_pred).sum(dim=-1)  # [B,T]
    return (-cos).mean()


def spatial_zscore(feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score along spatial (token) dimension: (B,T,D)
    """
    feat = feat.float()
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True)
    return (feat - float(alpha) * mean) / (std + float(eps))
