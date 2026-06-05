from __future__ import annotations

from typing import Tuple

import torch


RGB_MAP_CELLPAINTING_6CH: list[list[int]] = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]
TO_RGB_SCALE_DIV: float = 3.0
TO_RGB_RESCALE_BOUNDS: Tuple[float, float] = (0.1, 99.9)
TO_RGB_RESCALE_OUT_RANGE: Tuple[float, float] = (0.0, 1.0)
TO_RGB_RESCALE_SAMPLE_STRIDE: int = 100


def rescale_intensity(
    arr: torch.Tensor,
    *,
    bounds: Tuple[float, float] = (0.5, 99.5),
    out_range: Tuple[float, float] = (0.0, 1.0),
    sample_stride: int = 100,
) -> torch.Tensor:
    arr = arr.float()
    max_val = float(arr.max().detach().cpu()) if arr.numel() else 0.0
    if max_val > 1.0:
        arr = arr / (65535.0 if max_val > 255.0 else 255.0)

    sample = arr.flatten()[:: int(sample_stride)]
    if sample.numel() == 0:
        return arr

    percentiles = torch.quantile(
        sample,
        torch.tensor([float(bounds[0]) / 100.0, float(bounds[1]) / 100.0], device=arr.device),
    )
    lo, hi = percentiles[0], percentiles[1]
    denom = torch.clamp(hi - lo, min=1e-6)
    arr = torch.clamp(arr, lo, hi)
    arr = (arr - lo) / denom
    arr = arr * (float(out_range[1]) - float(out_range[0])) + float(out_range[0])
    return arr


def to_rgb(img: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {img.shape}")
    b, c, h, w = img.shape
    if c != 6:
        raise ValueError(f"Expected 6 channels, got {c}")
    prepped = img

    rgb_map = torch.tensor(RGB_MAP_CELLPAINTING_6CH, dtype=dtype, device=prepped.device)
    rgb_img = torch.einsum("nchw,ct->nthw", prepped.to(dtype=dtype), rgb_map) / float(TO_RGB_SCALE_DIV)
    return rescale_intensity(
        rgb_img,
        bounds=TO_RGB_RESCALE_BOUNDS,
        out_range=TO_RGB_RESCALE_OUT_RANGE,
        sample_stride=TO_RGB_RESCALE_SAMPLE_STRIDE,
    )
