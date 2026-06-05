from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_flow2.utils.io import read_json, write_json


@dataclass(frozen=True)
class VAELatentsMeta:
    version: int
    dtype: str
    num_rows: int
    latent_channels: int
    latent_h: int
    latent_w: int
    mean_file: str
    std_file: str
    sources: dict[str, Any]


class VAELatentsCache:
    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / "vae_latents_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"vae_latents_meta.json not found: {meta_path}")
        obj = read_json(meta_path)
        self.meta = VAELatentsMeta(
            version=int(obj["version"]),
            dtype=str(obj["dtype"]),
            num_rows=int(obj["num_rows"]),
            latent_channels=int(obj["latent_channels"]),
            latent_h=int(obj["latent_h"]),
            latent_w=int(obj["latent_w"]),
            mean_file=str(obj["mean_file"]),
            std_file=str(obj["std_file"]),
            sources=dict(obj.get("sources", {})),
        )

        self.mean = np.load(cache_dir / self.meta.mean_file, mmap_mode="r")
        self.std = np.load(cache_dir / self.meta.std_file, mmap_mode="r")

        if self.mean.shape != self.std.shape:
            raise ValueError(f"mean/std shape mismatch: {self.mean.shape} vs {self.std.shape}")
        if self.mean.ndim != 4:
            raise ValueError(f"Expected [N,C,H,W] arrays, got {self.mean.shape}")
        if int(self.mean.shape[0]) != int(self.meta.num_rows):
            raise ValueError(f"num_rows mismatch: {self.mean.shape[0]=} vs {self.meta.num_rows=}")

    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        return np.asarray(self.mean[indices]), np.asarray(self.std[indices])

    def sample_batch(
        self,
        indices: np.ndarray,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        std_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mean_np, std_np = self.get_batch(indices)
        mean = torch.from_numpy(mean_np).to(device=device, dtype=dtype)
        std = torch.from_numpy(std_np).to(device=device, dtype=dtype)
        if generator is None:
            eps = torch.randn_like(mean)
        else:
            eps = torch.randn(mean.shape, device=device, dtype=dtype, generator=generator)
        return mean + (float(std_scale) * std) * eps


def write_vae_latents_meta(
    *,
    out_dir: str | Path,
    mean_file: str,
    std_file: str,
    num_rows: int,
    latent_channels: int,
    latent_h: int,
    latent_w: int,
    dtype: str,
    sources: dict[str, Any],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = VAELatentsMeta(
        version=1,
        dtype=str(dtype),
        num_rows=int(num_rows),
        latent_channels=int(latent_channels),
        latent_h=int(latent_h),
        latent_w=int(latent_w),
        mean_file=str(mean_file),
        std_file=str(std_file),
        sources=dict(sources),
    )
    write_json(out_dir / "vae_latents_meta.json", meta.__dict__)

