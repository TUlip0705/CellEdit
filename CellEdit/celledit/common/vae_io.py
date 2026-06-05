from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import yaml

from adapter_vae.data.preprocessing import CellPaintingPreprocessor
from celledit.common.vae_loader import load_vae_model


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[2] / p).resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_repo_path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML root: {p}")
    return obj


def build_preprocessor_from_vae_config(vae_config_path: str | Path) -> CellPaintingPreprocessor:
    cfg = _load_yaml(vae_config_path)

    fixed_stats_path = cfg.get("fixed_stats_path", None)
    if not fixed_stats_path:
        raise ValueError("VAE config missing `fixed_stats_path`.")

    bounds = cfg.get("percentile_bounds", "0.5,99.5")
    if isinstance(bounds, str):
        parts = [p.strip() for p in bounds.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid percentile_bounds string: {bounds}")
        percentile_bounds = (float(parts[0]), float(parts[1]))
    else:
        percentile_bounds = (float(bounds[0]), float(bounds[1]))

    return CellPaintingPreprocessor(
        percentile_bounds=percentile_bounds,
        use_channel_standardize=bool(cfg.get("use_channel_standardize", False)),
        percentile_per_channel=bool(cfg.get("percentile_per_channel", True)),
        sample_stride=int(cfg.get("sample_stride", 100)),
        fixed_stats_path=str(fixed_stats_path),
    )


def pretrained_vae_from_config(vae_config_path: str | Path) -> Optional[str]:
    cfg = _load_yaml(vae_config_path)
    pretrained_vae = cfg.get("pretrained_vae", None)
    if pretrained_vae in (None, ""):
        return None
    return str(_resolve_repo_path(pretrained_vae))


def load_vae(
    *,
    vae_model: str,
    vae_ckpt: str | Path,
    device: torch.device,
    pretrained_vae: Optional[str] = None,
    vae_config_path: Optional[str | Path] = None,
) -> torch.nn.Module:
    resolved_pretrained = pretrained_vae
    if resolved_pretrained is None and vae_config_path is not None:
        resolved_pretrained = pretrained_vae_from_config(vae_config_path)
    if resolved_pretrained is not None:
        resolved_pretrained = str(_resolve_repo_path(resolved_pretrained))

    return load_vae_model(
        model_type=str(vae_model),
        ckpt_path=str(_resolve_repo_path(vae_ckpt)),
        device=device,
        pretrained_vae=resolved_pretrained,
        strict=True,
    )


@dataclass(frozen=True)
class VaeEncodeResult:
    mean: torch.Tensor
    std: torch.Tensor
    sample: torch.Tensor


@torch.no_grad()
def vae_encode(
    vae: torch.nn.Module,
    x: torch.Tensor,
    *,
    sample_latent: bool = False,
    generator: Optional[torch.Generator] = None,
    amp: bool = True,
) -> VaeEncodeResult:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (B,6,H,W) or (6,H,W), got {x.shape}")

    use_amp = bool(amp) and x.device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        dist = vae.encode(x)

    mean = getattr(dist, "mean", None)
    if mean is None:
        raise RuntimeError("VAE encode() did not return distribution with `.mean`")

    logvar = getattr(dist, "logvar", None)
    if logvar is not None:
        std = torch.exp(0.5 * logvar)
    else:
        std = getattr(dist, "std", None)
        if std is None:
            raise RuntimeError("VAE distribution missing both `.logvar` and `.std`")

    if sample_latent:
        if generator is None:
            eps = torch.randn_like(mean)
        else:
            eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        z = mean + std * eps
    else:
        z = mean

    return VaeEncodeResult(mean=mean, std=std, sample=z)


@torch.no_grad()
def vae_decode(
    vae: torch.nn.Module,
    z: torch.Tensor,
    *,
    amp: bool = True,
    clamp: bool = True,
) -> torch.Tensor:
    if z.ndim == 3:
        z = z.unsqueeze(0)
    if z.ndim != 4:
        raise ValueError(f"Expected z shape (B,C,H,W), got {z.shape}")

    use_amp = bool(amp) and z.device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        x = vae.decode(z)

    if clamp:
        x = torch.clamp(x, -1.0, 1.0)
    return x


def load_raw_6ch_npy(rxrx3_root: str | Path, rel_path: str | Path) -> np.ndarray:
    root = _resolve_repo_path(rxrx3_root)
    p = root / str(rel_path)
    return np.load(p)


def preprocess_raw_6ch(
    raw: np.ndarray,
    preproc: CellPaintingPreprocessor,
    *,
    device: torch.device,
) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(raw)).to(device=device)
    if x.ndim != 3:
        raise ValueError(f"Expected raw shape (C,H,W), got {x.shape}")
    x = x.float()
    x = preproc(x)
    return x.unsqueeze(0)
