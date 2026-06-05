from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Tuple

import torch
import yaml

from latent_flow2.models.cond import (
    KPGTCondConfig,
    KPGTConditionEncoder,
    KPGTFiLMCondConfig,
    KPGTFiLMConditionEncoder,
)
from latent_flow2.models.sit import LatentSiT, LatentSiTConfig


DoseMode = Literal["bins", "cont_film"]


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[2] / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_repo_path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML root (expected dict): {p}")
    return obj


def parse_dose_mode(train_cfg: dict[str, Any]) -> DoseMode:
    mode = str(train_cfg.get("cond", {}).get("dose_mode", "bins")).strip().lower()
    if mode in ("bins", "discrete", "dose_id", ""):
        return "bins"
    if mode in ("cont_film", "log10_film", "film"):
        return "cont_film"
    raise ValueError(f"Unsupported cond.dose_mode: {mode}")


def build_cond_and_sit(train_cfg: dict[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module, DoseMode]:
    cond_cfg = train_cfg.get("cond", {}) if isinstance(train_cfg.get("cond", {}), dict) else {}
    model_cfg = train_cfg.get("model", {}) if isinstance(train_cfg.get("model", {}), dict) else {}

    dose_mode = parse_dose_mode(train_cfg)

    if dose_mode == "bins":
        cond = KPGTConditionEncoder(
            KPGTCondConfig(
                kpgt_dim=int(cond_cfg.get("kpgt_dim", 2304)),
                hidden_dim=int(cond_cfg.get("hidden_dim", 384)),
                dose_bins=int(cond_cfg.get("dose_bins", 9)),
                cond_drop_prob=0.0,
                keep_empty_id=bool(cond_cfg.get("keep_empty_id", True)),
                kpgt_dropout=float(cond_cfg.get("kpgt_dropout", 0.1)),
            )
        )
    else:  # cont_film
        cond = KPGTFiLMConditionEncoder(
            KPGTFiLMCondConfig(
                kpgt_dim=int(cond_cfg.get("kpgt_dim", 2304)),
                hidden_dim=int(cond_cfg.get("hidden_dim", 384)),
                cond_drop_prob=0.0,
                keep_empty_id=bool(cond_cfg.get("keep_empty_id", True)),
                kpgt_dropout=float(cond_cfg.get("kpgt_dropout", 0.1)),
                dose_fourier_k=int(cond_cfg.get("dose_fourier_k", 16)),
                dose_mlp_dim=int(cond_cfg.get("dose_mlp_dim", int(cond_cfg.get("hidden_dim", 384)))),
                dose_alpha_init=float(cond_cfg.get("dose_alpha_init", 0.1)),
            )
        )

    sit = LatentSiT(
        LatentSiTConfig(
            input_size=int(model_cfg.get("input_size", 64)),
            patch_size=int(model_cfg.get("patch_size", 2)),
            in_channels=int(model_cfg.get("in_channels", 24)),
            hidden_size=int(model_cfg.get("hidden_size", 384)),
            depth=int(model_cfg.get("depth", 12)),
            num_heads=int(model_cfg.get("num_heads", 6)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            encoder_depth=int(model_cfg.get("encoder_depth", 4)),
            time_dim=int(model_cfg.get("time_dim", 128)),
            cond_dim=int(cond_cfg.get("hidden_dim", 384)),
            teacher_dim=int(model_cfg.get("teacher_dim", 384)),
            projector_type=str(model_cfg.get("projector_type", "mlp")),
            projector_dim=int(model_cfg.get("projector_dim", 2048)),
            conv_kernel=int(model_cfg.get("conv_kernel", 3)),
            use_input_bn=bool(model_cfg.get("use_input_bn", False)),
            bn_momentum=float(model_cfg.get("bn_momentum", 0.1)),
            bn_eps=float(model_cfg.get("bn_eps", 1e-4)),
        )
    )
    return cond, sit, dose_mode


@dataclass(frozen=True)
class LatentFlowBundle:
    cond: torch.nn.Module
    sit: torch.nn.Module
    dose_mode: DoseMode
    sit_state_used: str


def load_latent_flow_bundle(
    *,
    ckpt_path: str | Path,
    train_config: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> LatentFlowBundle:
    cfg = load_yaml(train_config)
    cond, sit, dose_mode = build_cond_and_sit(cfg)

    ckpt = torch.load(str(_resolve_repo_path(ckpt_path)), map_location="cpu")
    if not isinstance(ckpt, dict) or "cond_state" not in ckpt or "sit_state" not in ckpt:
        raise ValueError(f"Unexpected latent_flow2 ckpt format: {ckpt_path}")

    cond.load_state_dict(ckpt["cond_state"], strict=True)
    sit_state_used = "sit_state"
    if bool(use_ema) and ckpt.get("ema_state", None) is not None:
        sit.load_state_dict(ckpt["ema_state"], strict=True)
        sit_state_used = "ema_state"
    else:
        sit.load_state_dict(ckpt["sit_state"], strict=True)

    cond.to(device).eval()
    sit.to(device).eval()
    for p in cond.parameters():
        p.requires_grad_(False)
    for p in sit.parameters():
        p.requires_grad_(False)

    return LatentFlowBundle(cond=cond, sit=sit, dose_mode=dose_mode, sit_state_used=sit_state_used)


def cond_vec_uncond_batch(cond_model: torch.nn.Module, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return cond_model.uncond_batch(int(batch_size), device=device, dtype=dtype)


@torch.no_grad()
def eval_velocity_cfg(
    *,
    sit: torch.nn.Module,
    z: torch.Tensor,            # (B,C,H,W)
    t: torch.Tensor,            # (B,)
    cond_vec: torch.Tensor,     # (B,D)
    uncond_vec: torch.Tensor,   # (B,D)
    cfg_scale: float = 1.0,     # 1.0 = off; >1 stronger; 0 => unconditional
    amp: bool = True,
) -> torch.Tensor:
    if z.ndim != 4:
        raise ValueError(f"Expected z shape (B,C,H,W), got {z.shape}")
    if t.ndim != 1:
        t = t.view(-1)
    if t.shape[0] != z.shape[0]:
        t = t.expand(z.shape[0])

    use_amp = bool(amp) and z.device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        v_c = sit(z, t, cond_vec)
        if float(cfg_scale) == 1.0:
            return v_c
        v_u = sit(z, t, uncond_vec)
        return v_u + float(cfg_scale) * (v_c - v_u)
