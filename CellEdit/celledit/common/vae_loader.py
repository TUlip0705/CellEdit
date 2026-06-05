from __future__ import annotations

import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def extract_state_dict(ckpt_obj: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("model_state_dict"), dict):
        out: dict[str, torch.Tensor] = {}
        for key, value in ckpt_obj["model_state_dict"].items():
            clean_key = key[len("module.") :] if str(key).startswith("module.") else str(key)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Non-tensor checkpoint value for key: {key}")
            out[clean_key] = value
        return out
    raise RuntimeError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


def set_hf_offline_defaults() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _build_vae_model(
    *,
    model_type: str,
    device: torch.device,
    pretrained_vae: str | None = None,
) -> torch.nn.Module:
    if model_type == "hybrid":
        if pretrained_vae is None:
            raise ValueError("`pretrained_vae` is required for hybrid VAE.")
        set_hf_offline_defaults()
        from adapter_vae.models.hybrid_vae_24ch import HybridVAE24Ch

        pretrained_path = resolve_path(pretrained_vae)
        model = HybridVAE24Ch(
            pretrained_vae_path=str(pretrained_path),
            in_channels=6,
            out_channels=6,
            latent_channels=24,
            freeze_pretrained=False,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type} (expected 'hybrid')")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _clean_state_dict_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key[len("module.") :] if str(key).startswith("module.") else str(key)
        if isinstance(value, torch.Tensor):
            out[clean_key] = value
    return out


def load_vae_model(
    *,
    model_type: str,
    ckpt_path: str | Path,
    device: torch.device,
    pretrained_vae: str | None = None,
    strict: bool = True,
) -> torch.nn.Module:
    ckpt = torch.load(str(resolve_path(ckpt_path)), map_location="cpu")
    state = extract_state_dict(ckpt)
    model = _build_vae_model(model_type=str(model_type), device=device, pretrained_vae=pretrained_vae)
    missing, unexpected = model.load_state_dict(_clean_state_dict_prefix(state), strict=bool(strict))
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    return model
