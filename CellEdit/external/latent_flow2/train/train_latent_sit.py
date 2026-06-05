from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

SCRIPT_DIR = Path(__file__).resolve()
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from latent_flow2.data.dataset import (
    CachedLatentsDataset,
    CachedLatentsDatasetConfig,
    RawImagesDataset,
    RawImagesDatasetConfig,
)
from latent_flow2.losses.flow_matching import flow_matching_linear_path, mse_loss, sample_time_uniform
from latent_flow2.losses.repa import cosine_repa_loss, spatial_zscore
from latent_flow2.models.cond import (
    build_condition_encoder,
    canonicalize_dose_mode,
    condition_forward,
)
from latent_flow2.models.sit import LatentSiT, LatentSiTConfig
from latent_flow2.utils.io import write_json


class CheckpointSaveError(RuntimeError):
    """Checkpoint save failed."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CellEdit latent SiT models with (i)REPA / REPA-E.")
    p.add_argument("--mode", type=str, required=True, choices=["repa", "irepa", "repae", "irepae"])
    p.add_argument("--config", type=str, required=True, help="YAML config path, e.g. CellEdit/configs/rxrx3.yaml")
    p.add_argument("--local_rank", type=int, default=-1, help="DDP local rank (torchrun sets LOCAL_RANK)")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return p.parse_args()


def setup_ddp(local_rank: int) -> tuple[int, int, int]:
    if local_rank == -1:
        env_lr = os.environ.get("LOCAL_RANK", None)
        if env_lr is not None:
            try:
                local_rank = int(env_lr)
            except ValueError:
                local_rank = -1
    if local_rank == -1:
        return -1, 0, 1

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return int(local_rank), int(rank), int(world_size)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return int(rank) == 0


def set_seed(seed: int, rank: int) -> None:
    torch.manual_seed(int(seed) + int(rank))
    torch.cuda.manual_seed_all(int(seed) + int(rank))


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for k, p in model_params.items():
        if k in ema_params:
            ema_params[k].mul_(decay).add_(p.data, alpha=1 - decay)
    # buffers (e.g. BN stats)
    ema_bufs = dict(ema_model.named_buffers())
    model_bufs = dict(model.named_buffers())
    for k, b in model_bufs.items():
        if k in ema_bufs and b.dtype.is_floating_point:
            ema_bufs[k].mul_(decay).add_(b.data, alpha=1 - decay)


def append_metrics_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML root (expected dict): {path}")
    return cfg


def _human_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(value)} {units[unit_idx]}"
    return f"{value:.2f} {units[unit_idx]}"


def _estimate_object_nbytes(obj: Any) -> int:
    seen: set[int] = set()

    def _walk(x: Any) -> int:
        obj_id = id(x)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if torch.is_tensor(x):
            return int(x.numel()) * int(x.element_size())
        if isinstance(x, dict):
            return sum(_walk(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return sum(_walk(v) for v in x)
        return 0

    return _walk(obj)


def _safe_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    estimate = _estimate_object_nbytes(obj)
    required = int(estimate * 1.05) + (128 * 1024 * 1024)
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes < required:
        raise CheckpointSaveError(
            "Insufficient free space for checkpoint save: "
            f"path={path}, free={_human_bytes(free_bytes)}, "
            f"estimated_required={_human_bytes(required)} "
            f"(payload={_human_bytes(estimate)})."
        )

    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(path)
    except (OSError, RuntimeError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise CheckpointSaveError(
            f"Failed to save checkpoint to {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _resolve_checkpoint_dir(cfg: dict[str, Any], output_dir: Path) -> Path:
    paths_cfg = cfg.get("paths", {})
    ckpt_dir = paths_cfg.get("checkpoint_dir", None)
    if ckpt_dir:
        return Path(str(ckpt_dir))
    return output_dir / "checkpoints"


def to_device(batch: dict[str, Any], device: torch.device, *, dtype: torch.dtype) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=dtype, non_blocking=True)
            else:
                out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def toggle_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(bool(requires_grad))


def calculate_adaptive_weight(
    nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor, *, max_val: float = 1.0
) -> torch.Tensor:
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True, create_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True, create_graph=True)[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    return torch.clamp(d_weight, 0.0, float(max_val)).detach()


def get_vae_last_layer(model: torch.nn.Module) -> torch.Tensor:
    mod = model.module if hasattr(model, "module") else model
    return mod.decoder.conv_out.weight


def build_models(cfg: dict[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    cond_cfg = cfg.get("cond", {})
    model_cfg = cfg.get("model", {})

    cond, _drug_mode, _dose_mode = build_condition_encoder(cond_cfg)

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

    ema = LatentSiT(sit.cfg)
    ema.load_state_dict(sit.state_dict(), strict=True)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()

    return cond, sit, ema


def save_checkpoint(
    path: Path,
    *,
    step: int,
    epoch: int,
    cond: torch.nn.Module,
    sit: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "cond_state": cond.state_dict(),
        "sit_state": sit.state_dict(),
        "ema_state": ema.state_dict(),
        "opt_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    _safe_torch_save(ckpt, path)


def load_checkpoint(path: str | Path, cond: torch.nn.Module, sit: torch.nn.Module, ema: torch.nn.Module, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    cond.load_state_dict(ckpt["cond_state"], strict=True)
    sit.load_state_dict(ckpt["sit_state"], strict=True)
    ema.load_state_dict(ckpt.get("ema_state", ckpt["sit_state"]), strict=True)
    optimizer.load_state_dict(ckpt["opt_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def save_checkpoint_repae(
    path: Path,
    *,
    step: int,
    epoch: int,
    cond: torch.nn.Module,
    sit: torch.nn.Module,
    ema: torch.nn.Module,
    vae: torch.nn.Module,
    disc: torch.nn.Module | None,
    opt_sit: torch.optim.Optimizer,
    opt_vae: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "cond_state": cond.state_dict(),
        "sit_state": sit.state_dict(),
        "ema_state": ema.state_dict(),
        "vae_state": vae.state_dict(),
        "disc_state": disc.state_dict() if disc is not None else None,
        "opt_sit_state": opt_sit.state_dict(),
        "opt_vae_state": opt_vae.state_dict(),
        "opt_disc_state": opt_disc.state_dict() if opt_disc is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    _safe_torch_save(ckpt, path)


def load_checkpoint_repae(
    path: str | Path,
    *,
    cond: torch.nn.Module,
    sit: torch.nn.Module,
    ema: torch.nn.Module,
    vae: torch.nn.Module,
    disc: torch.nn.Module | None,
    opt_sit: torch.optim.Optimizer,
    opt_vae: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    cond.load_state_dict(ckpt["cond_state"], strict=True)
    sit.load_state_dict(ckpt["sit_state"], strict=True)
    ema.load_state_dict(ckpt.get("ema_state", ckpt["sit_state"]), strict=True)
    vae.load_state_dict(ckpt["vae_state"], strict=True)
    if disc is not None and ckpt.get("disc_state") is not None:
        disc.load_state_dict(ckpt["disc_state"], strict=True)
    opt_sit.load_state_dict(ckpt["opt_sit_state"])
    opt_vae.load_state_dict(ckpt["opt_vae_state"])
    if opt_disc is not None and ckpt.get("opt_disc_state") is not None:
        opt_disc.load_state_dict(ckpt["opt_disc_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def train_cached(cfg: dict[str, Any], mode: str, *, local_rank: int, rank: int, world_size: int) -> None:
    device_str = str(cfg.get("device", "cuda"))
    device = torch.device(device_str if local_rank == -1 else f"cuda:{local_rank}")
    mixed = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = torch.float16

    paths = cfg.get("paths", {})
    output_dir = Path(paths.get("output_dir", f"latent_flow2/outputs/{mode}"))
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "config_resolved.json", cfg)

    ds_cfg = CachedLatentsDatasetConfig(
        meta_csv=str(paths["meta_csv"]),
        split=str(cfg.get("data", {}).get("split", "train")),
        oph_tokens_dir=str(paths["oph_tokens_dir"]),
        kpgt_dir=str(paths["kpgt_dir"]),
        vae_latents_dir=str(paths["vae_latents_dir"]),
        max_rows=int(cfg.get("data", {}).get("max_rows", 0)),
    )
    dataset = CachedLatentsDataset(ds_cfg)

    cond_cfg = cfg.get("cond", {})
    drug_mode = str(cond_cfg.get("drug_mode", "kpgt"))
    dose_mode = canonicalize_dose_mode(cond_cfg.get("dose_mode", "bins"))
    if dose_mode == "cont_film":
        non_empty = dataset.meta.empty_id == 0
        if non_empty.any():
            # If dose_cont column is missing in an older cache, our loader fills zeros.
            # Fail fast to avoid silently training with a constant dose signal.
            if float(dataset.meta.dose_cont[non_empty].std()) < 1e-8:
                raise SystemExit(
                    "cond.dose_mode=cont_film requires a non-constant `dose_cont` in metadata.csv. "
                    "Rebuild meta cache with the updated build_meta_cache.py."
                )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("data", {}).get("batch_size", 8)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=bool(cfg.get("data", {}).get("pin_memory", True)),
        drop_last=True,
    )

    cond, sit, ema = build_models(cfg)
    cond = cond.to(device)
    sit = sit.to(device)
    ema = ema.to(device)

    if world_size > 1:
        cond = DDP(cond, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        sit = DDP(sit, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optim_cfg = cfg.get("optim", {})
    optimizer = AdamW(
        list(cond.parameters()) + list(sit.parameters()),
        lr=float(optim_cfg.get("lr", 1e-4)),
        betas=tuple(float(x) for x in optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=mixed)

    step = 0
    epoch = 0
    if cfg.get("seed") is not None:
        set_seed(int(cfg["seed"]), rank)

    resume_path = cfg.get("resume") or cfg.get("train", {}).get("resume")
    if resume_path:
        step, epoch = load_checkpoint(
            resume_path,
            cond.module if hasattr(cond, "module") else cond,
            sit.module if hasattr(sit, "module") else sit,
            ema,
            optimizer,
            scaler,
        )

    max_steps = int(cfg.get("train", {}).get("max_steps", 200_000))
    log_every = int(cfg.get("train", {}).get("log_every", 50))
    ckpt_every = int(cfg.get("train", {}).get("ckpt_every", 5000))

    loss_cfg = cfg.get("loss", {})
    flow_w = float(loss_cfg.get("flow_weight", 1.0))
    repa_w = float(loss_cfg.get("repa_weight", 0.5))

    irepa_cfg = cfg.get("irepa", {})
    spnorm_method = str(irepa_cfg.get("spnorm", "none")).lower()
    zscore_alpha = float(irepa_cfg.get("zscore_alpha", 1.0))
    zscore_eps = float(irepa_cfg.get("eps", 1e-6))

    metrics_path = output_dir / "metrics.csv"
    fields = ["step", "epoch", "loss", "flow_loss", "repa_loss", "lr", "sec_per_step"]

    sit_train = sit
    cond_train = cond
    sit_mod = sit.module if hasattr(sit, "module") else sit
    cond_mod = cond.module if hasattr(cond, "module") else cond

    it = iter(loader)
    if is_main_process(rank):
        pbar = tqdm(total=max_steps, initial=step, desc=f"train({mode})")
    else:
        pbar = None

    t0 = time.time()
    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(loader)
            batch = next(it)

        batch = to_device(batch, device, dtype=torch.float32)

        drug_fp = batch["kpgt_fp"]
        treatment_id = batch["treatment_id"].long()
        empty_id = batch["empty_id"].long()
        if dose_mode == "cont_film":
            dose_cont = batch["dose_cont"].float()
        else:
            dose_id = batch["dose_id"].long()
        z_mean = batch["z_mean"]
        z_std = batch["z_std"]
        teacher = batch["oph_tokens"]

        # sample data latents
        eps = torch.randn_like(z_mean)
        z_data = z_mean + z_std * eps

        # time + path
        t = sample_time_uniform(z_data.shape[0], device=device, dtype=z_data.dtype)
        z_t, v_target, _ = flow_matching_linear_path(z_data, t)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=mixed, dtype=amp_dtype):
            cond_vec = condition_forward(
                cond_train,
                drug_mode=drug_mode,
                dose_mode=dose_mode,
                drug_fp=drug_fp,
                treatment_id=treatment_id,
                dose_id=dose_id if dose_mode == "bins" else None,
                dose_cont=dose_cont if dose_mode == "cont_film" else None,
                empty_id=empty_id,
            )
            v_hat, proj = sit_train(z_t, t, cond_vec, return_proj=True)

            flow_loss = mse_loss(v_hat, v_target)

            teacher_use = teacher
            if mode == "irepa" and spnorm_method == "zscore":
                teacher_use = spatial_zscore(teacher_use, alpha=zscore_alpha, eps=zscore_eps)
            repa_loss = cosine_repa_loss(teacher_use, proj)

            loss = flow_w * flow_loss + repa_w * repa_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # EMA update on main process (model params are synced after step in DDP)
        if is_main_process(rank):
            update_ema(ema, sit_mod, decay=0.999)

        step += 1
        if pbar is not None:
            pbar.update(1)

        if step % log_every == 0 and is_main_process(rank):
            dt = time.time() - t0
            t0 = time.time()
            lr = float(optimizer.param_groups[0]["lr"])
            append_metrics_csv(
                metrics_path,
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "flow_loss": float(flow_loss.detach().cpu()),
                    "repa_loss": float(repa_loss.detach().cpu()),
                    "lr": lr,
                    "sec_per_step": dt / log_every,
                },
                fieldnames=fields,
            )
            if pbar is not None:
                pbar.set_postfix({"loss": float(loss.detach().cpu()), "flow": float(flow_loss.detach().cpu()), "repa": float(repa_loss.detach().cpu())})

        if step % ckpt_every == 0 and is_main_process(rank):
            ckpt_path = output_dir / "checkpoints" / f"step_{step:07d}.pt"
            save_checkpoint(
                ckpt_path,
                step=step,
                epoch=epoch,
                cond=cond_mod,
                sit=sit_mod,
                ema=ema,
                optimizer=optimizer,
                scaler=scaler,
                extra={"mode": mode},
            )

    if pbar is not None:
        pbar.close()


def train_repae(cfg: dict[str, Any], mode: str, *, local_rank: int, rank: int, world_size: int) -> None:
    # Minimal REPA-E loop (no GAN/LPIPS): VAE recon+KL + alignment; SiT diffusion+alignment; detach z for SiT update.
    mode = str(mode).strip().lower()
    device_str = str(cfg.get("device", "cuda"))
    device = torch.device(device_str if local_rank == -1 else f"cuda:{local_rank}")
    mixed = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = torch.float16

    paths = cfg.get("paths", {})
    output_dir = Path(paths.get("output_dir", f"latent_flow2/outputs/{mode}"))
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "config_resolved.json", cfg)

    # Load VAE run config for preprocessing + model args.
    vae_cfg_path = Path(cfg.get("vae", {}).get("config", ""))
    vae_ckpt = Path(cfg.get("vae", {}).get("ckpt", ""))
    if not vae_cfg_path.exists():
        raise SystemExit(f"VAE config not found: {vae_cfg_path}")
    if not vae_ckpt.exists():
        raise SystemExit(f"VAE ckpt not found: {vae_ckpt}")
    with open(vae_cfg_path, "r", encoding="utf-8") as f:
        vae_run_cfg = yaml.safe_load(f)
    fixed_stats_path = vae_run_cfg.get("fixed_stats_path", None)
    if not fixed_stats_path:
        raise SystemExit("VAE run config must contain fixed_stats_path")

    ds_cfg = RawImagesDatasetConfig(
        meta_csv=str(paths["meta_csv"]),
        split=str(cfg.get("data", {}).get("split", "train")),
        rxrx3_root=str(paths["rxrx3_root"]),
        oph_tokens_dir=str(paths["oph_tokens_dir"]),
        kpgt_dir=str(paths["kpgt_dir"]),
        max_rows=int(cfg.get("data", {}).get("max_rows", 0)),
        fixed_stats_path=str(fixed_stats_path),
        percentile_bounds=tuple(float(x) for x in str(vae_run_cfg.get("percentile_bounds", "0.5,99.5")).split(",")),
        percentile_per_channel=bool(vae_run_cfg.get("percentile_per_channel", True)),
        use_channel_standardize=bool(vae_run_cfg.get("use_channel_standardize", False)),
    )
    dataset = RawImagesDataset(ds_cfg)
    cond_cfg = cfg.get("cond", {})
    drug_mode = str(cond_cfg.get("drug_mode", "kpgt"))
    dose_mode = canonicalize_dose_mode(cond_cfg.get("dose_mode", "bins"))
    if dose_mode == "cont_film":
        non_empty = dataset.meta.empty_id == 0
        if non_empty.any():
            if float(dataset.meta.dose_cont[non_empty].std()) < 1e-8:
                raise SystemExit(
                    "cond.dose_mode=cont_film requires a non-constant `dose_cont` in metadata.csv. "
                    "Rebuild meta cache with the updated build_meta_cache.py."
                )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("data", {}).get("batch_size", 4)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=bool(cfg.get("data", {}).get("pin_memory", True)),
        drop_last=True,
    )

    cond, sit, ema = build_models(cfg)
    cond = cond.to(device)
    sit = sit.to(device)
    ema = ema.to(device)

    model_cfg = cfg.get("model", {})
    init_cfg = cfg.get("init", {})
    init_cached_ckpt = init_cfg.get("cached_ckpt", None) if isinstance(init_cfg, dict) else None

    try:
        from adapter_vae.models.hybrid_vae_24ch import HybridVAE24Ch
    except ImportError as e:
        raise RuntimeError(
            "Failed to import adapter_vae HybridVAE24Ch (diffusers missing). Install deps and retry."
        ) from e

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vae = HybridVAE24Ch(
        pretrained_vae_path=str(vae_run_cfg.get("pretrained_vae", "stabilityai/sd-vae-ft-mse")),
        in_channels=int(vae_run_cfg.get("in_channels", 6)),
        out_channels=int(vae_run_cfg.get("out_channels", 6)),
        latent_channels=int(vae_run_cfg.get("latent_channels", 24)),
        freeze_pretrained=bool(vae_run_cfg.get("freeze_pretrained", False)),
    ).to(device)

    ckpt = torch.load(str(vae_ckpt), map_location="cpu")
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model_state_dict"), dict):
        raise RuntimeError(f"Unrecognized VAE checkpoint format: {vae_ckpt}")
    state = ckpt["model_state_dict"]
    new_state = {}
    for k, v in state.items():
        kk = k[len("module.") :] if str(k).startswith("module.") else k
        new_state[kk] = v
    vae.load_state_dict(new_state, strict=True)

    resume_path = cfg.get("resume") or cfg.get("train", {}).get("resume")
    if init_cached_ckpt and not resume_path:
        init_path = Path(str(init_cached_ckpt))
        if not init_path.exists():
            raise SystemExit(f"init.cached_ckpt not found: {init_path}")
        init_ckpt = torch.load(str(init_path), map_location="cpu")
        cond.load_state_dict(init_ckpt["cond_state"], strict=True)
        sit.load_state_dict(init_ckpt["sit_state"], strict=True)
        ema.load_state_dict(init_ckpt.get("ema_state", init_ckpt["sit_state"]), strict=True)
        if is_main_process(rank):
            print(f"[init.cached_ckpt] initialized cond/sit/ema from {init_path}")

    # ---- REPA-E latent BN bridge (Pit 1) ----
    # REPA-E uses a BatchNorm layer between VAE latents and the diffusion/flow backbone,
    # applied to *clean* latents (z1) before noise mixing. For VAE updates, BN must run in
    # eval mode to avoid updating running stats; for SiT updates, BN runs in train mode.
    #
    # We initialize BN running stats from dataset latents for stable early training, mirroring
    # REPA-E's `*-latents-stats.pt` initialization when available.
    if getattr(sit, "input_bn", None) is not None and not resume_path:
        bn = sit.input_bn

        stats_path = model_cfg.get("bn_stats_path", None)
        stats_path = Path(str(stats_path)) if stats_path else vae_ckpt.with_name(f"{vae_ckpt.stem}-latents-stats.pt")

        latents_bias = None
        latents_scale = None
        if stats_path.exists():
            obj = torch.load(str(stats_path), map_location="cpu")
            if isinstance(obj, dict) and "latents_bias" in obj and "latents_scale" in obj:
                latents_bias = obj["latents_bias"]
                latents_scale = obj["latents_scale"]

        if latents_bias is not None and latents_scale is not None:
            latents_bias = latents_bias.detach().view(-1).to(device=device, dtype=torch.float32)
            latents_scale = latents_scale.detach().view(-1).to(device=device, dtype=torch.float32)
            if latents_bias.numel() != bn.running_mean.numel():
                raise SystemExit(
                    f"BN stats mismatch: latents_bias has C={latents_bias.numel()} but BN expects C={bn.running_mean.numel()}"
                )
            bn.running_mean.data.copy_(latents_bias)
            bn.running_var.data.copy_((1.0 / latents_scale).pow(2))
            if getattr(ema, "input_bn", None) is not None:
                ema.input_bn.running_mean.data.copy_(bn.running_mean.data)
                ema.input_bn.running_var.data.copy_(bn.running_var.data)
            if is_main_process(rank):
                print(f"[repae] Initialized latent BN from stats: {stats_path}")
        else:
            bn_init_batches = int(model_cfg.get("bn_init_batches", 50))
            if bn_init_batches > 0:
                if is_main_process(rank):
                    print(f"[repae] Estimating latent BN stats from {bn_init_batches} batches (no stats file found).")

                # Accumulate per-channel mean/var over z1 samples: shape (B,C,H,W).
                c = int(bn.running_mean.numel())
                sum_c = torch.zeros((c,), device=device, dtype=torch.float64)
                sumsq_c = torch.zeros((c,), device=device, dtype=torch.float64)
                count = torch.zeros((), device=device, dtype=torch.float64)

                vae_was_training = vae.training
                vae.eval()
                with torch.no_grad():
                    for i, b0 in enumerate(loader):
                        if i >= bn_init_batches:
                            break
                        b0 = to_device(b0, device, dtype=torch.float32)
                        img0 = b0["image"]
                        with torch.cuda.amp.autocast(enabled=mixed, dtype=amp_dtype):
                            dist0 = vae.encode(img0)
                            z0 = dist0.mean + dist0.std * torch.randn_like(dist0.mean)
                        z0 = z0.float()
                        sum_c += z0.double().sum(dim=(0, 2, 3))
                        sumsq_c += (z0.double() ** 2).sum(dim=(0, 2, 3))
                        count += float(z0.shape[0] * z0.shape[2] * z0.shape[3])

                if vae_was_training:
                    vae.train()

                if world_size > 1 and dist.is_initialized():
                    dist.all_reduce(sum_c, op=dist.ReduceOp.SUM)
                    dist.all_reduce(sumsq_c, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count, op=dist.ReduceOp.SUM)

                if float(count.item()) <= 0.0:
                    raise SystemExit("BN init failed: no samples collected for latent stats.")

                mean = (sum_c / count).to(dtype=torch.float32)
                var = (sumsq_c / count - mean.double() ** 2).to(dtype=torch.float32).clamp(min=1e-6)
                bn.running_mean.data.copy_(mean)
                bn.running_var.data.copy_(var)
                if getattr(ema, "input_bn", None) is not None:
                    ema.input_bn.running_mean.data.copy_(bn.running_mean.data)
                    ema.input_bn.running_var.data.copy_(bn.running_var.data)
                if is_main_process(rank):
                    print(
                        f"[repae] BN init done: mean[0]={float(mean[0].cpu()):.4f} std[0]={float(var[0].sqrt().cpu()):.4f}"
                    )

    # REPA-E uses SyncBN when distributed.
    if world_size > 1:
        sit = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sit)

    if world_size > 1:
        cond = DDP(cond, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        sit = DDP(sit, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optim_cfg = cfg.get("optim", {})
    loss_cfg = cfg.get("loss", {})
    flow_w = float(loss_cfg.get("flow_weight", 1.0))
    repa_w = float(loss_cfg.get("repa_weight", 0.5))
    vae_align_w = float(loss_cfg.get("vae_align_weight", repa_w))

    try:
        from adapter_vae.utils.losses import VAELoss
        from vae6.discriminators import PatchDiscriminator, hinge_d_loss, hinge_g_loss
    except ImportError as e:
        raise RuntimeError(
            "Missing adapter_vae/vae6 dependencies for REPA-E VAE regularization (LPIPS/GAN)."
        ) from e

    vae_rec_type = str(vae_run_cfg.get("rec_type", "l1")).lower()
    vae_rec_weight = float(vae_run_cfg.get("rec_weight", 1.0))
    vae_kl_target = float(vae_run_cfg.get("kl_weight", 1e-6))
    vae_kl_mode = str(vae_run_cfg.get("kl_mode", "sum")).lower()
    vae_lpips_weight = float(vae_run_cfg.get("lpips_weight", 0.0))
    vae_sobel_weight = float(vae_run_cfg.get("sobel_weight", 0.0))
    vae_use_lpips = bool(vae_run_cfg.get("use_lpips", True))
    vae_lpips_mode = str(vae_run_cfg.get("lpips_mode", "channel_group")).lower()

    lambda_gan = float(vae_run_cfg.get("lambda_gan", 0.0))
    disc_start_step = int(vae_run_cfg.get("disc_start_step", 0))
    disc_weight_max = float(vae_run_cfg.get("disc_weight_max", 1.0))
    disc_ndf = int(vae_run_cfg.get("disc_ndf", 128))
    disc_layers = int(vae_run_cfg.get("disc_layers", 4))
    disable_adaptive_weight = bool(vae_run_cfg.get("disable_adaptive_weight", False))

    warmup_epochs = int(vae_run_cfg.get("warmup_epochs", 0))
    kl_warmup_steps = int(vae_run_cfg.get("kl_warmup_steps", 0))
    if kl_warmup_steps <= 0 and warmup_epochs > 0:
        kl_warmup_steps = int(warmup_epochs) * int(len(loader))
    kl_warmup_start_step = int(vae_run_cfg.get("kl_warmup_start_step", 0))

    vae_loss_fn = VAELoss(
        rec_type=vae_rec_type,
        rec_weight=vae_rec_weight,
        kl_weight=vae_kl_target,
        kl_mode=vae_kl_mode,
        lpips_weight=vae_lpips_weight,
        sobel_weight=vae_sobel_weight,
        use_lpips=vae_use_lpips,
        lpips_mode=vae_lpips_mode,
        device=device,
    ).to(device)

    disc_img = None
    opt_disc_img = None
    if lambda_gan > 0.0:
        disc_img = PatchDiscriminator(
            in_channels=int(vae_run_cfg.get("in_channels", 6)),
            ndf=disc_ndf,
            n_layers=disc_layers,
        ).to(device)

        if not resume_path and isinstance(ckpt, dict):
            disc_state_src = ckpt.get("disc_state_dict", None) or ckpt.get("disc_state", None)
            if disc_state_src is None:
                raise RuntimeError(f"VAE checkpoint has no discriminator state: {vae_ckpt}")
            disc_state = {}
            for k, v in disc_state_src.items():
                kk = k[len("module.") :] if str(k).startswith("module.") else k
                disc_state[kk] = v
            disc_img.load_state_dict(disc_state, strict=True)
            if is_main_process(rank):
                print(f"[repae] Initialized discriminator from VAE ckpt: {vae_ckpt}")

        if world_size > 1:
            disc_img = DDP(disc_img, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        disc_lr_cfg = vae_run_cfg.get("disc_lr", None)
        disc_lr = float(disc_lr_cfg) if disc_lr_cfg is not None else float(vae_run_cfg.get("lr", optim_cfg.get("lr", 1e-4)))
        opt_disc_img = AdamW(
            disc_img.parameters(),
            lr=disc_lr,
            betas=tuple(float(x) for x in optim_cfg.get("betas", [0.9, 0.999])),
            weight_decay=float(vae_run_cfg.get("weight_decay", optim_cfg.get("weight_decay", 0.01))),
        )
        if not resume_path and isinstance(ckpt, dict):
            disc_opt_state = ckpt.get("disc_optimizer_state_dict", None) or ckpt.get("disc_opt_state", None) or ckpt.get("opt_disc_state", None)
            if disc_opt_state is not None:
                opt_disc_img.load_state_dict(disc_opt_state)
                for pg in opt_disc_img.param_groups:
                    pg["lr"] = float(disc_lr)
                    pg["weight_decay"] = float(vae_run_cfg.get("weight_decay", optim_cfg.get("weight_decay", 0.01)))
                if is_main_process(rank):
                    print(f"[repae] Initialized discriminator optimizer from VAE ckpt: {vae_ckpt}")

    opt_sit = AdamW(
        list(cond.parameters()) + list(sit.parameters()),
        lr=float(optim_cfg.get("lr", 1e-4)),
        betas=tuple(float(x) for x in optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
    )
    opt_vae = AdamW(
        vae.parameters(),
        lr=float(optim_cfg.get("vae_lr", optim_cfg.get("lr", 1e-4))),
        betas=tuple(float(x) for x in optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(vae_run_cfg.get("weight_decay", optim_cfg.get("weight_decay", 0.01))),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=mixed)

    sit_mod = sit.module if hasattr(sit, "module") else sit
    cond_mod = cond.module if hasattr(cond, "module") else cond
    vae_mod = vae.module if hasattr(vae, "module") else vae
    vae_last_layer = get_vae_last_layer(vae)

    max_steps = int(cfg.get("train", {}).get("max_steps", 200_000))
    log_every = int(cfg.get("train", {}).get("log_every", 50))
    ckpt_every = int(cfg.get("train", {}).get("ckpt_every", 5000))

    metrics_path = output_dir / "metrics.csv"
    fields = [
        "step",
        "epoch",
        "sit_loss",
        "vae_loss",
        "flow_loss",
        "repa_loss",
        "vae_nll",
        "vae_rec",
        "vae_kl",
        "vae_lpips",
        "vae_sobel",
        "vae_align",
        "gan_g",
        "gan_d",
        "d_weight",
        "kl_w",
    ]

    step = 0
    epoch = 0
    if cfg.get("seed") is not None:
        set_seed(int(cfg["seed"]), rank)

    if resume_path:
        step, epoch = load_checkpoint_repae(
            resume_path,
            cond=cond_mod,
            sit=sit_mod,
            ema=ema,
            vae=vae_mod,
            disc=disc_img.module if hasattr(disc_img, "module") else disc_img,
            opt_sit=opt_sit,
            opt_vae=opt_vae,
            opt_disc=opt_disc_img,
            scaler=scaler,
        )

    it = iter(loader)
    if is_main_process(rank):
        pbar = tqdm(total=max_steps, initial=step, desc=f"train({mode})")
    else:
        pbar = None
    t0 = time.time()

    irepa_cfg = cfg.get("irepa", {})
    spnorm_method = str(irepa_cfg.get("spnorm", "none")).lower()
    zscore_alpha = float(irepa_cfg.get("zscore_alpha", 1.0))
    zscore_eps = float(irepa_cfg.get("eps", 1e-6))

    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(loader)
            batch = next(it)

        batch = to_device(batch, device, dtype=torch.float32)
        image = batch["image"]
        teacher = batch["oph_tokens"]
        teacher_use = teacher
        if mode == "irepae" and spnorm_method == "zscore":
            teacher_use = spatial_zscore(teacher_use, alpha=zscore_alpha, eps=zscore_eps)
        drug_fp = batch["kpgt_fp"]
        treatment_id = batch["treatment_id"].long()
        empty_id = batch["empty_id"].long()
        if dose_mode == "cont_film":
            dose_cont = batch["dose_cont"].float()
        else:
            dose_id = batch["dose_id"].long()

        # KL warmup (match adapter_vae)
        kl_w = float(vae_kl_target)
        if int(kl_warmup_steps) > 0 and kl_w > 0.0:
            progress_step = float(step - int(kl_warmup_start_step))
            warm = 0.0 if progress_step <= 0 else min(1.0, progress_step / float(kl_warmup_steps))
            kl_w = kl_w * warm

        gan_factor = 0.0
        if disc_img is not None and float(lambda_gan) > 0.0 and int(step) >= int(disc_start_step):
            gan_factor = 1.0

        # VAE forward (DDP-safe): recon + posterior stats + regularization losses
        with torch.cuda.amp.autocast(enabled=mixed, dtype=amp_dtype):
            x_rec, dist_post = vae(image)
            mean = dist_post.mean
            logvar = getattr(dist_post, "logvar", None)
            std = dist_post.std if logvar is None else torch.exp(0.5 * logvar)
            z1 = mean + std * torch.randn_like(mean)
            vae_nll, vae_comps = vae_loss_fn(
                image,
                x_rec,
                dist_post,
                return_components=True,
                return_tensors=True,
                kl_weight_override=kl_w,
            )

        # Sample time/noise once and reuse (match REPA-E: reuse the same `t` and `noises` for VAE and SiT updates)
        z1 = z1.float()
        t = sample_time_uniform(z1.shape[0], device=device, dtype=torch.float32)
        z_noise = torch.randn_like(z1)

        # ---- VAE update (freeze cond+sit; VAE gets: nll + GAN(G) + align) ----
        toggle_grad(sit, False)
        toggle_grad(cond, False)
        sit.eval()
        cond.eval()

        d_weight = torch.tensor(0.0, device=device)
        loss_gan_g = torch.tensor(0.0, device=device)
        loss_gan_d = torch.tensor(0.0, device=device)

        opt_vae.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=mixed, dtype=amp_dtype):
            # recompute cond without dropout (eval mode)
            cond_vec_eval = condition_forward(
                cond,
                drug_mode=drug_mode,
                dose_mode=dose_mode,
                drug_fp=drug_fp,
                treatment_id=treatment_id,
                dose_id=dose_id if dose_mode == "bins" else None,
                dose_cont=dose_cont if dose_mode == "cont_film" else None,
                empty_id=empty_id,
                apply_cfg_dropout=False,
            )
            apply_input_bn = bool(getattr(sit_mod, "input_bn", None) is None)
            z1_norm_eval = sit_mod.input_bn(z1) if not apply_input_bn else z1
            z_t_eval = (1.0 - t.view(-1, 1, 1, 1)) * z1_norm_eval + t.view(-1, 1, 1, 1) * z_noise
            proj = sit(z_t_eval, t, cond_vec_eval, return_proj=True, align_only=True, apply_input_bn=apply_input_bn)
            vae_align = cosine_repa_loss(teacher_use, proj)
            vae_loss = vae_nll + vae_align_w * vae_align

        if disc_img is not None and opt_disc_img is not None and gan_factor > 0.0:
            toggle_grad(disc_img, False)
            disc_in = x_rec
            if mixed:
                disc_in = disc_in.float()
            logits_fake = disc_img(disc_in)
            loss_gan_g = hinge_g_loss(logits_fake)
            if disable_adaptive_weight:
                d_weight = torch.tensor(1.0, device=device)
            else:
                nll_for_weight = (
                    vae_comps["rec"]
                    + float(vae_lpips_weight) * vae_comps["lpips"]
                    + float(vae_sobel_weight) * vae_comps["sobel"]
                )
                d_weight = calculate_adaptive_weight(nll_for_weight, loss_gan_g, vae_last_layer, max_val=disc_weight_max)
            vae_loss = vae_loss + float(lambda_gan) * float(gan_factor) * d_weight * loss_gan_g

        scaler.scale(vae_loss).backward()
        scaler.step(opt_vae)

        # Discriminator update
        if disc_img is not None and opt_disc_img is not None and gan_factor > 0.0:
            toggle_grad(disc_img, True)
            disc_real = image
            disc_fake = x_rec.detach()
            if mixed:
                disc_real = disc_real.float()
                disc_fake = disc_fake.float()
            logits_real = disc_img(disc_real)
            logits_fake = disc_img(disc_fake)
            loss_d = hinge_d_loss(logits_real, logits_fake)
            opt_disc_img.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_disc_img.step()
            loss_gan_d = loss_d.detach()

        # ---- SiT update (detach z1): SiT gets flow + align; no VAE grads ----
        toggle_grad(sit, True)
        toggle_grad(cond, True)
        sit.train()
        cond.train()

        opt_sit.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=mixed, dtype=amp_dtype):
            z1_det = z1.detach()
            apply_input_bn = bool(getattr(sit_mod, "input_bn", None) is None)
            z1_norm_det = sit_mod.input_bn(z1_det) if not apply_input_bn else z1_det
            z_t_det = (1.0 - t.view(-1, 1, 1, 1)) * z1_norm_det + t.view(-1, 1, 1, 1) * z_noise
            v_target_det = z_noise - z1_norm_det
            cond_vec = condition_forward(
                cond,
                drug_mode=drug_mode,
                dose_mode=dose_mode,
                drug_fp=drug_fp,
                treatment_id=treatment_id,
                dose_id=dose_id if dose_mode == "bins" else None,
                dose_cont=dose_cont if dose_mode == "cont_film" else None,
                empty_id=empty_id,
            )
            v_hat, proj = sit(z_t_det, t, cond_vec, return_proj=True, apply_input_bn=apply_input_bn)
            flow_loss = mse_loss(v_hat, v_target_det)
            repa_loss = cosine_repa_loss(teacher_use, proj)
            sit_loss = flow_w * flow_loss + repa_w * repa_loss
        scaler.scale(sit_loss).backward()
        scaler.step(opt_sit)
        scaler.update()

        if is_main_process(rank):
            update_ema(ema, sit_mod, decay=0.999)

        step += 1
        if pbar is not None:
            pbar.update(1)

        if step % log_every == 0 and is_main_process(rank):
            dt = time.time() - t0
            t0 = time.time()
            append_metrics_csv(
                metrics_path,
                {
                    "step": step,
                    "epoch": epoch,
                    "sit_loss": float(sit_loss.detach().cpu()),
                    "vae_loss": float(vae_loss.detach().cpu()),
                    "flow_loss": float(flow_loss.detach().cpu()),
                    "repa_loss": float(repa_loss.detach().cpu()),
                    "vae_nll": float(vae_nll.detach().cpu()),
                    "vae_rec": float(vae_comps["rec"].detach().cpu()),
                    "vae_kl": float(vae_comps["kl"].detach().cpu()),
                    "vae_lpips": float(vae_comps["lpips"].detach().cpu()),
                    "vae_sobel": float(vae_comps["sobel"].detach().cpu()),
                    "vae_align": float(vae_align.detach().cpu()),
                    "gan_g": float(loss_gan_g.detach().cpu()),
                    "gan_d": float(loss_gan_d.detach().cpu()),
                    "d_weight": float(d_weight.detach().cpu()),
                    "kl_w": float(kl_w),
                },
                fieldnames=fields,
            )
            if pbar is not None:
                pbar.set_postfix(
                    {
                        "sit": float(sit_loss.detach().cpu()),
                        "vae": float(vae_loss.detach().cpu()),
                        "align": float(vae_align.detach().cpu()),
                        "ganG": float(loss_gan_g.detach().cpu()),
                        "sec/step": dt / log_every,
                    }
                )

        if step % ckpt_every == 0 and is_main_process(rank):
            ckpt_path = output_dir / "checkpoints" / f"step_{step:07d}.pt"
            save_checkpoint_repae(
                ckpt_path,
                step=step,
                epoch=epoch,
                cond=cond_mod,
                sit=sit_mod,
                ema=ema,
                vae=vae_mod,
                disc=disc_img.module if hasattr(disc_img, "module") else disc_img,
                opt_sit=opt_sit,
                opt_vae=opt_vae,
                opt_disc=opt_disc_img,
                scaler=scaler,
                extra={"mode": mode},
            )

    if pbar is not None:
        pbar.close()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg["mode"] = str(args.mode)
    cfg["resume"] = args.resume

    local_rank, rank, world_size = setup_ddp(args.local_rank)
    try:
        if args.mode in ("repa", "irepa"):
            train_cached(cfg, args.mode, local_rank=local_rank, rank=rank, world_size=world_size)
        else:
            train_repae(cfg, args.mode, local_rank=local_rank, rank=rank, world_size=world_size)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
