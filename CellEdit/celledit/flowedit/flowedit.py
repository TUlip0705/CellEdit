from __future__ import annotations

"""
FlowEdit-style editing adapted to latent_flow2.

We operate on VAE latents z (shape [B,24,64,64]) and latent_flow2 time t in [0,1].

Key idea (mirrors FlowEdit):
  - Maintain a learnable/iterative estimate z_edit (at t=0 data end).
  - For a given time t, sample a shared noise eps and construct:
      z_src_t = (1-t) * z_src + t * eps
      z_tar_t = z_edit + z_src_t - z_src
    Then use delta-velocity v_tar - v_src to update z_edit.
  - Optionally switch to a "tail" phase (SDEdit-like) for the last n_min steps:
      initialize x_tar_t at the boundary and integrate using only target velocity.

Conventions (match latent_flow2 sampling scripts):
  - t grid decreases from 1 -> 0
  - Euler update: z <- z - dt * v
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from celledit.common.model_loader import eval_velocity_cfg
from celledit.common.time_grid import make_time_grid, step_dt


@dataclass(frozen=True)
class FlowEditConfig:
    steps: int = 150
    n_max: int = 120
    n_min: int = 5
    n_avg: int = 1

    cfg_src: float = 1.0
    cfg_tgt: float = 1.0

    snapshot_count: int = 0
    amp: bool = True


def _snapshot_schedule(total_updates: int, snapshot_count: int) -> set[int]:
    if snapshot_count <= 0:
        return set()
    if total_updates <= 0:
        return {0}
    if snapshot_count == 1:
        return {total_updates}
    # Capture roughly evenly spaced update numbers (1..total_updates)
    pts = torch.linspace(1, total_updates, steps=int(snapshot_count))
    return {int(round(float(x))) for x in pts.tolist()}


@torch.no_grad()
def flowedit_latent(
    *,
    sit: torch.nn.Module,
    z_src: torch.Tensor,          # (B,C,H,W) latent at data end
    cond_src: torch.Tensor,       # (B,D)
    cond_tgt: torch.Tensor,       # (B,D)
    uncond: torch.Tensor,         # (B,D)
    cfg: FlowEditConfig = FlowEditConfig(),
    seed: int = 0,
) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    if z_src.ndim != 4:
        raise ValueError(f"Expected z_src (B,C,H,W), got {z_src.shape}")
    device = z_src.device
    dtype = z_src.dtype
    b = int(z_src.shape[0])

    if cond_src.shape[0] != b or cond_tgt.shape[0] != b or uncond.shape[0] != b:
        raise ValueError("cond/uncond batch size mismatch with z_src")

    steps = int(cfg.steps)
    n_max = int(cfg.n_max)
    n_min = int(cfg.n_min)
    n_avg = max(1, int(cfg.n_avg))
    n_max = max(0, min(n_max, steps))
    n_min = max(0, min(n_min, steps))
    if n_max < n_min:
        raise ValueError(f"Invalid FlowEdit window: require n_max >= n_min, got n_max={n_max} n_min={n_min} (steps={steps})")

    # time grid: (steps+1,) from 1 -> 0
    t_grid = make_time_grid(steps, device=device, dtype=torch.float32)

    # Determine how many updates we will actually do (skip first steps-n_max).
    update_indices = [i for i in range(steps) if (steps - i) <= n_max]
    total_updates = len(update_indices)
    capture_updates = _snapshot_schedule(total_updates, int(cfg.snapshot_count))
    snapshots: Optional[List[torch.Tensor]] = [z_src.detach().clone()] if cfg.snapshot_count and cfg.snapshot_count > 0 else None

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    z_edit = z_src.clone()
    x_tar_t: Optional[torch.Tensor] = None
    update_no = 0

    for i in range(steps):
        t_i = t_grid[i]
        t_next = t_grid[i + 1]
        dt = step_dt(t_i, t_next)  # positive

        # skip early (high-noise) steps
        if (steps - i) > n_max:
            continue

        if (steps - i) > n_min:
            # delta-velocity phase
            v_delta = torch.zeros_like(z_edit, dtype=torch.float32)
            for _ in range(n_avg):
                # torch.randn_like() does not support `generator` in some torch versions
                eps = torch.randn(z_src.shape, generator=g, device=device, dtype=dtype)
                # linear path mixing weight = t
                z_src_t = (1.0 - t_i) * z_src + t_i * eps
                z_tgt_t = z_edit + z_src_t - z_src

                v_s = eval_velocity_cfg(
                    sit=sit,
                    z=z_src_t,
                    t=t_i.expand(b),
                    cond_vec=cond_src,
                    uncond_vec=uncond,
                    cfg_scale=float(cfg.cfg_src),
                    amp=bool(cfg.amp),
                )
                v_t = eval_velocity_cfg(
                    sit=sit,
                    z=z_tgt_t,
                    t=t_i.expand(b),
                    cond_vec=cond_tgt,
                    uncond_vec=uncond,
                    cfg_scale=float(cfg.cfg_tgt),
                    amp=bool(cfg.amp),
                )
                v_delta = v_delta + (v_t - v_s).float() / float(n_avg)

            z_edit = (z_edit.float() - dt * v_delta).to(dtype)
            update_no += 1
            if snapshots is not None and update_no in capture_updates:
                snapshots.append(z_edit.detach().clone())
        else:
            # tail phase: only target velocity
            if (steps - i) == n_min:
                eps = torch.randn(z_src.shape, generator=g, device=device, dtype=dtype)
                z_src_t = (1.0 - t_i) * z_src + t_i * eps
                x_tar_t = z_edit + z_src_t - z_src

            if x_tar_t is None:
                raise RuntimeError("Internal error: tail phase started without initialization")

            v_t = eval_velocity_cfg(
                sit=sit,
                z=x_tar_t,
                t=t_i.expand(b),
                cond_vec=cond_tgt,
                uncond_vec=uncond,
                cfg_scale=float(cfg.cfg_tgt),
                amp=bool(cfg.amp),
            )
            x_tar_t = (x_tar_t.float() - dt * v_t.float()).to(dtype)
            update_no += 1
            if snapshots is not None and update_no in capture_updates:
                snapshots.append(x_tar_t.detach().clone())

    z_final = z_edit if n_min == 0 else x_tar_t
    if z_final is None:
        raise RuntimeError("FlowEdit produced no final latent")
    if snapshots is not None and (len(snapshots) == 0 or snapshots[-1] is not z_final):
        snapshots.append(z_final.detach().clone())
    return z_final, snapshots
