from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image

from celledit.common.image_utils import to_rgb


def sanitize_filename(name: str) -> str:
    return str(name).strip().replace("/", "-").replace(" ", "_").replace(":", "-")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _to_uint8_gray(x01: torch.Tensor) -> np.ndarray:
    arr = (x01.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _to_01(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    xmin = float(x.min())
    xmax = float(x.max())
    if xmin >= 0.0 - 1e-6 and xmax <= 1.0 + 1e-6:
        return torch.clamp(x, 0.0, 1.0)
    if xmin >= -1.0 - 1e-6 and xmax <= 1.0 + 1e-6:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    return torch.clamp(x, 0.0, 1.0)


def save_gray_channels(
    x: torch.Tensor,
    *,
    out_dir: Path,
    prefix: str,
    also_grid: bool = True,
) -> None:
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected (6,H,W) or (1,6,H,W), got {x.shape}")
    if x.shape[0] < 6:
        raise ValueError(f"Expected >=6 channels, got {x.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    x = x[:6].float()
    x01 = _to_01(x)

    tiles: list[np.ndarray] = []
    for ch in range(6):
        u8 = _to_uint8_gray(x01[ch])
        tiles.append(u8)
        Image.fromarray(u8, mode="L").save(out_dir / f"{prefix}_ch{ch}.png")

    if also_grid:
        grid = make_grid_gray(tiles, rows=1, cols=6, pad=2, outer_pad=2, force_full_range=True)
        grid.save(out_dir / f"{prefix}_6ch_grid.png")


def save_rgb_preview(
    x: torch.Tensor,
    *,
    out_path: Path,
) -> None:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected (B,6,H,W), got {x.shape}")
    x = x[:, :6].float()
    x01 = _to_01(x)

    rgb = to_rgb(x01, dtype=torch.float32)[0]  # (3,H,W) in [0,1]
    rgb_u8 = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8, mode="RGB").save(out_path)


def make_grid_gray(
    tiles: list[np.ndarray],
    *,
    rows: int,
    cols: int,
    pad: int = 2,
    outer_pad: int = 2,
    force_full_range: bool = True,
) -> Image.Image:
    if not tiles:
        raise ValueError("no tiles")
    h, w = tiles[0].shape
    grid_h = rows * h + (rows - 1) * pad + 2 * outer_pad
    grid_w = cols * w + (cols - 1) * pad + 2 * outer_pad
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, img in enumerate(tiles):
        r = i // cols
        c = i % cols
        y = outer_pad + r * (h + pad)
        x = outer_pad + c * (w + pad)
        grid[y : y + h, x : x + w] = img
    if force_full_range and grid.size >= 2:
        grid[0, 0] = 0
        grid[0, 1] = 255
    return Image.fromarray(grid, mode="L")


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class SaveSpec:
    save_npy: bool = True
    save_source_npy: bool = True
    save_target_npy: bool = True
    save_edited_npy: bool = True
    save_gray: bool = True
    save_rgb: bool = True


def save_triplet(
    *,
    out_dir: str | Path,
    source: torch.Tensor,
    target_real: torch.Tensor,
    edited: torch.Tensor,
    meta: dict[str, Any],
    spec: SaveSpec = SaveSpec(),
) -> Path:
    out_dir = ensure_dir(out_dir)

    def _save_npy(name: str, x: torch.Tensor) -> None:
        arr = x.detach().cpu().float().numpy()
        np.save(out_dir / f"{name}.npy", arr)

    if spec.save_npy:
        if spec.save_source_npy:
            _save_npy("source_empty_6ch", source)
        if spec.save_target_npy:
            _save_npy("target_real_6ch", target_real)
        if spec.save_edited_npy:
            _save_npy("edited_6ch", edited)

    if spec.save_gray:
        save_gray_channels(source, out_dir=out_dir, prefix="source_empty", also_grid=True)
        save_gray_channels(target_real, out_dir=out_dir, prefix="target_real", also_grid=True)
        save_gray_channels(edited, out_dir=out_dir, prefix="edited", also_grid=True)

        tiles = []
        for x in (source, target_real, edited):
            if x.ndim == 4:
                xx = x[0]
            else:
                xx = x
            xx = xx[:6].float()
            xx01 = torch.clamp((xx + 1.0) / 2.0, 0.0, 1.0) if float(xx.min()) < 0 else torch.clamp(xx, 0.0, 1.0)
            for ch in range(6):
                tiles.append(_to_uint8_gray(xx01[ch]))
        grid = make_grid_gray(tiles, rows=3, cols=6, pad=2, outer_pad=2, force_full_range=True)
        grid.save(out_dir / "compare_6ch_grid.png")

    if spec.save_rgb:
        save_rgb_preview(source, out_path=out_dir / "source_empty_rgb.png")
        save_rgb_preview(target_real, out_path=out_dir / "target_real_rgb.png")
        save_rgb_preview(edited, out_path=out_dir / "edited_rgb.png")

    write_json(out_dir / "meta.json", meta)
    return out_dir
