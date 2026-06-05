from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_flow2.data.kpgt import load_kpgt_cache
from latent_flow2.data.meta import MetaTable, load_meta_csv
from latent_flow2.data.oph_tokens import OpenPhenomTokens
from latent_flow2.data.vae_latents import VAELatentsCache


@dataclass(frozen=True)
class CachedLatentsDatasetConfig:
    meta_csv: str
    split: str
    oph_tokens_dir: str
    kpgt_dir: str
    vae_latents_dir: str
    max_rows: int = 0


class CachedLatentsDataset(Dataset):
    """
    Training dataset for REPA/iREPA that reads:
      - metadata.csv (latent_flow2/cache/meta/metadata.csv)
      - VAE latent mean/std cache (.npy)
      - OpenPhenom patch tokens cache (.npy)
      - KPGT table cache (.npy)

    Returns CPU tensors; the trainer moves them to GPU.
    """

    def __init__(self, cfg: CachedLatentsDatasetConfig):
        self.cfg = cfg
        self.meta: MetaTable = load_meta_csv(cfg.meta_csv)
        self.indices = self.meta.indices_for_split(cfg.split)
        if int(cfg.max_rows) > 0:
            self.indices = self.indices[: int(cfg.max_rows)]

        # Store paths; open memmaps lazily per-worker.
        self._oph_tokens_dir = Path(cfg.oph_tokens_dir)
        self._kpgt_dir = Path(cfg.kpgt_dir)
        self._vae_latents_dir = Path(cfg.vae_latents_dir)

        self._oph: OpenPhenomTokens | None = None
        self._kpgt_table: np.ndarray | None = None
        self._kpgt_has: np.ndarray | None = None
        self._vae: VAELatentsCache | None = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _lazy_init(self) -> None:
        if self._oph is None:
            self._oph = OpenPhenomTokens(self._oph_tokens_dir)
        if self._kpgt_table is None or self._kpgt_has is None:
            table, has, _, _ = load_kpgt_cache(self._kpgt_dir)
            self._kpgt_table = table
            self._kpgt_has = has
        if self._vae is None:
            self._vae = VAELatentsCache(self._vae_latents_dir)

    def __getitem__(self, i: int) -> dict[str, Any]:
        self._lazy_init()
        assert self._oph is not None
        assert self._vae is not None
        assert self._kpgt_table is not None
        assert self._kpgt_has is not None

        idx = int(self.indices[int(i)])

        z_mean = np.asarray(self._vae.mean[idx]).copy()  # (C,H,W) fp16
        z_std = np.asarray(self._vae.std[idx]).copy()    # (C,H,W) fp16

        oph_row = int(self.meta.oph_row[idx])
        if oph_row < 0:
            raise RuntimeError(f"Missing oph_row for meta index {idx} (sample_id={int(self.meta.sample_id[idx])})")
        oph_tokens = np.asarray(self._oph.get_by_row(oph_row)).copy()  # (T,D) fp16

        treatment_id = int(self.meta.treatment_id[idx])
        kpgt_fp = np.asarray(self._kpgt_table[treatment_id]).copy()  # (D,) fp16
        has_kpgt = int(self._kpgt_has[treatment_id]) if self._kpgt_has is not None else 0

        return {
            "index": idx,
            "sample_id": int(self.meta.sample_id[idx]),
            "treatment_id": treatment_id,
            "dose_id": int(self.meta.dose_id[idx]),
            "dose_cont": float(self.meta.dose_cont[idx]),
            "empty_id": int(self.meta.empty_id[idx]),
            "kpgt_fp": torch.from_numpy(kpgt_fp),          # (2304,)
            "has_kpgt": has_kpgt,
            "z_mean": torch.from_numpy(z_mean),            # (C,H,W)
            "z_std": torch.from_numpy(z_std),              # (C,H,W)
            "oph_tokens": torch.from_numpy(oph_tokens),    # (T,D)
        }


@dataclass(frozen=True)
class RawImagesDatasetConfig:
    meta_csv: str
    split: str
    rxrx3_root: str
    oph_tokens_dir: str
    kpgt_dir: str

    # preprocessing (match adapter_vae)
    fixed_stats_path: str
    max_rows: int = 0
    percentile_bounds: tuple[float, float] = (0.5, 99.5)
    percentile_per_channel: bool = True
    use_channel_standardize: bool = False


class RawImagesDataset(Dataset):
    """
    Dataset for REPA-E (end-to-end): loads raw 6ch images + teacher tokens + condition ids.
    """

    def __init__(self, cfg: RawImagesDatasetConfig):
        self.cfg = cfg
        self.meta: MetaTable = load_meta_csv(cfg.meta_csv)
        self.indices = self.meta.indices_for_split(cfg.split)
        if int(cfg.max_rows) > 0:
            self.indices = self.indices[: int(cfg.max_rows)]

        self._rxrx3_root = Path(cfg.rxrx3_root)
        self._oph_tokens_dir = Path(cfg.oph_tokens_dir)
        self._kpgt_dir = Path(cfg.kpgt_dir)

        self._oph: OpenPhenomTokens | None = None
        self._kpgt_table: np.ndarray | None = None
        self._kpgt_has: np.ndarray | None = None
        self._preproc = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _lazy_init(self) -> None:
        if self._oph is None:
            self._oph = OpenPhenomTokens(self._oph_tokens_dir)
        if self._kpgt_table is None or self._kpgt_has is None:
            table, has, _, _ = load_kpgt_cache(self._kpgt_dir)
            self._kpgt_table = table
            self._kpgt_has = has
        if self._preproc is None:
            from adapter_vae.data.preprocessing import CellPaintingPreprocessor

            self._preproc = CellPaintingPreprocessor(
                percentile_bounds=tuple(self.cfg.percentile_bounds),
                use_channel_standardize=bool(self.cfg.use_channel_standardize),
                percentile_per_channel=bool(self.cfg.percentile_per_channel),
                fixed_stats_path=str(self.cfg.fixed_stats_path),
            )

    def __getitem__(self, i: int) -> dict[str, Any]:
        self._lazy_init()
        assert self._oph is not None
        assert self._kpgt_table is not None
        assert self._kpgt_has is not None
        assert self._preproc is not None

        idx = int(self.indices[int(i)])
        rel = str(self.meta.npy_path[idx])
        path = self._rxrx3_root / rel
        arr = np.load(path)
        x = torch.from_numpy(arr).float()
        x = self._preproc(x)  # [-1,1]

        oph_row = int(self.meta.oph_row[idx])
        if oph_row < 0:
            raise RuntimeError(f"Missing oph_row for meta index {idx} (sample_id={int(self.meta.sample_id[idx])})")
        oph_tokens = np.asarray(self._oph.get_by_row(oph_row)).copy()

        treatment_id = int(self.meta.treatment_id[idx])
        kpgt_fp = np.asarray(self._kpgt_table[treatment_id]).copy()
        has_kpgt = int(self._kpgt_has[treatment_id])

        return {
            "index": idx,
            "sample_id": int(self.meta.sample_id[idx]),
            "treatment_id": treatment_id,
            "dose_id": int(self.meta.dose_id[idx]),
            "dose_cont": float(self.meta.dose_cont[idx]),
            "empty_id": int(self.meta.empty_id[idx]),
            "kpgt_fp": torch.from_numpy(kpgt_fp),
            "has_kpgt": has_kpgt,
            "image": x,
            "oph_tokens": torch.from_numpy(oph_tokens),
        }
