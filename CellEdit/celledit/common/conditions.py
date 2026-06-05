from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from latent_flow2.data.dose import DoseBins, dose_id_and_value
from latent_flow2.data.kpgt import load_kpgt_cache

from .meta_ext import MetaExt


def dose_id_from_concentration_bins(concentration: float, treatment: str) -> int:
    bins = DoseBins()
    did, _ = dose_id_and_value(float(concentration), str(treatment), bins)
    return int(did)


@dataclass(frozen=True)
class CondInputsBins:
    kpgt_fp: torch.Tensor
    dose_id: torch.Tensor
    empty_id: torch.Tensor


@dataclass(frozen=True)
class CondInputsContFilm:
    kpgt_fp: torch.Tensor
    dose_cont: torch.Tensor
    empty_id: torch.Tensor


def load_kpgt_table(kpgt_dir: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    table, has, vocab, _ = load_kpgt_cache(kpgt_dir)
    return table, has, vocab


def cond_inputs_from_meta_index_bins(
    meta: MetaExt,
    *,
    kpgt_table: np.ndarray,
    index: int,
    device: torch.device,
) -> CondInputsBins:
    i = int(index)
    tid = int(meta.treatment_id[i])
    kpgt = np.asarray(kpgt_table[tid]).copy()
    kpgt_fp = torch.from_numpy(kpgt).to(device=device, dtype=torch.float32).view(1, -1)

    dose_id = torch.tensor([int(meta.dose_id[i])], device=device, dtype=torch.long)
    empty_id = torch.tensor([int(meta.empty_id[i])], device=device, dtype=torch.long)
    return CondInputsBins(kpgt_fp=kpgt_fp, dose_id=dose_id, empty_id=empty_id)


def cond_inputs_from_treatment_and_dose_bins(
    *,
    treatment: str,
    treatment_id: int,
    dose_id: int,
    empty_id: int,
    kpgt_table: np.ndarray,
    device: torch.device,
) -> CondInputsBins:
    tid = int(treatment_id)
    kpgt = np.asarray(kpgt_table[tid]).copy()
    kpgt_fp = torch.from_numpy(kpgt).to(device=device, dtype=torch.float32).view(1, -1)
    return CondInputsBins(
        kpgt_fp=kpgt_fp,
        dose_id=torch.tensor([int(dose_id)], device=device, dtype=torch.long),
        empty_id=torch.tensor([int(empty_id)], device=device, dtype=torch.long),
    )


def cond_inputs_from_meta_index_cont_film(
    meta: MetaExt,
    *,
    kpgt_table: np.ndarray,
    index: int,
    device: torch.device,
) -> CondInputsContFilm:
    i = int(index)
    tid = int(meta.treatment_id[i])
    kpgt = np.asarray(kpgt_table[tid]).copy()
    kpgt_fp = torch.from_numpy(kpgt).to(device=device, dtype=torch.float32).view(1, -1)

    dose_cont = torch.tensor([float(meta.dose_cont[i])], device=device, dtype=torch.float32)
    empty_id = torch.tensor([int(meta.empty_id[i])], device=device, dtype=torch.long)
    return CondInputsContFilm(kpgt_fp=kpgt_fp, dose_cont=dose_cont, empty_id=empty_id)


def ensure_empty_control_ids(meta: MetaExt, idx: int) -> None:
    i = int(idx)
    if int(meta.empty_id[i]) != 1:
        raise ValueError(f"Row is not empty_id=1: idx={i} treatment={meta.treatment[i]} empty_id={meta.empty_id[i]}")
    if str(meta.treatment[i]) != "EMPTY_control":
        raise ValueError(f"Expected treatment=EMPTY_control for empty rows: idx={i} treatment={meta.treatment[i]}")
