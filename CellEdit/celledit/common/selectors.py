from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .meta_ext import MetaExt, plate_key


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _sanitize_split(split: str) -> str:
    return str(split).strip()


def find_empty_candidates_for_plate(
    meta: MetaExt,
    *,
    experiment_name: str,
    plate: int,
    cell_type: Optional[str],
    prefer_split: Optional[str] = None,
) -> np.ndarray:
    exp = str(experiment_name)
    plate_i = int(plate)
    cell = str(cell_type) if cell_type is not None else None

    m = (meta.experiment_name == exp) & (meta.plate == plate_i) & (meta.empty_id.astype(np.int64) == 1)
    if cell is not None:
        m = m & (meta.cell_type == cell)
    idx = np.nonzero(m)[0].astype(np.int64)
    if idx.size == 0:
        return idx

    if prefer_split is None:
        return idx

    sp = _sanitize_split(prefer_split)
    sp_mask = meta.split[idx] == sp
    idx_pref = idx[sp_mask]
    idx_other = idx[~sp_mask]
    if idx_pref.size == 0:
        return idx
    return np.concatenate([idx_pref, idx_other], axis=0)


def pick_empty_for_target(
    meta: MetaExt,
    *,
    tgt_index: int,
    seed: int,
    prefer_same_site: bool = True,
    prefer_split: Optional[str] = None,
    include_cell_type_in_plate_key: bool = True,
) -> int:
    exp, plate, cell = plate_key(meta, int(tgt_index), include_cell_type=include_cell_type_in_plate_key)
    empties = find_empty_candidates_for_plate(meta, experiment_name=exp, plate=plate, cell_type=cell, prefer_split=prefer_split)
    if empties.size == 0:
        raise RuntimeError(f"No EMPTY_control found for target plate: exp={exp} plate={plate} cell_type={cell}")

    rng = _rng(seed)
    if prefer_same_site:
        tgt_site = int(meta.site[int(tgt_index)])
        same_site = empties[meta.site[empties] == tgt_site]
        if same_site.size > 0:
            return int(rng.choice(same_site))

    return int(rng.choice(empties))


def candidate_targets(
    meta: MetaExt,
    *,
    treatment: str,
    dose_id: int,
    split: str,
    include_cell_type_in_plate_key: bool = True,
) -> np.ndarray:
    treatment = str(treatment)
    split = _sanitize_split(split)
    did = int(dose_id)
    m = (
        (meta.treatment == treatment)
        & (meta.dose_id.astype(np.int64) == did)
        & (meta.empty_id.astype(np.int64) == 0)
        & (meta.split == split)
    )
    idx = np.nonzero(m)[0].astype(np.int64)
    if idx.size == 0:
        return idx

    keep = []
    for i in idx.tolist():
        exp, plate, cell = plate_key(meta, i, include_cell_type=include_cell_type_in_plate_key)
        empties = find_empty_candidates_for_plate(meta, experiment_name=exp, plate=plate, cell_type=cell)
        if empties.size > 0:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def pick_target(
    meta: MetaExt,
    *,
    treatment: str,
    dose_id: int,
    split: str,
    seed: int,
    include_cell_type_in_plate_key: bool = True,
) -> int:
    idx = candidate_targets(
        meta,
        treatment=str(treatment),
        dose_id=int(dose_id),
        split=str(split),
        include_cell_type_in_plate_key=include_cell_type_in_plate_key,
    )
    if idx.size == 0:
        raise RuntimeError(f"No valid target rows for treatment={treatment} dose_id={dose_id} split={split} (after plate-empty filtering)")
    rng = _rng(seed)
    return int(rng.choice(idx))


@dataclass(frozen=True)
class CounterfactualPair:
    src_index: int
    tgt_index: int


def pick_counterfactual_pair(
    meta: MetaExt,
    *,
    treatment: str,
    dose_id: int,
    split: str,
    seed: int,
    prefer_same_site: bool = True,
    prefer_empty_split: Optional[str] = None,
    include_cell_type_in_plate_key: bool = True,
) -> CounterfactualPair:
    tgt = pick_target(
        meta,
        treatment=str(treatment),
        dose_id=int(dose_id),
        split=str(split),
        seed=int(seed),
        include_cell_type_in_plate_key=include_cell_type_in_plate_key,
    )
    src = pick_empty_for_target(
        meta,
        tgt_index=int(tgt),
        seed=int(seed) + 999_983,
        prefer_same_site=bool(prefer_same_site),
        prefer_split=prefer_empty_split,
        include_cell_type_in_plate_key=include_cell_type_in_plate_key,
    )
    return CounterfactualPair(src_index=int(src), tgt_index=int(tgt))
