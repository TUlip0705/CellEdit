from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class MetaTable:
    sample_id: np.ndarray         # (N,) int64
    split: np.ndarray             # (N,) str
    npy_path: np.ndarray          # (N,) str (relative to rxrx3_root)
    treatment: np.ndarray         # (N,) str
    treatment_id: np.ndarray      # (N,) int64
    concentration: np.ndarray     # (N,) float32
    dose_log10: np.ndarray        # (N,) float32 (log10(concentration); 0 for empty/invalid)
    dose_cont: np.ndarray         # (N,) float32 (scaled dose, typically in [-1,1]; 0 for empty/invalid)
    dose_id: np.ndarray           # (N,) int64
    empty_id: np.ndarray          # (N,) int64
    oph_row: np.ndarray           # (N,) int64 (row index into oph tokens cache; -1 if missing)

    @property
    def num_rows(self) -> int:
        return int(self.sample_id.shape[0])

    def indices_for_split(self, split: str) -> np.ndarray:
        split = str(split)
        return np.nonzero(self.split == split)[0].astype(np.int64)


def load_meta_csv(meta_csv: str | Path) -> MetaTable:
    meta_csv = Path(meta_csv)
    rows: list[dict[str, Any]] = []
    with open(meta_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        have = set(r.fieldnames or [])
        required = [
            "sample_id",
            "split",
            "npy_path",
            "treatment",
            "treatment_id",
            "concentration",
            "dose_id",
            "empty_id",
            "oph_row",
        ]
        missing = [c for c in required if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in {meta_csv}: {missing}")
        for row in r:
            rows.append(row)

    def _int(col: str) -> np.ndarray:
        return np.asarray([int(float(x[col])) for x in rows], dtype=np.int64)

    def _float(col: str) -> np.ndarray:
        return np.asarray([float(x[col]) for x in rows], dtype=np.float32)

    def _str(col: str) -> np.ndarray:
        return np.asarray([str(x[col]) for x in rows], dtype=object)

    def _float_optional(col: str, default: float = 0.0) -> np.ndarray:
        if col not in have:
            return np.full((len(rows),), float(default), dtype=np.float32)
        out = []
        for x in rows:
            v = x.get(col, default)
            if v in ("", None):
                out.append(float(default))
            else:
                out.append(float(v))
        return np.asarray(out, dtype=np.float32)

    return MetaTable(
        sample_id=_int("sample_id"),
        split=_str("split"),
        npy_path=_str("npy_path"),
        treatment=_str("treatment"),
        treatment_id=_int("treatment_id"),
        concentration=_float("concentration"),
        dose_log10=_float_optional("dose_log10", default=0.0),
        dose_cont=_float_optional("dose_cont", default=0.0),
        dose_id=_int("dose_id"),
        empty_id=_int("empty_id"),
        oph_row=_int("oph_row"),
    )


def iter_image_paths(rxrx3_root: str | Path, meta: MetaTable) -> Iterable[Path]:
    rxrx3_root = Path(rxrx3_root)
    for rel in meta.npy_path.tolist():
        yield rxrx3_root / str(rel)
