from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def _parse_int(x: Any, column: str) -> int:
    if x is None:
        raise ValueError(f"Missing integer value for column `{column}`")
    s = str(x).strip()
    if s == "":
        raise ValueError(f"Empty integer value for column `{column}`")
    try:
        return int(float(s))
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for column `{column}`: {x!r}") from exc


def _parse_float(x: Any, column: str) -> float:
    if x is None:
        raise ValueError(f"Missing float value for column `{column}`")
    s = str(x).strip()
    if s == "":
        raise ValueError(f"Empty float value for column `{column}`")
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for column `{column}`: {x!r}") from exc


def _parse_str(x: Any, column: str) -> str:
    if x is None:
        raise ValueError(f"Missing string value for column `{column}`")
    return str(x)


@dataclass(frozen=True)
class MetaExt:
    sample_id: np.ndarray
    split: np.ndarray
    npy_path: np.ndarray

    experiment_name: np.ndarray
    plate: np.ndarray
    well: np.ndarray
    site: np.ndarray
    base_key: np.ndarray
    well_id: np.ndarray

    treatment: np.ndarray
    treatment_id: np.ndarray
    concentration: np.ndarray
    dose_id: np.ndarray
    dose_value: np.ndarray
    empty_id: np.ndarray

    dose_log10: np.ndarray
    dose_cont: np.ndarray
    perturbation_type: np.ndarray
    cell_type: np.ndarray
    oph_row: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.sample_id.shape[0])

    def indices_for_split(self, split: str) -> np.ndarray:
        split = str(split)
        return np.nonzero(self.split == split)[0].astype(np.int64)

    def iter_rows(self, indices: Iterable[int]) -> Iterable[dict[str, Any]]:
        for i in indices:
            yield self.row_dict(int(i))

    def row_dict(self, i: int) -> dict[str, Any]:
        ii = int(i)
        return {
            "meta_index": ii,
            "sample_id": int(self.sample_id[ii]),
            "split": str(self.split[ii]),
            "npy_path": str(self.npy_path[ii]),
            "experiment_name": str(self.experiment_name[ii]),
            "plate": int(self.plate[ii]),
            "well": str(self.well[ii]),
            "site": int(self.site[ii]),
            "base_key": str(self.base_key[ii]),
            "well_id": str(self.well_id[ii]),
            "treatment": str(self.treatment[ii]),
            "treatment_id": int(self.treatment_id[ii]),
            "concentration": float(self.concentration[ii]),
            "dose_id": int(self.dose_id[ii]),
            "dose_value": float(self.dose_value[ii]),
            "empty_id": int(self.empty_id[ii]),
            "dose_log10": float(self.dose_log10[ii]),
            "dose_cont": float(self.dose_cont[ii]),
            "perturbation_type": str(self.perturbation_type[ii]),
            "cell_type": str(self.cell_type[ii]),
            "oph_row": int(self.oph_row[ii]),
        }


def load_meta_ext(meta_csv: str | Path) -> MetaExt:
    meta_csv = Path(meta_csv)
    rows: list[dict[str, Any]] = []
    with open(meta_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        have = set(r.fieldnames or [])

        required = [
            "sample_id",
            "split",
            "npy_path",
            "experiment_name",
            "plate",
            "well",
            "site",
            "base_key",
            "well_id",
            "treatment",
            "treatment_id",
            "concentration",
            "dose_id",
            "dose_value",
            "empty_id",
            "dose_log10",
            "dose_cont",
            "cell_type",
            "perturbation_type",
            "oph_row",
        ]
        missing = [c for c in required if c not in have]
        if missing:
            raise ValueError(f"Missing required columns in {meta_csv}: {missing}")

        for row in r:
            rows.append(row)

    def _int(col: str) -> np.ndarray:
        return np.asarray([_parse_int(x.get(col), col) for x in rows], dtype=np.int64)

    def _float(col: str) -> np.ndarray:
        return np.asarray([_parse_float(x.get(col), col) for x in rows], dtype=np.float32)

    def _str(col: str) -> np.ndarray:
        return np.asarray([_parse_str(x.get(col), col) for x in rows], dtype=object)

    return MetaExt(
        sample_id=_int("sample_id"),
        split=_str("split"),
        npy_path=_str("npy_path"),
        experiment_name=_str("experiment_name"),
        plate=_int("plate"),
        well=_str("well"),
        site=_int("site"),
        base_key=_str("base_key"),
        well_id=_str("well_id"),
        treatment=_str("treatment"),
        treatment_id=_int("treatment_id"),
        concentration=_float("concentration"),
        dose_id=_int("dose_id"),
        dose_value=_float("dose_value"),
        empty_id=_int("empty_id"),
        dose_log10=_float("dose_log10"),
        dose_cont=_float("dose_cont"),
        perturbation_type=_str("perturbation_type"),
        cell_type=_str("cell_type"),
        oph_row=_int("oph_row"),
    )


def plate_key(meta: MetaExt, idx: int, *, include_cell_type: bool = True) -> tuple[str, int, Optional[str]]:
    i = int(idx)
    exp = str(meta.experiment_name[i])
    plate = int(meta.plate[i])
    if include_cell_type:
        return exp, plate, str(meta.cell_type[i])
    return exp, plate, None
