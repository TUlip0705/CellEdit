from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latent_flow2.utils.io import read_json, write_json


@dataclass(frozen=True)
class KPGTCacheMeta:
    version: int
    dtype: str
    kpgt_dim: int
    num_treatments: int
    table_file: str
    has_file: str
    treatment_vocab_file: str
    sources: dict[str, Any]


def load_treatment_vocab(path: str | Path) -> dict[str, int]:
    vocab = read_json(path)
    if not isinstance(vocab, dict):
        raise ValueError(f"treatment_vocab must be a dict, got {type(vocab)}")
    out: dict[str, int] = {}
    for k, v in vocab.items():
        out[str(k)] = int(v)
    return out


def load_kpgt_npz(npz_path: str | Path) -> np.ndarray:
    arr = np.load(npz_path, allow_pickle=False)
    if "fps" not in arr:
        raise ValueError(f"`fps` not found in {npz_path}; keys={list(arr.keys())}")
    fps = np.asarray(arr["fps"])
    if fps.ndim != 2:
        raise ValueError(f"Expected fps to be 2D, got {fps.shape=}")
    return fps


def load_kpgt_map_csv(map_csv: str | Path) -> dict[str, int]:
    """
    Returns mapping: treatment -> row_id (index into fps array)
    """
    mp: dict[str, int] = {}
    with open(map_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = ["row_id", "treatment"]
        missing = [c for c in required if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in {map_csv}: {missing}")
        for row in r:
            t = str(row["treatment"])
            i = int(float(row["row_id"]))
            if t in mp and mp[t] != i:
                raise ValueError(f"Duplicate treatment with different row_id: {t} ({mp[t]} vs {i})")
            mp[t] = i
    return mp


def write_kpgt_cache(
    *,
    out_dir: str | Path,
    treatment_vocab: dict[str, int],
    kpgt_fps: np.ndarray,
    kpgt_map: dict[str, int],
    dtype: str = "float16",
    allow_missing: bool = False,
    empty_treatment: str = "EMPTY_control",
    table_name: str = "kpgt_table_fp16.npy",
    has_name: str = "has_kpgt.npy",
    meta_name: str = "kpgt_meta.json",
    treatment_vocab_name: str = "treatment_vocab.json",
    sources: dict[str, Any] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_treatments = int(len(treatment_vocab))
    kpgt_dim = int(kpgt_fps.shape[1])

    if dtype not in ("float16", "float32"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    np_dtype = np.float16 if dtype == "float16" else np.float32

    table = np.zeros((num_treatments, kpgt_dim), dtype=np_dtype)
    has = np.zeros((num_treatments,), dtype=np.uint8)

    missing: list[str] = []
    for t, tid in treatment_vocab.items():
        if t == empty_treatment:
            continue
        rid = kpgt_map.get(t, None)
        if rid is None:
            missing.append(t)
            continue
        if rid < 0 or rid >= kpgt_fps.shape[0]:
            raise ValueError(f"row_id out of range for {t}: {rid}")
        table[int(tid)] = np.asarray(kpgt_fps[int(rid)], dtype=np_dtype)
        has[int(tid)] = 1

    if missing and not allow_missing:
        raise ValueError(
            f"Missing KPGT features for {len(missing)} treatments (first 10): {missing[:10]}. "
            f"Use --allow_missing to fill zeros."
        )

    np.save(out_dir / table_name, table)
    np.save(out_dir / has_name, has)
    write_json(out_dir / treatment_vocab_name, treatment_vocab)

    meta = KPGTCacheMeta(
        version=1,
        dtype=dtype,
        kpgt_dim=kpgt_dim,
        num_treatments=num_treatments,
        table_file=table_name,
        has_file=has_name,
        treatment_vocab_file=treatment_vocab_name,
        sources=sources or {},
    )
    write_json(out_dir / meta_name, meta.__dict__)


def load_kpgt_cache(kpgt_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, int], KPGTCacheMeta]:
    kpgt_dir = Path(kpgt_dir)
    meta_path = kpgt_dir / "kpgt_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"kpgt_meta.json not found: {meta_path}")
    meta_obj = read_json(meta_path)
    meta = KPGTCacheMeta(
        version=int(meta_obj["version"]),
        dtype=str(meta_obj["dtype"]),
        kpgt_dim=int(meta_obj["kpgt_dim"]),
        num_treatments=int(meta_obj["num_treatments"]),
        table_file=str(meta_obj["table_file"]),
        has_file=str(meta_obj["has_file"]),
        treatment_vocab_file=str(meta_obj["treatment_vocab_file"]),
        sources=dict(meta_obj.get("sources", {})),
    )
    table = np.load(kpgt_dir / meta.table_file, mmap_mode="r")
    has = np.load(kpgt_dir / meta.has_file, mmap_mode="r")
    vocab = load_treatment_vocab(kpgt_dir / meta.treatment_vocab_file)
    return table, has, vocab, meta

