from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latent_flow2.utils.io import read_json


@dataclass(frozen=True)
class OpenPhenomMeta:
    dtype: str
    join_key: str
    num_rows: int
    token_dim: int
    tokens_h: int
    tokens_w: int
    keys_file: str
    tokens_file: str
    version: int

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "OpenPhenomMeta":
        return OpenPhenomMeta(
            dtype=str(obj["dtype"]),
            join_key=str(obj.get("join_key", "sample_id")),
            num_rows=int(obj["num_rows"]),
            token_dim=int(obj["token_dim"]),
            tokens_h=int(obj["tokens_h"]),
            tokens_w=int(obj["tokens_w"]),
            keys_file=str(obj.get("keys_file", "keys.npy")),
            tokens_file=str(obj.get("tokens_file", "tokens_fp16.npy")),
            version=int(obj.get("version", 1)),
        )


class OpenPhenomTokens:
    """
    Thin loader for the OpenPhenom patch-token cache:

      cache_dir/
        meta.json
        keys.npy          (int sample_id)
        tokens_fp16.npy   (float16) shape [N, 1024, 384]
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        self.meta = OpenPhenomMeta.from_json(read_json(meta_path))

        keys_path = self.cache_dir / self.meta.keys_file
        tokens_path = self.cache_dir / self.meta.tokens_file
        if not keys_path.exists():
            raise FileNotFoundError(f"keys file not found: {keys_path}")
        if not tokens_path.exists():
            raise FileNotFoundError(f"tokens file not found: {tokens_path}")

        self.keys = np.load(keys_path)
        self.tokens = np.load(tokens_path, mmap_mode="r")

        if self.keys.ndim != 1:
            raise ValueError(f"Expected keys to be 1D, got {self.keys.shape=}")
        if self.tokens.ndim != 3:
            raise ValueError(f"Expected tokens to be [N,T,D], got {self.tokens.shape=}")
        if int(self.tokens.shape[0]) != int(self.meta.num_rows):
            raise ValueError(f"tokens first dim mismatch: {self.tokens.shape[0]=} vs {self.meta.num_rows=}")
        if int(self.tokens.shape[1]) != int(self.meta.tokens_h * self.meta.tokens_w):
            raise ValueError(
                f"tokens T mismatch: {self.tokens.shape[1]=} vs {self.meta.tokens_h=}*{self.meta.tokens_w=}"
            )
        if int(self.tokens.shape[2]) != int(self.meta.token_dim):
            raise ValueError(f"tokens dim mismatch: {self.tokens.shape[2]=} vs {self.meta.token_dim=}")

    @property
    def token_count(self) -> int:
        return int(self.meta.tokens_h * self.meta.tokens_w)

    def get_by_row(self, row: int) -> np.ndarray:
        if row < 0:
            raise IndexError(f"Invalid row index: {row}")
        return np.asarray(self.tokens[int(row)])

    def get_batch_by_row(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        if rows.ndim != 1:
            raise ValueError(f"rows must be 1D, got {rows.shape=}")
        if (rows < 0).any():
            raise IndexError("rows contains negative indices")
        return np.asarray(self.tokens[rows])

