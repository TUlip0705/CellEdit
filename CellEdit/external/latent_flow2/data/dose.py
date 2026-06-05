from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DoseBins:
    """
    Fixed 9-bin dose scheme:
      - 8 canonical non-empty doses (IDs 0..7)
      - 1 EMPTY bin (ID 8) for concentration <= 0 or treatment == EMPTY_control
    """

    nonempty_doses: tuple[float, ...] = (0.0025, 0.01, 0.025, 0.1, 0.25, 1.0, 2.5, 10.0)
    empty_id: int = 8

    def all_bins(self) -> tuple[float, ...]:
        return (*self.nonempty_doses, 0.0)


def nearest_dose_id(conc: float, bins: DoseBins) -> int:
    """
    Map a raw concentration to the nearest canonical non-empty dose id (0..7).
    Uses log10-distance for scale invariance.

    Caller must handle empty separately.
    """
    if not math.isfinite(conc) or conc <= 0:
        raise ValueError(f"nearest_dose_id expects conc>0 finite, got {conc}")

    logc = math.log10(conc)
    best_i = 0
    best_d = float("inf")
    for i, d in enumerate(bins.nonempty_doses):
        dd = abs(logc - math.log10(float(d)))
        if dd < best_d:
            best_d = dd
            best_i = i
    return int(best_i)


def dose_id_and_value(conc: float, treatment: str, bins: DoseBins) -> tuple[int, float]:
    """
    Returns:
      (dose_id, dose_value_mapped)
    """
    if str(treatment) == "EMPTY_control" or (isinstance(conc, (int, float)) and float(conc) <= 0.0):
        return int(bins.empty_id), 0.0
    did = nearest_dose_id(float(conc), bins)
    return did, float(bins.nonempty_doses[did])


def format_dose_ladder(doses: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(x):g}" for x in doses) + "]"

