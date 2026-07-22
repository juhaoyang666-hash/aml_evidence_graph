"""Pre-registered chronological data splits for AML experiments."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import pandas as pd


class TimeSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


SPLIT_BOUNDS: dict[TimeSplit, tuple[date, date]] = {
    TimeSplit.TRAIN: (date(2022, 10, 7), date(2023, 4, 30)),
    TimeSplit.VALIDATION: (date(2023, 5, 1), date(2023, 6, 30)),
    TimeSplit.TEST: (date(2023, 7, 1), date(2023, 8, 23)),
}


def assign_time_split(event_ts: pd.Series) -> pd.Series:
    """Assign every timestamp to the one pre-registered time split.

    Timestamps outside the protocol window are rejected instead of silently
    falling into a training or test set.
    """
    timestamps = pd.to_datetime(event_ts, errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("event_ts contains invalid timestamps.")

    result = pd.Series(pd.NA, index=event_ts.index, dtype="string")
    for split, (start, end) in SPLIT_BOUNDS.items():
        mask = (timestamps.dt.date >= start) & (timestamps.dt.date <= end)
        result.loc[mask] = split.value

    if result.isna().any():
        raise ValueError("event_ts contains values outside the approved split windows.")
    return result.astype("string")


def split_bounds_as_iso() -> dict[str, dict[str, str]]:
    """Expose split bounds for manifests and model metadata."""
    return {
        split.value: {"start": start.isoformat(), "end": end.isoformat()}
        for split, (start, end) in SPLIT_BOUNDS.items()
    }

