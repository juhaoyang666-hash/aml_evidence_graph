"""Pre-registered chronological data splits for AML experiments."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import polars as pl

from aml_evidence_graph.compat import to_polars_series


class TimeSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


SPLIT_BOUNDS: dict[TimeSplit, tuple[date, date]] = {
    TimeSplit.TRAIN: (date(2022, 10, 7), date(2023, 4, 30)),
    TimeSplit.VALIDATION: (date(2023, 5, 1), date(2023, 6, 30)),
    TimeSplit.TEST: (date(2023, 7, 1), date(2023, 8, 23)),
}


def _as_utc_datetime(event_ts: pl.Series) -> pl.Series:
    """Cast event timestamps to UTC datetime, coercing invalid values to null."""
    timestamps = event_ts
    if timestamps.dtype in (pl.Utf8, pl.String):
        return timestamps.str.to_datetime(time_zone="UTC", strict=False)
    if isinstance(timestamps.dtype, pl.Datetime):
        if timestamps.dtype.time_zone is None:
            return timestamps.dt.replace_time_zone("UTC")
        if timestamps.dtype.time_zone != "UTC":
            return timestamps.dt.convert_time_zone("UTC")
        return timestamps
    return timestamps.cast(pl.Datetime(time_zone="UTC"), strict=False)


def assign_time_split(event_ts: pl.Series | object) -> pl.Series:
    """Assign every timestamp to the one pre-registered time split.

    Timestamps outside the protocol window are rejected instead of silently
    falling into a training or test set.
    """
    timestamps = _as_utc_datetime(to_polars_series(event_ts, name="event_ts"))
    if timestamps.null_count() > 0:
        raise ValueError("event_ts contains invalid timestamps.")

    event_dates = timestamps.dt.date()
    expression = pl.lit(None, dtype=pl.Utf8)
    for split, (start, end) in SPLIT_BOUNDS.items():
        expression = (
            pl.when((pl.col("event_date") >= start) & (pl.col("event_date") <= end))
            .then(pl.lit(split.value))
            .otherwise(expression)
        )
    result = (
        pl.DataFrame({"event_date": event_dates})
        .select(expression.alias("split"))
        .get_column("split")
    )
    if result.null_count() > 0:
        raise ValueError("event_ts contains values outside the approved split windows.")
    return result


def split_bounds_as_iso() -> dict[str, dict[str, str]]:
    """Expose split bounds for manifests and model metadata."""
    return {
        split.value: {"start": start.isoformat(), "end": end.isoformat()}
        for split, (start, end) in SPLIT_BOUNDS.items()
    }
