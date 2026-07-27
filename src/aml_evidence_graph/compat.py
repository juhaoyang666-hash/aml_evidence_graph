"""Polars/pandas conversion helpers for gradual pipeline migration.

Polars is the primary in-memory tabular type. Pandas remains only at
sklearn/CatBoost boundaries that still require pandas DataFrames.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl


def to_polars(frame: pl.DataFrame | pd.DataFrame | Any) -> pl.DataFrame:
    """Convert a tabular object to a Polars DataFrame."""
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    if isinstance(frame, pl.Series):
        return frame.to_frame()
    if isinstance(frame, pd.Series):
        return pl.from_pandas(frame.to_frame())
    raise TypeError(f"Unsupported frame type for to_polars: {type(frame)!r}")


def to_pandas(frame: pl.DataFrame | pd.DataFrame | Any) -> pd.DataFrame:
    """Convert a tabular object to a pandas DataFrame (ML library boundary)."""
    if isinstance(frame, pd.DataFrame):
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    if isinstance(frame, pl.Series):
        return frame.to_pandas().to_frame()
    if isinstance(frame, pd.Series):
        return frame.to_frame()
    raise TypeError(f"Unsupported frame type for to_pandas: {type(frame)!r}")


def to_polars_series(values: pl.Series | pd.Series | Any, *, name: str | None = None) -> pl.Series:
    """Convert a 1-d object to a Polars Series."""
    if isinstance(values, pl.Series):
        return values if name is None else values.rename(name)
    if isinstance(values, pd.Series):
        series = pl.from_pandas(values)
        return series if name is None else series.rename(name)
    series = pl.Series(name or "values", values)
    return series


def is_numeric_dtype(dtype: pl.DataType | Any) -> bool:
    """Return True when a Polars (or pandas) dtype is numeric."""
    if isinstance(dtype, pl.DataType):
        return dtype.is_numeric()
    try:
        return bool(pd.api.types.is_numeric_dtype(dtype))
    except (TypeError, ValueError):
        return False


def stable_row_hash(values: pl.Series, *, seed: int = 0) -> pl.Series:
    """Deterministic uint64 hashes for reproducible negative downsampling."""
    return values.hash(seed=seed)
