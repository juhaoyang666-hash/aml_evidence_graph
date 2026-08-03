from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.training.table_baseline import (
    feature_split_schema,
    load_feature_split,
)


def _write_partition(root: Path, event_date: str, values: list[int]) -> None:
    target = root / f"event_date={event_date}" / "split=train"
    target.mkdir(parents=True)
    pl.DataFrame(
        {
            "transaction_id": [f"tx-{value}" for value in values],
            "amount": values,
            "unused_feature": [value * 10 for value in values],
        }
    ).write_parquet(target / "part.parquet")


def test_load_feature_split_projects_physical_and_partition_columns(tmp_path: Path) -> None:
    _write_partition(tmp_path, "2023-01-01", [1, 2])
    _write_partition(tmp_path, "2023-01-02", [3])

    schema = feature_split_schema(tmp_path, TimeSplit.TRAIN)
    projected = load_feature_split(
        tmp_path,
        TimeSplit.TRAIN,
        columns=("transaction_id", "amount", "event_date", "split"),
    )

    assert "unused_feature" in schema
    assert schema["event_date"] == pl.String
    assert projected.columns == ["transaction_id", "amount", "event_date", "split"]
    assert projected["amount"].to_list() == [1, 2, 3]
    assert projected["event_date"].to_list() == ["2023-01-01", "2023-01-01", "2023-01-02"]
    assert projected["split"].to_list() == ["train", "train", "train"]


def test_load_feature_split_rejects_missing_projected_column(tmp_path: Path) -> None:
    _write_partition(tmp_path, "2023-01-01", [1])

    with pytest.raises(ValueError, match="missing_column"):
        load_feature_split(
            tmp_path,
            TimeSplit.TRAIN,
            columns=("transaction_id", "missing_column"),
        )
