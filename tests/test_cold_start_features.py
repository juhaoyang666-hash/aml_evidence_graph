from pathlib import Path

import polars as pl
import pytest

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.cold_start import (
    CANDIDATE_FEATURE_NAMES,
    COLD_START_FEATURE_FAMILIES,
    ColdStartGraphFeatureBuilder,
    build_cold_start_feature_sidecar,
)


def test_cold_start_feature_families_are_disjoint_and_complete() -> None:
    novelty = set(COLD_START_FEATURE_FAMILIES["event_time_novelty"])
    degree = set(COLD_START_FEATURE_FAMILIES["degree_interactions"])

    assert len(novelty) == 6
    assert len(degree) == 7
    assert novelty.isdisjoint(degree)
    assert novelty | degree == set(CANDIDATE_FEATURE_NAMES)


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col(CANONICAL.event_ts).str.to_datetime(time_zone="UTC")
    )


def _base_row(
    transaction_id: str,
    event_ts: str,
    sender: str,
    receiver: str,
    row_number: int,
    degrees: tuple[float, float, float, float],
) -> dict[str, object]:
    return {
        CANONICAL.transaction_id: transaction_id,
        CANONICAL.event_ts: event_ts,
        CANONICAL.sender_account_id: sender,
        CANONICAL.receiver_account_id: receiver,
        CANONICAL.source_row_number: row_number,
        CANONICAL.is_laundering: 0,
        "graph_sender_historical_out_degree": degrees[0],
        "graph_sender_historical_in_degree": degrees[1],
        "graph_receiver_historical_out_degree": degrees[2],
        "graph_receiver_historical_in_degree": degrees[3],
    }


def test_cold_start_features_exclude_same_timestamp_accounts() -> None:
    first_partition = _frame(
        [
            _base_row("one", "2023-01-01T10:00:00Z", "a", "b", 1, (0, 0, 0, 0)),
            _base_row("two", "2023-01-01T10:00:00Z", "c", "a", 2, (0, 0, 0, 0)),
        ]
    )
    second_partition = _frame(
        [_base_row("three", "2023-01-02T10:00:00Z", "b", "a", 3, (0, 1, 1, 1))]
    )
    builder = ColdStartGraphFeatureBuilder()

    simultaneous = builder.transform_partition(first_partition)
    later = builder.transform_partition(second_partition).row(0, named=True)

    assert simultaneous["graph_sender_seen_before_any_role"].to_list() == [0.0, 0.0]
    assert simultaneous["graph_receiver_seen_before_any_role"].to_list() == [0.0, 0.0]
    assert simultaneous["graph_both_endpoints_unseen_before"].to_list() == [1.0, 1.0]
    assert later["graph_sender_seen_before_any_role"] == 1.0
    assert later["graph_receiver_seen_before_any_role"] == 1.0
    assert later["graph_sender_observed_age_days"] == 1.0
    assert later["graph_receiver_observed_age_days"] == 1.0
    assert later["graph_sender_total_historical_degree"] == 1.0
    assert later["graph_receiver_total_historical_degree"] == 2.0
    assert later["graph_endpoint_min_historical_degree"] == 1.0
    assert later["graph_endpoint_max_historical_degree"] == 2.0
    assert later["graph_endpoint_degree_imbalance_ratio"] == 0.25
    assert later["graph_sender_receiver_only_transition"] == 1.0
    assert later["graph_receiver_sender_only_transition"] == 0.0


def _write_partition(
    root: Path,
    event_date: str,
    split: str,
    rows: list[dict[str, object]],
) -> None:
    target = root / f"event_date={event_date}" / f"split={split}"
    target.mkdir(parents=True)
    _frame(rows).write_parquet(target / "part.parquet")


def test_build_cold_start_sidecar_preserves_base_rows_and_writes_contract(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "pit_features"
    _write_partition(
        input_root,
        "2023-01-01",
        "train",
        [_base_row("one", "2023-01-01T10:00:00Z", "a", "b", 1, (0, 0, 0, 0))],
    )
    _write_partition(
        input_root,
        "2023-01-02",
        "train",
        [_base_row("two", "2023-01-02T10:00:00Z", "b", "a", 2, (0, 1, 1, 0))],
    )
    output_root = tmp_path / "cold_start_sidecar"

    summary = build_cold_start_feature_sidecar(input_root, output_root)

    assert summary.partition_count == 2
    assert summary.row_count == 2
    assert summary.feature_count == len(CANDIDATE_FEATURE_NAMES)
    output = pl.read_parquet(
        output_root / "event_date=2023-01-02" / "split=train" / "part-00000.parquet"
    )
    assert output[CANONICAL.transaction_id].to_list() == ["two"]
    assert output[CANONICAL.is_laundering].to_list() == [0]
    assert set(CANDIDATE_FEATURE_NAMES).issubset(output.columns)
    assert (output_root / "_cold_start_feature_contract.json").is_file()
    assert (output_root / "_cold_start_feature_summary.json").is_file()
    assert (output_root / "_run_manifest.json").is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_cold_start_feature_sidecar(input_root, output_root)
