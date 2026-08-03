from pathlib import Path

import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.build import build_pit_feature_dataset
from aml_evidence_graph.ingestion.smoke_subset import prepare_smoke_subset


def test_prepare_smoke_subset_copies_requested_dates(tmp_path: Path) -> None:
    source = tmp_path / "prepared"
    for event_date in ("2023-04-30", "2023-05-01", "2023-07-01"):
        target = source / f"event_date={event_date}" / "split=train"
        target.mkdir(parents=True)
        pl.DataFrame({"transaction_id": [event_date]}).write_parquet(target / "part.parquet")

    summary = prepare_smoke_subset(
        source,
        tmp_path / "smoke",
        event_dates=("2023-04-30", "2023-05-01", "2023-07-02"),
    )

    assert summary.copied_dates == ["2023-04-30", "2023-05-01"]
    assert summary.missing_dates == ["2023-07-02"]
    assert (tmp_path / "smoke" / "event_date=2023-04-30").is_dir()


def _write_partition(root: Path, event_date: str, rows: list[dict[str, object]]) -> None:
    target = root / f"event_date={event_date}" / "split=train"
    target.mkdir(parents=True)
    pl.DataFrame(rows).with_columns(
        pl.col(CANONICAL.event_ts).str.to_datetime(time_zone="UTC")
    ).write_parquet(target / "part.parquet")


def test_build_feature_dataset_preserves_history_and_adds_approved_rule(tmp_path: Path) -> None:
    input_root = tmp_path / "prepared"
    _write_partition(
        input_root,
        "2023-01-01",
        [
            {
                CANONICAL.transaction_id: "one",
                CANONICAL.event_ts: "2023-01-01T12:00:00Z",
                CANONICAL.sender_account_id: "account_a",
                CANONICAL.receiver_account_id: "account_b",
                CANONICAL.amount: 10.0,
                CANONICAL.payment_currency: "USD",
                CANONICAL.sender_location: "US",
                CANONICAL.receiver_location: "CA",
                CANONICAL.source_row_number: 1,
                CANONICAL.is_laundering: 0,
                CANONICAL.laundering_type: "None",
            }
        ],
    )
    _write_partition(
        input_root,
        "2023-01-02",
        [
            {
                CANONICAL.transaction_id: "two",
                CANONICAL.event_ts: "2023-01-02T12:00:00Z",
                CANONICAL.sender_account_id: "account_a",
                CANONICAL.receiver_account_id: "account_c",
                CANONICAL.amount: 20.0,
                CANONICAL.payment_currency: "USD",
                CANONICAL.sender_location: "US",
                CANONICAL.receiver_location: "CA",
                CANONICAL.source_row_number: 2,
                CANONICAL.is_laundering: 1,
                CANONICAL.laundering_type: "Structuring",
            }
        ],
    )

    rule_file = tmp_path / "rules.yaml"
    rule_file.write_text(
        """
version: "test-v1"
rules:
  - rule_id: R-HISTORY
    status: approved
    owner: test-owner
    effective_from: "2022-01-01"
    approval_reference: "test-approval"
    backtest_summary: "Synthetic test rule."
    kind: numeric_threshold
    operator: gte
    feature: sender_outgoing_cross_border_count_14d
    required_features:
      - sender_outgoing_cross_border_count_14d
    parameters:
      threshold: 1
    explanation_template: "Prior cross-border count reached threshold."
""".strip(),
        encoding="utf-8",
    )
    summary = build_pit_feature_dataset(
        input_root,
        tmp_path / "features",
        rules_path=rule_file,
    )

    assert summary.row_count == 2
    assert summary.rule_hit_count == 1
    assert summary.feature_registry_version == "features-v2"
    output = pl.read_parquet(
        tmp_path / "features" / "event_date=2023-01-02" / "split=train" / "part-00000.parquet"
    )
    row = output.row(0, named=True)
    assert row["sender_outgoing_count_7d"] == 1.0
    assert row["is_cross_border_current_transaction"] == 1.0
    assert row["graph_sender_historical_out_degree"] == 1.0
    assert row["graph_directed_edge_prior_count"] == 0.0
    assert row["rule_R-HISTORY_hit"] == 1
    assert row["any_rule_hit"] == 1
    assert row["rule_hit_count"] == 1
    assert row["hour_of_day"] == 12
    assert (tmp_path / "features" / "_rule_evidence" / "event_date=2023-01-02.json").is_file()
    assert summary.run_id
    assert (tmp_path / "features" / "_run_manifest.json").is_file()
    registry = (tmp_path / "features" / "_feature_registry.json").read_text(encoding="utf-8")
    assert "sender_outgoing_count_7d" in registry
    assert "rule_R-HISTORY_hit" in registry
    assert "is_high_risk_corridor" in registry
    assert "any_rule_hit" in registry


def test_build_feature_dataset_respects_max_dates(tmp_path: Path) -> None:
    input_root = tmp_path / "prepared"
    for event_date, transaction_id in (
        ("2023-01-01", "one"),
        ("2023-01-02", "two"),
        ("2023-01-03", "three"),
    ):
        _write_partition(
            input_root,
            event_date,
            [
                {
                    CANONICAL.transaction_id: transaction_id,
                    CANONICAL.event_ts: f"{event_date}T12:00:00Z",
                    CANONICAL.sender_account_id: "account_a",
                    CANONICAL.receiver_account_id: "account_b",
                    CANONICAL.amount: 10.0,
                    CANONICAL.payment_currency: "USD",
                    CANONICAL.sender_location: "US",
                    CANONICAL.receiver_location: "CA",
                    CANONICAL.source_row_number: 1,
                    CANONICAL.is_laundering: 0,
                    CANONICAL.laundering_type: "None",
                }
            ],
        )

    summary = build_pit_feature_dataset(
        input_root,
        tmp_path / "features",
        max_dates=2,
    )

    assert summary.partition_count == 2
    assert summary.event_date_min == "2023-01-01"
    assert summary.event_date_max == "2023-01-02"
    assert not (tmp_path / "features" / "event_date=2023-01-03").exists()
