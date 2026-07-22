from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.build import build_pit_feature_dataset


def _write_partition(root: Path, event_date: str, rows: list[dict[str, object]]) -> None:
    target = root / f"event_date={event_date}" / "split=train"
    target.mkdir(parents=True)
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    pq.write_table(table, target / "part.parquet")


def test_build_feature_dataset_preserves_history_and_adds_approved_rule(tmp_path: Path) -> None:
    input_root = tmp_path / "tokenized"
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
    assert summary.feature_registry_version == "features-v1"
    output = pd.read_parquet(
        tmp_path / "features" / "event_date=2023-01-02" / "split=train" / "part-00000.parquet"
    )
    assert output.loc[0, "sender_outgoing_count_7d"] == 1.0
    assert output.loc[0, "is_cross_border_current_transaction"] == 1.0
    assert output.loc[0, "graph_sender_historical_out_degree"] == 1.0
    assert output.loc[0, "graph_directed_edge_prior_count"] == 0.0
    assert output.loc[0, "rule_R-HISTORY_hit"] == 1
    assert (tmp_path / "features" / "_rule_evidence" / "event_date=2023-01-02.json").is_file()
    assert summary.run_id
    assert (tmp_path / "features" / "_run_manifest.json").is_file()
    registry = (tmp_path / "features" / "_feature_registry.json").read_text(encoding="utf-8")
    assert "sender_outgoing_count_7d" in registry
    assert "rule_R-HISTORY_hit" in registry
