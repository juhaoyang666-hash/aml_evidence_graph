import json

import pandas as pd
import pytest

from aml_evidence_graph.ingestion.profile import build_manifest, write_manifest


def test_build_manifest_writes_only_aggregates(tmp_path) -> None:
    path = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "Time": ["10:00:00", "11:00:00"],
            "Date": ["2022-10-07", "2023-07-01"],
            "Sender_account": ["private-a", "private-b"],
            "Receiver_account": ["private-c", "private-d"],
            "Amount": [100.0, 200.0],
            "Payment_currency": ["USD", "USD"],
            "Received_currency": ["USD", "USD"],
            "Sender_bank_location": ["US", "US"],
            "Receiver_bank_location": ["US", "US"],
            "Payment_type": ["Transfer", "Transfer"],
            "Is_laundering": [0, 1],
            "Laundering_type": ["Normal", "Fan_In"],
        }
    ).to_csv(path, index=False)

    manifest = build_manifest(path, chunk_size=1)
    output = tmp_path / "manifest.json"
    write_manifest(manifest, output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    rendered = output.read_text(encoding="utf-8")
    assert loaded["aggregate_profile"]["row_count"] == 2
    assert loaded["aggregate_profile"]["positive_count"] == 1
    assert "private-a" not in rendered
    assert "private-d" not in rendered
    assert loaded["data_configuration"]["version"] == "data-v1"


def test_manifest_applies_required_column_null_rate_gate(tmp_path) -> None:
    path = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "Time": ["10:00:00"],
            "Date": ["2022-10-07"],
            "Sender_account": [None],
            "Receiver_account": ["private-c"],
            "Amount": [100.0],
            "Payment_currency": ["USD"],
            "Received_currency": ["USD"],
            "Sender_bank_location": ["US"],
            "Receiver_bank_location": ["US"],
            "Payment_type": ["Transfer"],
            "Is_laundering": [0],
            "Laundering_type": ["Normal"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="null rate"):
        build_manifest(path)
