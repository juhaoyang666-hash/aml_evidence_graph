import pandas as pd
import pytest

from aml_evidence_graph.data.contract import DataContractError, normalize_transaction_chunk


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": ["10:00:00", "10:00:01"],
            "Date": ["2022-10-07", "2022-10-07"],
            "Sender_account": ["sender-a", "sender-b"],
            "Receiver_account": ["receiver-a", "receiver-b"],
            "Amount": [100.0, 200.0],
            "Payment_currency": ["USD", "USD"],
            "Received_currency": ["USD", "USD"],
            "Sender_bank_location": ["US", "US"],
            "Receiver_bank_location": ["US", "US"],
            "Payment_type": ["Transfer", "Transfer"],
            "Is_laundering": [0, 1],
            "Laundering_type": ["Normal", "Fan_In"],
        }
    )


def test_normalize_transaction_chunk_creates_canonical_schema() -> None:
    normalized = normalize_transaction_chunk(raw_frame(), source_row_start=17)

    assert normalized["transaction_id"].tolist() == ["txn-row-000000000017", "txn-row-000000000018"]
    assert normalized["is_laundering"].tolist() == [0, 1]
    assert str(normalized["event_ts"].dt.tz) == "UTC"


def test_normalize_transaction_chunk_rejects_invalid_binary_label() -> None:
    raw = raw_frame()
    raw.loc[0, "Is_laundering"] = 2

    with pytest.raises(DataContractError, match="binary"):
        normalize_transaction_chunk(raw, source_row_start=1)


def test_normalize_transaction_chunk_rejects_missing_required_source_column() -> None:
    raw = raw_frame().drop(columns="Payment_type")

    with pytest.raises(DataContractError, match="missing required columns"):
        normalize_transaction_chunk(raw, source_row_start=1)
