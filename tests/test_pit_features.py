import pandas as pd

from aml_evidence_graph.features.pit import PITFeatureBuilder


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "event_ts": pd.to_datetime(
                [
                    "2022-10-07T10:00:00Z",
                    "2022-10-07T10:00:00Z",
                    "2022-10-07T10:01:00Z",
                ],
                utc=True,
            ),
            "sender_account_id": ["a", "a", "a"],
            "receiver_account_id": ["b", "c", "d"],
            "amount": [100.0, 200.0, 300.0],
            "payment_currency": ["USD", "USD", "USD"],
            "sender_location": ["US", "US", "US"],
            "receiver_location": ["US", "CA", "US"],
            "source_row_number": [1, 2, 3],
        }
    )


def test_same_timestamp_transactions_do_not_leak_into_each_other() -> None:
    result = PITFeatureBuilder().transform_partition(transactions())

    first = result.loc[result["transaction_id"] == "t1"].iloc[0]
    second = result.loc[result["transaction_id"] == "t2"].iloc[0]
    third = result.loc[result["transaction_id"] == "t3"].iloc[0]

    assert first["sender_outgoing_count_1h"] == 0
    assert second["sender_outgoing_count_1h"] == 0
    assert third["sender_outgoing_count_1h"] == 2
    assert third["sender_outgoing_same_currency_amount_sum_1h"] == 300
    assert third["sender_outgoing_unique_counterparties_1h"] == 2
    assert third["amount_log1p"] > 0
    assert third["is_currency_conversion"] == 0
