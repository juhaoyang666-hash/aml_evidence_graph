import math
from dataclasses import replace

import polars as pl

from aml_evidence_graph.features.engineering_config import FeatureEngineeringConfig
from aml_evidence_graph.features.pit import PITFeatureBuilder


def transactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "event_ts": pl.Series(
                [
                    "2022-10-07T10:00:00Z",
                    "2022-10-07T10:00:00Z",
                    "2022-10-07T10:01:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
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

    first = result.filter(pl.col("transaction_id") == "t1").row(0, named=True)
    second = result.filter(pl.col("transaction_id") == "t2").row(0, named=True)
    third = result.filter(pl.col("transaction_id") == "t3").row(0, named=True)

    assert first["sender_outgoing_count_1h"] == 0
    assert second["sender_outgoing_count_1h"] == 0
    assert third["sender_outgoing_count_1h"] == 2
    assert third["sender_outgoing_same_currency_amount_sum_1h"] == 300
    assert third["sender_outgoing_unique_counterparties_1h"] == 2
    assert third["amount_log1p"] > 0
    assert third["is_currency_conversion"] == 0
    assert third["hour_of_day"] == 10
    assert third["day_of_week"] == 4  # Friday
    assert third["is_weekend"] == 0
    assert third["sender_outgoing_count_1d_over_30d"] == 2.0 / 3.0
    assert first["seconds_since_last_outgoing"] == 2_592_000.0
    assert third["seconds_since_last_outgoing"] == 60.0


def test_high_risk_and_payment_type_flags_use_config() -> None:
    config = replace(
        FeatureEngineeringConfig.defaults(),
        high_risk_locations=frozenset({"Mexico"}),
        cash_like_payment_types=frozenset({"Cash Deposit"}),
        cross_border_payment_types=frozenset({"Cross-border"}),
        reporting_threshold=1000.0,
        just_below_reporting_threshold_ratio=0.9,
        small_amount_threshold=150.0,
    )
    frame = pl.DataFrame(
        {
            "transaction_id": ["a1", "a2", "a3"],
            "event_ts": pl.Series(
                [
                    "2022-10-07T09:00:00Z",
                    "2022-10-07T10:00:00Z",
                    "2022-10-07T11:00:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            "sender_account_id": ["s1", "s1", "s1"],
            "receiver_account_id": ["r1", "r2", "r3"],
            "amount": [100.0, 950.0, 1000.5],
            "payment_currency": ["USD", "USD", "USD"],
            "payment_type": ["Cash Deposit", "Cross-border", "Transfer"],
            "sender_location": ["Mexico", "US", "US"],
            "receiver_location": ["US", "Mexico", "US"],
            "source_row_number": [1, 2, 3],
        }
    )
    result = PITFeatureBuilder(config).transform_partition(frame)
    rows = {
        row["transaction_id"]: row
        for row in result.iter_rows(named=True)
    }

    assert rows["a1"]["is_high_risk_sender_location"] == 1
    assert rows["a1"]["is_high_risk_corridor"] == 1
    assert rows["a1"]["is_cash_like_payment"] == 1
    assert rows["a1"]["is_round_amount"] == 1
    assert rows["a2"]["is_high_risk_receiver_location"] == 1
    assert rows["a2"]["is_cross_border_payment_type"] == 1
    assert rows["a2"]["is_just_below_reporting_threshold"] == 1
    assert rows["a3"]["is_just_below_reporting_threshold"] == 0
    assert rows["a3"]["is_round_amount"] == 0
    # Prior small-amount outs from s1: a1 (100) to r1 only before a3.
    assert rows["a3"]["sender_small_amount_unique_receivers_7d"] == 1
    expected_ratio = 1000.5 / (1.0 + (100.0 + 950.0) / 2.0)
    assert math.isclose(
        rows["a3"]["amount_to_sender_outgoing_mean_ratio_30d"],
        expected_ratio,
        rel_tol=1e-9,
    )


def test_deposit_send_and_same_second_cash_in_do_not_leak() -> None:
    config = replace(
        FeatureEngineeringConfig.defaults(),
        cash_like_payment_types=frozenset({"Cash"}),
        cash_in_then_out_window_hours=2.0,
        missing_recency_seconds=9999.0,
    )
    frame = pl.DataFrame(
        {
            "transaction_id": ["cash_in", "same_second_out", "later_out"],
            "event_ts": pl.Series(
                [
                    "2022-10-07T12:00:00Z",
                    "2022-10-07T12:00:00Z",
                    "2022-10-07T12:30:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            "sender_account_id": ["payer", "hub", "hub"],
            "receiver_account_id": ["hub", "dest", "dest2"],
            "amount": [50.0, 40.0, 30.0],
            "payment_currency": ["USD", "USD", "USD"],
            "payment_type": ["Cash", "Transfer", "Transfer"],
            "sender_location": ["US", "US", "US"],
            "receiver_location": ["US", "US", "US"],
            "source_row_number": [1, 2, 3],
        }
    )
    result = PITFeatureBuilder(config).transform_partition(frame)
    rows = {
        row["transaction_id"]: row
        for row in result.iter_rows(named=True)
    }

    assert rows["same_second_out"]["cash_in_then_out_within_window"] == 0
    assert rows["same_second_out"]["seconds_since_last_incoming"] == 9999.0
    assert rows["later_out"]["cash_in_then_out_within_window"] == 1
    assert rows["later_out"]["seconds_since_last_incoming"] == 1800.0
    assert rows["later_out"]["seconds_since_last_outgoing"] == 1800.0


def test_receiver_small_amount_unique_senders_is_pit_safe() -> None:
    config = replace(
        FeatureEngineeringConfig.defaults(),
        small_amount_threshold=100.0,
    )
    frame = pl.DataFrame(
        {
            "transaction_id": ["in1", "in2", "probe"],
            "event_ts": pl.Series(
                [
                    "2022-10-07T08:00:00Z",
                    "2022-10-07T09:00:00Z",
                    "2022-10-07T10:00:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            "sender_account_id": ["s1", "s2", "s3"],
            "receiver_account_id": ["recv", "recv", "recv"],
            "amount": [50.0, 200.0, 10.0],
            "payment_currency": ["USD", "USD", "USD"],
            "sender_location": ["US", "US", "US"],
            "receiver_location": ["US", "US", "US"],
            "source_row_number": [1, 2, 3],
        }
    )
    result = PITFeatureBuilder(config).transform_partition(frame)
    probe = result.filter(pl.col("transaction_id") == "probe").row(0, named=True)
    assert probe["receiver_small_amount_unique_senders_7d"] == 1
