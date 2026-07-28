"""Small Spark PIT replay for representative features; no Cartesian joins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

WINDOW_SECONDS = {"1d": 86_400, "7d": 604_800, "30d": 2_592_000}


def build_representative_pit_features(transactions: Any) -> Any:
    """Build five strictly historical window features on canonical transactions.

    The ordered epoch-second window ends at ``-1``. Consequently, future and
    same-timestamp transactions cannot contribute to the current row. This
    implementation uses window aggregation and contains no cross/cartesian join.
    """
    try:
        from pyspark.sql import Window
        from pyspark.sql import functions as functions
    except ImportError as error:
        raise RuntimeError("Install the 'spark' optional dependency group.") from error

    required = {
        "transaction_id",
        "event_ts",
        "sender_account_id",
        "receiver_account_id",
        "amount",
    }
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise ValueError(f"Spark input is missing canonical columns: {missing}")

    frame = transactions.withColumn("_event_epoch", functions.col("event_ts").cast("long"))
    for suffix, seconds in WINDOW_SECONDS.items():
        sender_window = (
            Window.partitionBy("sender_account_id")
            .orderBy(functions.col("_event_epoch"))
            .rangeBetween(-seconds, -1)
        )
        frame = frame.withColumn(
            f"sender_outgoing_count_{suffix}", functions.count("transaction_id").over(sender_window)
        )
    sender_30d = (
        Window.partitionBy("sender_account_id")
        .orderBy(functions.col("_event_epoch"))
        .rangeBetween(-WINDOW_SECONDS["30d"], -1)
    )
    receiver_7d = (
        Window.partitionBy("receiver_account_id")
        .orderBy(functions.col("_event_epoch"))
        .rangeBetween(-WINDOW_SECONDS["7d"], -1)
    )
    return (
        frame.withColumn(
            "sender_outgoing_amount_sum_30d",
            functions.coalesce(functions.sum("amount").over(sender_30d), functions.lit(0.0)),
        )
        .withColumn(
            "receiver_incoming_count_7d",
            functions.count("transaction_id").over(receiver_7d),
        )
        .drop("_event_epoch")
    )


def replay_parquet(input_path: Path, output_path: Path, *, master: str) -> None:
    """Run a local or cluster Spark replay over partitioned Parquet."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise RuntimeError("Install the 'spark' optional dependency group.") from error
    spark = (
        SparkSession.builder.appName("aml-pit-feature-replay")
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    try:
        frame = spark.read.parquet(str(input_path))
        build_representative_pit_features(frame).write.mode("overwrite").parquet(
            str(output_path)
        )
    finally:
        spark.stop()
