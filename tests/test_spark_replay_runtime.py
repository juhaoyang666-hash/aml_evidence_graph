from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime, timedelta

import pytest

if os.environ.get("AML_RUN_SPARK_TESTS") != "1":
    pytest.skip(
        "Set AML_RUN_SPARK_TESTS=1 for the explicit JVM-backed Spark test.",
        allow_module_level=True,
    )

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from aml_evidence_graph.features.spark_replay import (
    REPRESENTATIVE_FEATURES,
    _configure_bundled_java,
    build_representative_pit_features,
)


def test_spark_windows_exclude_same_timestamp_and_future_rows() -> None:
    _configure_bundled_java()
    if not os.environ.get("JAVA_HOME") and shutil.which("java") is None:
        pytest.skip("Java runtime is unavailable.")
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("spark-pit-runtime-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    started = datetime(2023, 7, 1, tzinfo=UTC)
    try:
        frame = spark.createDataFrame(
            [
                ("txn-1", started, "A", "B"),
                ("txn-2", started, "A", "C"),
                ("txn-3", started + timedelta(seconds=1), "A", "D"),
                ("txn-4", started + timedelta(seconds=2), "A", "B"),
            ],
            [
                "transaction_id",
                "event_ts",
                "sender_account_id",
                "receiver_account_id",
            ],
        )
        rows = {
            row["transaction_id"]: row.asDict()
            for row in build_representative_pit_features(frame)
            .select("transaction_id", *REPRESENTATIVE_FEATURES)
            .collect()
        }
    finally:
        spark.stop()

    assert rows["txn-1"]["sender_outgoing_count_1d"] == 0
    assert rows["txn-2"]["sender_outgoing_count_1d"] == 0
    assert rows["txn-3"]["sender_outgoing_count_1d"] == 2
    assert rows["txn-4"]["sender_outgoing_count_1d"] == 3
    assert rows["txn-4"]["receiver_incoming_count_7d"] == 1
    assert rows["txn-4"]["relationship_count_7d"] == 1
