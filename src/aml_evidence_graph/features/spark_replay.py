"""Small Spark PIT replay for representative features; no Cartesian joins."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

WINDOW_SECONDS = {"1d": 86_400, "7d": 604_800, "30d": 2_592_000}
REPRESENTATIVE_FEATURES = (
    "sender_outgoing_count_1d",
    "sender_outgoing_count_7d",
    "sender_outgoing_count_30d",
    "receiver_incoming_count_7d",
    "relationship_count_7d",
)


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
    receiver_7d = (
        Window.partitionBy("receiver_account_id")
        .orderBy(functions.col("_event_epoch"))
        .rangeBetween(-WINDOW_SECONDS["7d"], -1)
    )
    relationship_7d = (
        Window.partitionBy("sender_account_id", "receiver_account_id")
        .orderBy(functions.col("_event_epoch"))
        .rangeBetween(-WINDOW_SECONDS["7d"], -1)
    )
    return (
        frame.withColumn(
            "receiver_incoming_count_7d",
            functions.count("transaction_id").over(receiver_7d),
        )
        .withColumn(
            "relationship_count_7d",
            functions.count("transaction_id").over(relationship_7d),
        )
        .drop("_event_epoch")
    )


def _configure_bundled_java() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    candidate = Path(sys.prefix) / "Library"
    if (candidate / "bin" / "java.exe").is_file():
        os.environ["JAVA_HOME"] = str(candidate)
        os.environ["PATH"] = f"{candidate / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"


def _start_resource_monitor() -> tuple[threading.Event, list[float], threading.Thread]:
    stop = threading.Event()
    peak_rss_mb = [0.0]
    process = psutil.Process()

    def monitor() -> None:
        while not stop.wait(0.05):
            processes = [process, *process.children(recursive=True)]
            total = 0
            for item in processes:
                try:
                    total += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            peak_rss_mb[0] = max(peak_rss_mb[0], total / 1024 / 1024)

    thread = threading.Thread(target=monitor, name="spark-resource-monitor", daemon=True)
    thread.start()
    return stop, peak_rss_mb, thread


def replay_parquet(
    input_path: Path,
    output_path: Path,
    *,
    master: str,
    target_event_date: str | None = None,
    shuffle_partitions: int = 8,
) -> dict[str, object]:
    """Run a local or cluster Spark replay over partitioned Parquet."""
    _configure_bundled_java()
    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise RuntimeError("Install the 'spark' optional dependency group.") from error
    spark = (
        SparkSession.builder.appName("aml-pit-feature-replay")
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    started = time.perf_counter()
    stop_monitor, peak_rss_mb, monitor_thread = _start_resource_monitor()
    try:
        input_files = sorted(input_path.rglob("*.parquet"))
        if not input_files:
            raise FileNotFoundError(f"No Parquet files found below {input_path}")
        frame = (
            spark.read.option("basePath", str(input_path.resolve()))
            .parquet(*[str(path.resolve()) for path in input_files])
        )
        if target_event_date is not None:
            from datetime import date, timedelta

            from pyspark.sql import functions as functions

            target = date.fromisoformat(target_event_date)
            history_start = (target - timedelta(days=30)).isoformat()
            next_day = (target + timedelta(days=1)).isoformat()
            frame = frame.where(
                (functions.col("event_ts") >= functions.lit(history_start).cast("timestamp"))
                & (functions.col("event_ts") < functions.lit(next_day).cast("timestamp"))
            )
        frame = frame.persist()
        input_rows_scanned = frame.count()
        replayed = build_representative_pit_features(frame)
        if target_event_date is not None:
            replayed = replayed.where(replayed.event_date == target_event_date)
        replayed = replayed.persist()
        output_rows = replayed.count()
        physical_plan = replayed._jdf.queryExecution().executedPlan().toString()
        write_mode = "spark_parquet"
        if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
            import pyarrow as pa
            import pyarrow.parquet as pq

            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / "spark-replay.parquet"
            selected_columns = [
                "transaction_id",
                "event_ts",
                "event_date",
                "split",
                *REPRESENTATIVE_FEATURES,
            ]
            collected = replayed.select(*selected_columns).toPandas()
            pq.write_table(pa.Table.from_pandas(collected), output_file)
            write_mode = "driver_arrow_windows_fallback"
        else:
            replayed.write.mode("overwrite").parquet(str(output_path))
        replayed.unpersist()
        frame.unpersist()
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "target_event_date": target_event_date,
            "input_file_count": len(input_files),
            "input_rows_scanned": input_rows_scanned,
            "output_rows": output_rows,
            "duration_seconds": time.perf_counter() - started,
            "master": master,
            "shuffle_partitions": shuffle_partitions,
            "exchange_count": physical_plan.count("Exchange"),
            "write_mode": write_mode,
            "peak_process_tree_rss_mb": peak_rss_mb[0],
        }
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        spark.stop()
