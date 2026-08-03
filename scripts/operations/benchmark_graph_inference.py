"""Benchmark trusted graph-checkpoint loading and one frozen date partition."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import torch

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.graph.snapshots import DailyGraphSnapshotBuilder
from aml_evidence_graph.models.graph_loading import load_graphsage_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--event-date", type=date.fromisoformat, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive.")
    load_started = time.perf_counter()
    loaded = load_graphsage_artifact(args.checkpoint, device=args.device)
    _synchronize(loaded.device)
    load_ms = (time.perf_counter() - load_started) * 1_000

    dataset = ds.dataset(args.features, format="parquet", partitioning="hive")
    event_date = args.event_date.isoformat()
    history_start = (
        args.event_date - timedelta(days=loaded.config.history_window_days)
    ).isoformat()
    columns = sorted(
        {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.source_row_number,
            *loaded.edge_feature_columns,
        }
    )
    snapshot_started = time.perf_counter()
    history = pl.from_arrow(
        dataset.to_table(
            filter=(ds.field("event_date") >= history_start)
            & (ds.field("event_date") < event_date),
            columns=columns,
        )
    )
    current = pl.from_arrow(
        dataset.to_table(filter=ds.field("event_date") == event_date, columns=columns)
    )
    if current.is_empty():
        raise ValueError(f"No rows found for event date {event_date}.")
    builder = DailyGraphSnapshotBuilder(
        loaded.node_indexer,
        edge_feature_columns=loaded.edge_feature_columns,
        history_window=timedelta(days=loaded.config.history_window_days),
        store_relation_types=loaded.config.architecture == "rgcn",
    )
    if not history.is_empty():
        builder.build(history, include_labels=False)
    snapshots = builder.build(current, include_labels=False)
    snapshot_ms = (time.perf_counter() - snapshot_started) * 1_000

    if loaded.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(loaded.device)
    latencies_ms: list[float] = []
    score_count = 0
    for _ in range(args.repeats):
        _synchronize(loaded.device)
        started = time.perf_counter()
        scores = loaded.predict(snapshots)
        _synchronize(loaded.device)
        latencies_ms.append((time.perf_counter() - started) * 1_000)
        score_count = len(scores)
        if score_count != current.height or not np.isfinite(scores).all():
            raise AssertionError("Graph inference returned invalid or incomplete scores.")

    steady_state = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "local frozen partition benchmark; not a production SLA",
        "checkpoint": str(args.checkpoint),
        "architecture": loaded.config.architecture,
        "device": str(loaded.device),
        "event_date": event_date,
        "history_window_days": loaded.config.history_window_days,
        "history_rows": history.height,
        "scored_rows": score_count,
        "snapshot_count": len(snapshots),
        "repeats": args.repeats,
        "checkpoint_load_ms": load_ms,
        "snapshot_read_and_build_ms": snapshot_ms,
        "inference_latency_ms": {
            "values": latencies_ms,
            "cold_first": latencies_ms[0],
            "mean": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "steady_state_mean": float(np.mean(steady_state)),
            "steady_state_p95": float(np.percentile(steady_state, 95)),
        },
        "steady_state_rows_per_second": score_count
        / max(float(np.mean(steady_state)) / 1_000, 1e-9),
        "gpu_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(loaded.device) / 1024 / 1024
            if loaded.device.type == "cuda"
            else 0.0
        ),
        "gpu_peak_reserved_mb": (
            torch.cuda.max_memory_reserved(loaded.device) / 1024 / 1024
            if loaded.device.type == "cuda"
            else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
