#!/usr/bin/env python3
"""Evaluate one validation-selected graph checkpoint on the chronological test once."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import torch

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.evaluation.monitoring import measure_runtime
from aml_evidence_graph.graph.snapshots import (
    DailyGraphSnapshotBuilder,
    TemporalGraphSnapshot,
)
from aml_evidence_graph.models.graph_loading import load_graphsage_artifact
from aml_evidence_graph.training.run_graphsage import _write_graph_scores
from aml_evidence_graph.training.table_baseline import load_feature_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-label",
        default="validation-only graph candidate",
        help="Human-readable frozen selection protocol.",
    )
    parser.add_argument(
        "--selection-evidence",
        type=Path,
        help="Optional JSON gate that must have passed without reading test.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen test output: {args.output}")
    args.output.mkdir(parents=True)
    started_at = time.perf_counter()
    selection_evidence: dict[str, object] | None = None
    if args.selection_evidence is not None:
        selection_evidence = json.loads(args.selection_evidence.read_text(encoding="utf-8"))
        if selection_evidence.get("passed") is not True:
            raise ValueError("Selection evidence did not pass its frozen validation gate.")
        if selection_evidence.get("test_split_read") is not False:
            raise ValueError("Selection evidence does not prove that test remained frozen.")
    loaded = load_graphsage_artifact(args.checkpoint, device=args.device)

    # Read test only in this final selected-candidate command. Only the frozen
    # history window before its first day is needed to seed causal graph state.
    test, load_resources = measure_runtime(
        lambda: load_feature_split(args.features, TimeSplit.TEST)
    )
    first_test_ts = test[CANONICAL.event_ts].min()
    if first_test_ts is None:
        raise ValueError("The chronological test split is empty.")
    first_test_date = first_test_ts.date()
    history_start = first_test_date - timedelta(days=loaded.config.history_window_days)
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
    dataset = ds.dataset(args.features, format="parquet", partitioning="hive")
    history = pl.from_arrow(
        dataset.to_table(
            filter=(ds.field("event_date") >= history_start.isoformat())
            & (ds.field("event_date") < first_test_date.isoformat()),
            columns=columns,
        )
    )
    builder = DailyGraphSnapshotBuilder(
        loaded.node_indexer,
        edge_feature_columns=loaded.edge_feature_columns,
        history_window=timedelta(days=loaded.config.history_window_days),
        store_relation_types=loaded.config.architecture == "rgcn",
    )

    def build_test_snapshots() -> list[TemporalGraphSnapshot]:
        builder.build(history, include_labels=False)
        return builder.build(test, include_labels=False)

    test_snapshots, snapshot_resources = measure_runtime(build_test_snapshots)
    test_scores, inference_resources = measure_runtime(
        lambda: loaded.predict(test_snapshots)
    )
    ordered_test = test.sort(
        [CANONICAL.event_ts, CANONICAL.source_row_number], maintain_order=True
    )
    labels = ordered_test[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    if len(test_scores) != len(labels) or not np.isfinite(test_scores).all():
        raise AssertionError("Frozen checkpoint returned invalid or incomplete test scores.")
    metrics = evaluate_binary_risk_scores(labels, test_scores)
    _write_graph_scores(
        args.output,
        split_name="test",
        frame=test,
        scores=test_scores,
    )
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "selection": args.selection_label,
            "selection_evidence": (
                str(args.selection_evidence) if args.selection_evidence is not None else None
            ),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "test_evaluations": 1,
            "configuration_frozen": True,
            "history_start": history_start.isoformat(),
            "first_test_date": first_test_date.isoformat(),
            "history_rows": history.height,
        },
        "configuration": asdict(loaded.config),
        "edge_feature_columns": list(loaded.edge_feature_columns),
        "test_metrics": metrics,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started_at,
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(loaded.device) / 1024 / 1024
                if loaded.device.type == "cuda"
                else 0.0
            ),
            **{f"load_{key}": value for key, value in load_resources.items()},
            **{f"snapshot_{key}": value for key, value in snapshot_resources.items()},
            **{f"test_inference_{key}": value for key, value in inference_resources.items()},
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
