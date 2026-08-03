"""Benchmark frozen CatBoost + graph + fusion partition scoring in-process."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from aml_evidence_graph.api.private_scoring import PrivateFeaturePartitionScoringService
from aml_evidence_graph.api.services import InMemoryEvidenceStore
from aml_evidence_graph.training.graphsage import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table-model-dir", type=Path, required=True)
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--event-date", required=True)
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
    process = psutil.Process()
    device = resolve_device(args.device)
    rss_before_mb = process.memory_info().rss / 1024 / 1024
    initialization_started = time.perf_counter()
    store = InMemoryEvidenceStore()
    scorer = PrivateFeaturePartitionScoringService(
        feature_root=args.features,
        table_model_dir=args.table_model_dir,
        evidence_store=store,
        alert_threshold=0.5,
        source_version="frozen-local-benchmark",
        graphsage_artifact_path=args.graph_checkpoint,
        fusion_dir=args.fusion_dir,
        graphsage_device=args.device,
    )
    _synchronize(device)
    initialization_ms = (time.perf_counter() - initialization_started) * 1_000
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    alert_counts: list[int] = []
    for _ in range(args.repeats):
        _synchronize(device)
        started = time.perf_counter()
        result = scorer.score_partition(args.event_date)
        _synchronize(device)
        latencies_ms.append((time.perf_counter() - started) * 1_000)
        alert_counts.append(len(result.alert_ids))
    if len(set(alert_counts)) != 1:
        raise AssertionError(
            f"Repeated frozen scoring produced inconsistent alert counts: {alert_counts}."
        )

    rss_after_mb = process.memory_info().rss / 1024 / 1024
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "local in-process frozen partition benchmark; not HTTP and not production SLA",
        "event_date": args.event_date,
        "components": ["catboost", "graphsage_column_holding_gat", "oof_fusion"],
        "device": str(device),
        "repeats": args.repeats,
        "alert_count": alert_counts[0],
        "initialization_ms": initialization_ms,
        "partition_latency_ms": {
            "values": latencies_ms,
            "mean": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
        },
        "partitions_per_second": 1_000 / max(float(np.mean(latencies_ms)), 1e-9),
        "process_rss_before_mb": rss_before_mb,
        "process_rss_after_mb": rss_after_mb,
        "gpu_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if device.type == "cuda"
            else 0.0
        ),
        "gpu_peak_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / 1024 / 1024
            if device.type == "cuda"
            else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
