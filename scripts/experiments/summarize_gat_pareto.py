#!/usr/bin/env python3
"""Summarize validation-only GAT engineering candidates without reading test scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/gat_pareto"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/gat/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gat_pareto_summary.json"))
    parser.add_argument("--pr-auc-tolerance", type=float, default=0.002)
    return parser.parse_args()


def _candidate_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["configuration"]
    runtime = payload["runtime"]
    return {
        "name": name,
        "batch_size": int(config["batch_size"]),
        "num_neighbors": list(config["num_neighbors"]),
        "history_window_days": int(config["history_window_days"]),
        "validation_pr_auc": float(payload["validation_metrics"]["pr_auc"]),
        "training_wall_time_ms": float(runtime["training_wall_time_ms"]),
        "validation_inference_wall_time_ms": float(
            runtime["validation_inference_wall_time_ms"]
        ),
        "gpu_peak_memory_mb": float(runtime["gpu_peak_memory_mb"]),
        "selection_scope": payload["selection_scope"],
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    objectives = (
        (left["validation_pr_auc"], right["validation_pr_auc"], "max"),
        (left["training_wall_time_ms"], right["training_wall_time_ms"], "min"),
        (
            left["validation_inference_wall_time_ms"],
            right["validation_inference_wall_time_ms"],
            "min",
        ),
        (left["gpu_peak_memory_mb"], right["gpu_peak_memory_mb"], "min"),
    )
    no_worse = all(a >= b if direction == "max" else a <= b for a, b, direction in objectives)
    strictly_better = any(
        a > b if direction == "max" else a < b for a, b, direction in objectives
    )
    return no_worse and strictly_better


def main() -> None:
    args = parse_args()
    if args.pr_auc_tolerance < 0:
        raise ValueError("pr-auc-tolerance must be non-negative.")
    paths = sorted(args.root.glob("*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No completed candidate metrics under {args.root}")
    rows = [
        _candidate_row(path.parent.name, json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]

    # Existing baseline is admitted using validation and resource fields only. Its
    # test metrics are intentionally neither copied nor consulted by selection.
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_config = baseline.get("configuration") or {
        "batch_size": 2048,
        "num_neighbors": [15, 10],
        "history_window_days": 30,
    }
    baseline_runtime = baseline["runtime"]
    rows.append(
        {
            "name": "baseline",
            "batch_size": int(baseline_config.get("batch_size", 2048)),
            "num_neighbors": list(baseline_config.get("num_neighbors", [15, 10])),
            "history_window_days": int(baseline_config.get("history_window_days", 30)),
            "validation_pr_auc": float(baseline["validation_metrics"]["pr_auc"]),
            "training_wall_time_ms": float(baseline_runtime["training_wall_time_ms"]),
            "validation_inference_wall_time_ms": float(
                baseline_runtime["validation_inference_wall_time_ms"]
            ),
            "gpu_peak_memory_mb": float(baseline_runtime["gpu_peak_memory_mb"]),
            "selection_scope": "validation_fields_only_from_frozen_baseline",
        }
    )
    rows.sort(key=lambda row: row["name"])
    pareto = [
        row["name"]
        for row in rows
        if not any(_dominates(other, row) for other in rows if other is not row)
    ]
    best_pr_auc = max(row["validation_pr_auc"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["validation_pr_auc"] >= best_pr_auc - args.pr_auc_tolerance
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["validation_inference_wall_time_ms"],
            row["gpu_peak_memory_mb"],
            row["training_wall_time_ms"],
            row["name"],
        ),
    )
    payload = {
        "schema_version": "1.0",
        "protocol": {
            "selection_split": "validation_only",
            "test_metrics_consulted": False,
            "pr_auc_tolerance": args.pr_auc_tolerance,
            "selection_rule": (
                "within tolerance of best validation PR-AUC; then minimum validation "
                "inference time, GPU peak memory, training time, and lexical name"
            ),
        },
        "rows": rows,
        "pareto_front": pareto,
        "selected_candidate": selected["name"],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
