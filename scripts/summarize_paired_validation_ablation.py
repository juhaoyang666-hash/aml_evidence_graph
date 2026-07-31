#!/usr/bin/env python3
"""Summarize a pre-registered paired validation-only feature ablation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pr-auc-delta", type=float, default=0.0)
    parser.add_argument("--maximum-recall-drop", type=float, default=0.01)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_scope") != "train_validation_only":
        raise ValueError(f"Metrics are not validation-only: {path}")
    if payload.get("test_split_read") is not False:
        raise ValueError(f"Metrics do not prove a frozen test split: {path}")
    return payload


def _recall_at_point_one_percent(metrics: dict[str, object]) -> float:
    validation = metrics["validation_metrics"]
    return float(validation["alert_budgets"]["0.1000%"]["recall_at_k"])


def main() -> None:
    args = parse_args()
    if args.maximum_recall_drop < 0:
        raise ValueError("--maximum-recall-drop must be non-negative.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite paired summary: {args.output}")
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    if baseline["feature_root"] != candidate["feature_root"]:
        raise ValueError("Paired models must use the same feature root.")
    controlled_keys = (
        "architecture",
        "hidden_dim",
        "num_layers",
        "num_neighbors",
        "batch_size",
        "epochs",
        "learning_rate",
        "dropout",
        "patience",
        "history_window_days",
        "random_seed",
        "num_relations",
    )
    mismatched = [
        key
        for key in controlled_keys
        if baseline["configuration"].get(key) != candidate["configuration"].get(key)
    ]
    if mismatched:
        raise ValueError("Paired configurations differ: " + ", ".join(mismatched))

    baseline_pr_auc = float(baseline["validation_metrics"]["pr_auc"])
    candidate_pr_auc = float(candidate["validation_metrics"]["pr_auc"])
    baseline_recall = _recall_at_point_one_percent(baseline)
    candidate_recall = _recall_at_point_one_percent(candidate)
    pr_auc_delta = candidate_pr_auc - baseline_pr_auc
    recall_delta = candidate_recall - baseline_recall
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_scope": "train_validation_only",
        "test_split_read": False,
        "baseline_metrics": str(args.baseline),
        "candidate_metrics": str(args.candidate),
        "configuration": {
            key: baseline["configuration"].get(key) for key in controlled_keys
        },
        "excluded_feature_families": candidate.get("excluded_feature_families", []),
        "baseline": {
            "pr_auc": baseline_pr_auc,
            "recall_at_0_1_percent": baseline_recall,
        },
        "candidate": {
            "pr_auc": candidate_pr_auc,
            "recall_at_0_1_percent": candidate_recall,
        },
        "delta": {
            "pr_auc": pr_auc_delta,
            "recall_at_0_1_percent": recall_delta,
        },
        "gate": {
            "minimum_pr_auc_delta": args.minimum_pr_auc_delta,
            "maximum_recall_drop": args.maximum_recall_drop,
            "pr_auc_pass": pr_auc_delta > args.minimum_pr_auc_delta,
            "recall_pass": recall_delta >= -args.maximum_recall_drop,
            "passed": (
                pr_auc_delta > args.minimum_pr_auc_delta
                and recall_delta >= -args.maximum_recall_drop
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
