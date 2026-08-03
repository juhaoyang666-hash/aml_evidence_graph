#!/usr/bin/env python3
"""Evaluate the pre-registered validation gate for event-time novelty features."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--baseline-slices", type=Path, required=True)
    parser.add_argument("--candidate-slices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_values(payload: dict[str, object]) -> tuple[float, float]:
    if payload.get("selection_scope") != "train_validation_only":
        raise ValueError("Candidate gate only accepts train/validation metrics.")
    if payload.get("test_split_read") is not False:
        raise ValueError("Candidate gate requires test_split_read=false.")
    validation = payload["validation_metrics"]
    return (
        float(validation["pr_auc"]),
        float(validation["alert_budgets"]["0.1000%"]["recall_at_k"]),
    )


def _slice_values(payload: dict[str, object]) -> dict[str, float]:
    if payload["protocol"].get("split") != "validation":
        raise ValueError("Candidate gate only accepts validation risk slices.")
    return {
        "event_time_either_endpoint_unseen": float(
            payload["event_time_novelty"]["either_endpoint_unseen_before"]["1.0"][
                "pr_auc"
            ]
        ),
        "training_either_endpoint_new": float(
            payload["new_account"]["either_endpoint_new"]["pr_auc"]
        ),
        "low_nonzero_degree": float(payload["degree_band"]["low_nonzero"]["pr_auc"]),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite novelty gate: {args.output}")
    baseline_metrics = _load(args.baseline_metrics)
    candidate_metrics = _load(args.candidate_metrics)
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
    if baseline_metrics["feature_root"] != candidate_metrics["feature_root"]:
        raise ValueError("Paired models must use the same feature root.")
    mismatched = [
        key
        for key in controlled_keys
        if baseline_metrics["configuration"].get(key)
        != candidate_metrics["configuration"].get(key)
    ]
    if mismatched:
        raise ValueError("Paired configurations differ: " + ", ".join(mismatched))

    baseline_pr_auc, baseline_recall = _metric_values(baseline_metrics)
    candidate_pr_auc, candidate_recall = _metric_values(candidate_metrics)
    baseline_slices = _slice_values(_load(args.baseline_slices))
    candidate_slices = _slice_values(_load(args.candidate_slices))
    metric_delta = {
        "pr_auc": candidate_pr_auc - baseline_pr_auc,
        "recall_at_0_1_percent": candidate_recall - baseline_recall,
    }
    slice_delta = {
        key: candidate_slices[key] - baseline_slices[key] for key in baseline_slices
    }
    checks = {
        "pr_auc_positive": metric_delta["pr_auc"] > 0.0,
        "recall_drop_within_0_01": metric_delta["recall_at_0_1_percent"] >= -0.01,
        "event_time_unseen_not_worse": (
            slice_delta["event_time_either_endpoint_unseen"] >= 0.0
        ),
        "training_new_not_worse": slice_delta["training_either_endpoint_new"] >= 0.0,
        "low_degree_drop_within_0_05": slice_delta["low_nonzero_degree"] >= -0.05,
    }
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_scope": "train_validation_only",
        "test_split_read": False,
        "baseline_metrics": str(args.baseline_metrics),
        "candidate_metrics": str(args.candidate_metrics),
        "baseline_slices": str(args.baseline_slices),
        "candidate_slices": str(args.candidate_slices),
        "configuration": {
            key: baseline_metrics["configuration"].get(key) for key in controlled_keys
        },
        "metrics": {
            "baseline": {
                "pr_auc": baseline_pr_auc,
                "recall_at_0_1_percent": baseline_recall,
            },
            "candidate": {
                "pr_auc": candidate_pr_auc,
                "recall_at_0_1_percent": candidate_recall,
            },
            "delta": metric_delta,
        },
        "slices": {
            "baseline": baseline_slices,
            "candidate": candidate_slices,
            "delta": slice_delta,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "decision_if_passed": "freeze_event_time_novelty_as_only_sidecar_test_candidate",
        "decision_if_failed": "do_not_read_test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
