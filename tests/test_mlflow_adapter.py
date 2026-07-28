import json
from pathlib import Path

import pytest

from aml_evidence_graph.tracking.mlflow_adapter import (
    candidate_gate,
    flatten_numeric_metrics,
    log_completed_run,
)


def test_flatten_numeric_metrics_ignores_non_metrics() -> None:
    payload = {"validation": {"pr_auc": 0.4, "ok": True}, "rows": [1, 2], "name": "x"}
    assert flatten_numeric_metrics(payload) == {"validation.pr_auc": 0.4}


def test_flatten_numeric_metrics_sanitizes_mlflow_names() -> None:
    flattened = flatten_numeric_metrics(
        {"alert_budgets": {"0.1000%": {"precision@k": 0.8}}}
    )

    assert flattened == {"alert_budgets.0.1000pct.precision_k": 0.8}


def test_candidate_gate_uses_validation_metric() -> None:
    assert candidate_gate({"pr_auc": 0.42}, {"pr_auc": 0.4}, minimum_relative_gain=0.05)[0]
    assert not candidate_gate({"pr_auc": 0.39}, {"pr_auc": 0.4})[0]
    with pytest.raises(ValueError, match="test metrics"):
        candidate_gate(
            {"nested_test_pr_auc": 0.9},
            {"nested_test_pr_auc": 0.8},
            metric="nested_test_pr_auc",
        )


def test_mlflow_sync_rejects_unproven_completion(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "run_manifest.json").write_text(
        json.dumps({"run_id": "legacy-run"}), encoding="utf-8"
    )
    (artifact / "metrics.json").write_text(
        json.dumps({"validation_metrics": {"pr_auc": 0.8}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="completed pipeline status"):
        log_completed_run(
            artifact,
            experiment_name="test",
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        )
