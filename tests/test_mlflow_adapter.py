from aml_evidence_graph.tracking.mlflow_adapter import (
    candidate_gate,
    flatten_numeric_metrics,
)


def test_flatten_numeric_metrics_ignores_non_metrics() -> None:
    payload = {"validation": {"pr_auc": 0.4, "ok": True}, "rows": [1, 2], "name": "x"}
    assert flatten_numeric_metrics(payload) == {"validation.pr_auc": 0.4}


def test_candidate_gate_uses_validation_metric() -> None:
    assert candidate_gate({"pr_auc": 0.42}, {"pr_auc": 0.4}, minimum_relative_gain=0.05)[0]
    assert not candidate_gate({"pr_auc": 0.39}, {"pr_auc": 0.4})[0]
