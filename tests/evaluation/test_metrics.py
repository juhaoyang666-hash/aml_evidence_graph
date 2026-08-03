from aml_evidence_graph.evaluation.metrics import (
    compare_alert_volume_at_fixed_recall,
    evaluate_binary_risk_scores,
)


def test_evaluate_binary_risk_scores_uses_probabilities_and_alert_budget() -> None:
    metrics = evaluate_binary_risk_scores(
        [0, 0, 1, 1],
        [0.01, 0.10, 0.80, 0.95],
        alert_budget_fractions=(0.5,),
        fixed_fpr_targets=(0.0,),
    )

    budget = metrics["alert_budgets"]["50.0000%"]
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["ks_statistic"] == 1.0
    assert budget["true_positives_found"] == 2
    assert budget["recall_at_k"] == 1.0


def test_precision_recall_curve_serialization_is_bounded() -> None:
    labels = [0, 1] * 1_500
    scores = [index / 3_000 for index in range(3_000)]

    metrics = evaluate_binary_risk_scores(labels, scores)

    assert len(metrics["curves"]["precision_recall"]["precision"]) <= 2_000


def test_compare_alert_volume_at_fixed_recall_reports_reduction() -> None:
    labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # Model ranks both positives first.
    model_scores = [0.99, 0.98, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    # Baseline needs many more alerts to reach the same recall.
    baseline_scores = [0.10, 0.09, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88]

    comparison = compare_alert_volume_at_fixed_recall(
        labels,
        model_scores,
        baseline_scores,
        recall_targets=(1.0,),
    )

    row = comparison["comparisons"]["recall_100%"]
    assert row["model"]["alert_count"] == 2
    assert row["baseline"]["alert_count"] == 10
    assert row["alert_reduction_rate"] == 0.8
    assert row["alerts_saved"] == 8
