from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores


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
