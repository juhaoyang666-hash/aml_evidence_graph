"""Metrics for highly imbalanced AML transaction-risk models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _validate_labels_and_scores(
    y_true: Iterable[int],
    probabilities: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and probabilities must be one-dimensional and have equal length.")
    if len(labels) == 0:
        raise ValueError("Labels and probabilities must not be empty.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if len(np.unique(labels)) < 2:
        raise ValueError("Both label classes are required for ranking metrics.")
    return labels, scores


def _recall_at_fixed_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    eligible = true_positive_rate[false_positive_rate <= target_fpr]
    return float(eligible.max()) if len(eligible) else 0.0


def _kolmogorov_smirnov_statistic(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    return float(np.max(true_positive_rate - false_positive_rate))


def _expected_calibration_error(labels: np.ndarray, scores: np.ndarray, *, bins: int = 10) -> float:
    bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error += abs(float(labels[mask].mean()) - float(scores[mask].mean())) * mask.mean()
    return float(error)


def _downsample_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    maximum_points: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound serialized curve size while retaining endpoints for report rendering."""
    if maximum_points < 2:
        raise ValueError("maximum_points must be at least two.")
    if len(x_values) <= maximum_points:
        return x_values, y_values
    indices = np.unique(
        np.linspace(0, len(x_values) - 1, maximum_points, dtype=int)
    )
    return x_values[indices], y_values[indices]


def evaluate_binary_risk_scores(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    alert_budget_fractions: tuple[float, ...] = (0.001, 0.005, 0.01),
    fixed_fpr_targets: tuple[float, ...] = (0.0001, 0.0005, 0.001),
) -> dict[str, Any]:
    """Evaluate calibrated probabilities without resampling the input set."""
    labels, scores = _validate_labels_and_scores(y_true, probabilities)
    result: dict[str, Any] = {
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "ks_statistic": _kolmogorov_smirnov_statistic(labels, scores),
        "brier_score": float(brier_score_loss(labels, scores)),
        "expected_calibration_error_10_bins": _expected_calibration_error(labels, scores),
        "alert_budgets": {},
        "recall_at_fixed_fpr": {},
    }
    ordered_labels = labels[np.argsort(-scores, kind="stable")]
    positive_total = int(labels.sum())
    for fraction in alert_budget_fractions:
        if not 0 < fraction <= 1:
            raise ValueError("alert_budget_fractions must be in (0, 1].")
        budget = max(1, int(np.ceil(len(labels) * fraction)))
        top_labels = ordered_labels[:budget]
        found = int(top_labels.sum())
        precision = found / budget
        recall = found / positive_total
        result["alert_budgets"][f"{fraction:.4%}"] = {
            "alert_count": budget,
            "true_positives_found": found,
            "precision_at_k": float(precision),
            "recall_at_k": float(recall),
            "alerts_per_true_positive": float(budget / found) if found else None,
            "unfound_positive_count": positive_total - found,
        }
    for target in fixed_fpr_targets:
        if not 0 <= target <= 1:
            raise ValueError("fixed_fpr_targets must be in [0, 1].")
        result["recall_at_fixed_fpr"][f"{target:.4%}"] = _recall_at_fixed_fpr(
            labels,
            scores,
            target,
        )

    curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
    curve_precision, curve_recall = _downsample_curve(
        curve_precision,
        curve_recall,
    )
    calibration_observed, calibration_predicted = calibration_curve(labels, scores, n_bins=10)
    result["curves"] = {
        "precision_recall": {
            "precision": curve_precision.tolist(),
            "recall": curve_recall.tolist(),
        },
        "calibration": {
            "predicted_probability": calibration_predicted.tolist(),
            "observed_positive_rate": calibration_observed.tolist(),
        },
    }
    return result


def _alerts_needed_for_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    target_recall: float,
) -> dict[str, Any]:
    """Return the smallest top-K alert budget that reaches the target recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1].")
    positive_total = int(labels.sum())
    if positive_total == 0:
        raise ValueError("At least one positive label is required.")
    needed_positives = int(np.ceil(positive_total * target_recall))
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(labels[order])
    hit_indices = np.flatnonzero(cumulative >= needed_positives)
    if len(hit_indices) == 0:
        return {
            "achieved": False,
            "alert_count": int(len(labels)),
            "true_positives_found": int(cumulative[-1]),
            "achieved_recall": float(cumulative[-1] / positive_total),
        }
    alert_count = int(hit_indices[0]) + 1
    found = int(cumulative[alert_count - 1])
    return {
        "achieved": True,
        "alert_count": alert_count,
        "true_positives_found": found,
        "achieved_recall": float(found / positive_total),
    }


def compare_alert_volume_at_fixed_recall(
    y_true: Iterable[int],
    model_probabilities: Iterable[float],
    baseline_scores: Iterable[float],
    *,
    recall_targets: tuple[float, ...] = (0.50, 0.70, 0.85),
) -> dict[str, Any]:
    """Compare model vs baseline alert volume at fixed recall targets.

    This is the primary AML operations KPI: at the same recall, how many fewer
    alerts does the model raise relative to a rule (or other) baseline.
    """
    labels, model_scores = _validate_labels_and_scores(y_true, model_probabilities)
    _, baseline = _validate_labels_and_scores(y_true, baseline_scores)
    comparisons: dict[str, Any] = {}
    for target in recall_targets:
        model_need = _alerts_needed_for_recall(labels, model_scores, target_recall=target)
        baseline_need = _alerts_needed_for_recall(labels, baseline, target_recall=target)
        baseline_alerts = baseline_need["alert_count"]
        model_alerts = model_need["alert_count"]
        reduction = None
        if baseline_need["achieved"] and baseline_alerts > 0 and model_need["achieved"]:
            reduction = float(1.0 - (model_alerts / baseline_alerts))
        comparisons[f"recall_{target:.0%}"] = {
            "target_recall": target,
            "model": model_need,
            "baseline": baseline_need,
            "alert_reduction_rate": reduction,
            "alerts_saved": (
                int(baseline_alerts - model_alerts)
                if baseline_need["achieved"] and model_need["achieved"]
                else None
            ),
        }
    return {
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "comparisons": comparisons,
    }
