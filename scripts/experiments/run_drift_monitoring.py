#!/usr/bin/env python3
"""Time-slice drift drill: monthly PR-AUC / alert rate / ECE + threshold recalibration.

Uses frozen CatBoost / GAT / fusion test scores (no retrain). Recalibration only
touches the validation-fit isotonic threshold / calibrator — never test labels for
fitting. Writes artifacts/drift_* and prints a short JSON summary path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.evaluation.monitoring import monthly_stability_report
from aml_evidence_graph.models.fusion import fit_validation_calibration_and_threshold


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error += abs(float(labels[mask].mean()) - float(scores[mask].mean())) * mask.mean()
    return float(error)


def _alert_rate_at_threshold(scores: np.ndarray, threshold: float) -> float:
    return float((scores >= threshold).mean())


def _quantile_threshold(scores: np.ndarray, alert_fraction: float) -> float:
    alert_count = max(1, int(np.ceil(len(scores) * alert_fraction)))
    return float(np.sort(scores)[-alert_count])


def _month_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame[CANONICAL.event_ts], utc=True).dt.strftime("%Y-%m")


def _monthly_ops_table(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
    model_name: str,
) -> list[dict[str, Any]]:
    months = _month_series(frame)
    labels = frame[CANONICAL.is_laundering].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy()
        y = labels[mask]
        s = scores[mask]
        if y.sum() == 0 or y.sum() == len(y):
            metrics: dict[str, Any] = {
                "available": False,
                "sample_count": int(len(y)),
                "positive_count": int(y.sum()),
            }
        else:
            metrics = {"available": True, **evaluate_binary_risk_scores(y, s)}
        alerts = s >= threshold
        precision = float(y[alerts].mean()) if alerts.any() else 0.0
        recall = float(y[alerts].sum() / y.sum()) if y.sum() else 0.0
        rows.append(
            {
                "model": model_name,
                "month": month,
                "sample_count": int(len(y)),
                "positive_count": int(y.sum()),
                "positive_rate": float(y.mean()),
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "ece_10": _ece(y, np.clip(s, 0.0, 1.0)) if metrics.get("available") else None,
                "alert_rate_at_frozen_threshold": _alert_rate_at_threshold(s, threshold),
                "precision_at_frozen_threshold": precision,
                "recall_at_frozen_threshold": recall,
                "frozen_threshold": threshold,
                "available": bool(metrics.get("available", False)),
            }
        )
    return rows


def _pack_recalibration_metrics(
    scores: np.ndarray,
    threshold: float,
    policy: str,
    *,
    test_labels: np.ndarray,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize one frozen policy without closing over a model-loop iteration."""
    clipped_scores = np.clip(scores, 0.0, 1.0)
    metrics = evaluate_binary_risk_scores(test_labels, clipped_scores)
    alerts = scores >= threshold
    return {
        "policy": policy,
        "threshold": threshold,
        "test_pr_auc": metrics["pr_auc"],
        "test_ece_10": metrics["expected_calibration_error_10_bins"],
        "test_alert_rate": float(alerts.mean()),
        "test_precision_at_threshold": (
            float(test_labels[alerts].mean()) if alerts.any() else 0.0
        ),
        "test_recall_at_threshold": float(
            test_labels[alerts].sum() / test_labels.sum()
        ),
        "monthly": monthly_stability_report(
            test_frame.assign(**{CANONICAL.is_laundering: test_labels}),
            clipped_scores,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/drift_monitoring"))
    parser.add_argument("--alert-fraction", type=float, default=0.005)
    parser.add_argument(
        "--table-test",
        type=Path,
        default=Path("artifacts/table_baseline_rules/scores/table_test_scores.parquet"),
    )
    parser.add_argument(
        "--table-validation",
        type=Path,
        default=Path("artifacts/table_baseline_rules/scores/table_validation_scores.parquet"),
    )
    parser.add_argument(
        "--gat-test",
        type=Path,
        default=Path("artifacts/gat/scores/graphsage_test_scores.parquet"),
    )
    parser.add_argument(
        "--gat-validation",
        type=Path,
        default=Path("artifacts/gat/scores/graphsage_validation_scores.parquet"),
    )
    parser.add_argument(
        "--fusion-test",
        type=Path,
        default=Path("artifacts/fusion_test_cb_gat/test_fusion_scores.parquet"),
    )
    parser.add_argument(
        "--threshold-policy",
        type=Path,
        default=Path("artifacts/fusion_cb_gat/threshold_policy.json"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_test = pd.read_parquet(args.table_test)
    table_val = pd.read_parquet(args.table_validation)
    gat_test = pd.read_parquet(args.gat_test)
    gat_val = pd.read_parquet(args.gat_validation)
    fusion_test = pd.read_parquet(args.fusion_test)
    policy = json.loads(args.threshold_policy.read_text())
    alert_fraction = float(policy.get("alert_fraction", args.alert_fraction))
    fusion_threshold = float(policy["calibrated_threshold"])

    # Align GAT / table on transaction_id for fusion-style comparisons already done;
    # here we score each stream independently with its own frozen operational cut.
    catboost_val_scores = table_val["catboost"].to_numpy(dtype=float)
    catboost_test_scores = table_test["catboost"].to_numpy(dtype=float)
    gat_val_scores = np.clip(gat_val["graphsage"].to_numpy(dtype=float), 0.0, 1.0)
    gat_test_scores = np.clip(gat_test["graphsage"].to_numpy(dtype=float), 0.0, 1.0)
    fusion_test_scores = fusion_test["fusion_calibrated_probability"].to_numpy(dtype=float)

    catboost_threshold = _quantile_threshold(catboost_val_scores, alert_fraction)
    gat_threshold = _quantile_threshold(gat_val_scores, alert_fraction)

    monthly_rows: list[dict[str, Any]] = []
    monthly_rows.extend(
        _monthly_ops_table(
            table_test, catboost_test_scores, threshold=catboost_threshold, model_name="catboost"
        )
    )
    monthly_rows.extend(
        _monthly_ops_table(
            gat_test, gat_test_scores, threshold=gat_threshold, model_name="gat"
        )
    )
    monthly_rows.extend(
        _monthly_ops_table(
            fusion_test,
            fusion_test_scores,
            threshold=fusion_threshold,
            model_name="fusion_cb_gat",
        )
    )
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(args.output_dir / "drift_monthly_curves.csv", index=False)
    monthly_df.to_parquet(args.output_dir / "drift_monthly_curves.parquet", index=False)

    # --- Threshold recalibration drill (validation only for fitting) ---
    # Stale: fit isotonic + threshold on first validation month only.
    # Fresh: fit on full validation (project policy).
    val_months = _month_series(table_val)
    ordered_val_months = sorted(val_months.unique())
    first_val_month = ordered_val_months[0]

    recalibration: dict[str, Any] = {
        "alert_fraction": alert_fraction,
        "stale_fit_month": first_val_month,
        "policy_fusion_threshold": fusion_threshold,
        "catboost": {},
        "gat": {},
        "note": (
            "Stale = isotonic+threshold fit on first validation month only; "
            "fresh = fit on full validation. Test labels used only for reporting."
        ),
    }

    def apply_cal(cal, scores: np.ndarray) -> np.ndarray:
        return np.asarray(cal.predict_proba(scores), dtype=float)

    for name, val_frame, val_raw, test_frame, test_raw in (
        ("catboost", table_val, catboost_val_scores, table_test, catboost_test_scores),
        ("gat", gat_val, gat_val_scores, gat_test, gat_test_scores),
    ):
        # Refit stale/fresh per model on that model's validation raw scores.
        early = _month_series(val_frame).eq(first_val_month)
        stale = fit_validation_calibration_and_threshold(
            val_raw[early.to_numpy()],
            val_frame.loc[early, CANONICAL.is_laundering],
            alert_fraction=alert_fraction,
            method="isotonic",
        )
        fresh = fit_validation_calibration_and_threshold(
            val_raw,
            val_frame[CANONICAL.is_laundering],
            alert_fraction=alert_fraction,
            method="isotonic",
        )
        test_y = test_frame[CANONICAL.is_laundering].astype(int).to_numpy()
        stale_test = apply_cal(stale, test_raw)
        fresh_test = apply_cal(fresh, test_raw)
        # Also: raw quantile threshold without recalibration (raw score space).
        raw_thr = _quantile_threshold(val_raw, alert_fraction)

        recalibration[name] = {
            "stale_threshold": stale.threshold,
            "fresh_threshold": fresh.threshold,
            "raw_quantile_threshold": raw_thr,
            "policies": [
                _pack_recalibration_metrics(
                    stale_test,
                    stale.threshold,
                    "stale_early_val_month",
                    test_labels=test_y,
                    test_frame=test_frame,
                ),
                _pack_recalibration_metrics(
                    fresh_test,
                    fresh.threshold,
                    "fresh_full_validation",
                    test_labels=test_y,
                    test_frame=test_frame,
                ),
                _pack_recalibration_metrics(
                    test_raw,
                    raw_thr,
                    "raw_quantile_no_isotonic",
                    test_labels=test_y,
                    test_frame=test_frame,
                ),
            ],
        }
        # Expanding-window: fit calibrator on validation months ≤ M, score the next month
        # (last step: first test month). Labels on the holdout month are evaluation-only.
        expanding_rows = []
        test_months = sorted(_month_series(test_frame).unique())
        for idx, month in enumerate(ordered_val_months):
            fit_mask = _month_series(val_frame).isin(ordered_val_months[: idx + 1]).to_numpy()
            cal = fit_validation_calibration_and_threshold(
                val_raw[fit_mask],
                val_frame.loc[fit_mask, CANONICAL.is_laundering],
                alert_fraction=alert_fraction,
                method="isotonic",
            )
            if idx + 1 < len(ordered_val_months):
                holdout_month = ordered_val_months[idx + 1]
                hold_mask = _month_series(val_frame).eq(holdout_month).to_numpy()
                hold_y = val_frame.loc[hold_mask, CANONICAL.is_laundering].astype(int).to_numpy()
                hold_s = apply_cal(cal, val_raw[hold_mask])
            else:
                holdout_month = test_months[0]
                hold_mask = _month_series(test_frame).eq(holdout_month).to_numpy()
                hold_y = test_frame.loc[hold_mask, CANONICAL.is_laundering].astype(int).to_numpy()
                hold_s = apply_cal(cal, test_raw[hold_mask])
            alerts = hold_s >= cal.threshold
            expanding_rows.append(
                {
                    "model": name,
                    "fit_through_month": month,
                    "holdout_month": holdout_month,
                    "threshold": cal.threshold,
                    "holdout_alert_rate": float(alerts.mean()),
                    "holdout_ece_10": _ece(hold_y, np.clip(hold_s, 0.0, 1.0))
                    if len(np.unique(hold_y)) > 1
                    else None,
                    "holdout_pr_auc": float(
                        evaluate_binary_risk_scores(hold_y, np.clip(hold_s, 0.0, 1.0))["pr_auc"]
                    )
                    if len(np.unique(hold_y)) > 1
                    else None,
                    "holdout_precision": float(hold_y[alerts].mean()) if alerts.any() else 0.0,
                    "holdout_recall": float(hold_y[alerts].sum() / hold_y.sum())
                    if hold_y.sum()
                    else 0.0,
                }
            )
        recalibration[name]["expanding_window"] = expanding_rows

    # Fusion: compare frozen policy threshold vs re-quantile on validation calibrated scores.
    # Validation fusion scores are not stored as a single file; reconstruct raw fusion is heavy.
    # Report test monthly under the stored policy threshold only + oracle monthly quantile.
    fusion_oracle_rows = []
    months = _month_series(fusion_test)
    y_all = fusion_test[CANONICAL.is_laundering].astype(int).to_numpy()
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy()
        s = fusion_test_scores[mask]
        y = y_all[mask]
        oracle_thr = _quantile_threshold(s, alert_fraction)
        frozen_alerts = s >= fusion_threshold
        oracle_alerts = s >= oracle_thr
        fusion_oracle_rows.append(
            {
                "month": month,
                "frozen_threshold": fusion_threshold,
                "oracle_month_quantile_threshold": oracle_thr,
                "frozen_alert_rate": float(frozen_alerts.mean()),
                "oracle_alert_rate": float(oracle_alerts.mean()),
                "frozen_precision": float(y[frozen_alerts].mean()) if frozen_alerts.any() else 0.0,
                "oracle_precision": float(y[oracle_alerts].mean()) if oracle_alerts.any() else 0.0,
                "frozen_recall": float(y[frozen_alerts].sum() / y.sum()) if y.sum() else 0.0,
                "oracle_recall": float(y[oracle_alerts].sum() / y.sum()) if y.sum() else 0.0,
                "pr_auc": float(evaluate_binary_risk_scores(y, s)["pr_auc"])
                if len(np.unique(y)) > 1
                else None,
            }
        )
    recalibration["fusion_cb_gat_oracle_vs_frozen"] = fusion_oracle_rows

    summary = {
        "protocol": {
            "dataset": "SAML-D synthetic",
            "test_months": sorted(_month_series(table_test).unique().tolist()),
            "validation_months": ordered_val_months,
            "alert_fraction": alert_fraction,
            "models": ["catboost", "gat", "fusion_cb_gat"],
            "honest_boundary": (
                "Frozen scores from existing main-line runs; no monthly retrain. "
                "Recalibration uses validation labels only. Synthetic data; "
                "not production monitoring."
            ),
        },
        "monthly_curve_rows": len(monthly_df),
        "recalibration": recalibration,
    }
    (args.output_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    pd.DataFrame(fusion_oracle_rows).to_csv(
        args.output_dir / "drift_fusion_oracle_vs_frozen.csv", index=False
    )
    expanding = []
    for name in ("catboost", "gat"):
        expanding.extend(recalibration[name]["expanding_window"])
    pd.DataFrame(expanding).to_csv(args.output_dir / "drift_expanding_window.csv", index=False)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "monthly_rows": len(monthly_df)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
