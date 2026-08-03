#!/usr/bin/env python3
"""Nonlinear OOF fusion ablation (HistGBDT / MLP) vs logistic main-line fusion.

Fits only on training OOF scores (catboost + GAT column `graphsage`), calibrates
threshold on validation, scores test. Does not replace the logistic fusion
main-line number unless it wins — report honestly either way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.models.fusion import (
    fit_oof_fusion,
    fit_validation_calibration_and_threshold,
)

MODEL_NAMES = ("catboost", "graphsage")


def _merge_oof(table_oof: Path, graph_oof: Path) -> pd.DataFrame:
    left = pd.read_parquet(table_oof)
    right = pd.read_parquet(graph_oof)
    merged = left.merge(
        right[[CANONICAL.transaction_id, "graphsage", "oof_fold_id"]],
        on=CANONICAL.transaction_id,
        how="inner",
        suffixes=("", "_graph"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"OOF merge size mismatch: table={len(left)} graph={len(right)} merged={len(merged)}"
        )
    return merged


def _align_test_components(table_test: Path, gat_test: Path) -> pd.DataFrame:
    left = pd.read_parquet(table_test)
    right = pd.read_parquet(gat_test)
    merged = left.merge(
        right[[CANONICAL.transaction_id, "graphsage"]],
        on=CANONICAL.transaction_id,
        how="inner",
    )
    if len(merged) != len(left):
        raise ValueError("Test component merge mismatch.")
    return merged


def _fit_nonlinear(kind: str, x: np.ndarray, y: np.ndarray, seed: int):
    if kind == "hist_gbdt":
        model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            random_state=seed,
        )
        model.fit(x, y)
        return model
    if kind == "mlp":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        max_iter=200,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x, y)
        return model
    raise ValueError(kind)


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/nonlinear_fusion"))
    parser.add_argument(
        "--table-oof",
        type=Path,
        default=Path("artifacts/table_oof/table_oof_scores.parquet"),
    )
    parser.add_argument(
        "--graph-oof",
        type=Path,
        default=Path("artifacts/graph_oof_gat/graphsage_oof_scores.parquet"),
    )
    parser.add_argument(
        "--table-validation",
        type=Path,
        default=Path("artifacts/table_baseline_rules/scores/table_validation_scores.parquet"),
    )
    parser.add_argument(
        "--gat-validation",
        type=Path,
        default=Path("artifacts/gat/scores/graphsage_validation_scores.parquet"),
    )
    parser.add_argument(
        "--table-test",
        type=Path,
        default=Path("artifacts/table_baseline_rules/scores/table_test_scores.parquet"),
    )
    parser.add_argument(
        "--gat-test",
        type=Path,
        default=Path("artifacts/gat/scores/graphsage_test_scores.parquet"),
    )
    parser.add_argument("--alert-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof = _merge_oof(args.table_oof, args.graph_oof)
    # Clip GAT column to [0,1] for logistic path; nonlinear can use raw clipped.
    for col in MODEL_NAMES:
        oof[col] = np.clip(oof[col].to_numpy(dtype=float), 0.0, 1.0)

    val = pd.read_parquet(args.table_validation).merge(
        pd.read_parquet(args.gat_validation)[[CANONICAL.transaction_id, "graphsage"]],
        on=CANONICAL.transaction_id,
        how="inner",
    )
    test = _align_test_components(args.table_test, args.gat_test)
    for frame in (val, test):
        frame["catboost"] = np.clip(frame["catboost"].to_numpy(dtype=float), 0.0, 1.0)
        frame["graphsage"] = np.clip(frame["graphsage"].to_numpy(dtype=float), 0.0, 1.0)

    x_oof = oof.loc[:, list(MODEL_NAMES)].to_numpy(dtype=float)
    y_oof = oof[CANONICAL.is_laundering].astype(int).to_numpy()
    x_val = val.loc[:, list(MODEL_NAMES)].to_numpy(dtype=float)
    y_val = val[CANONICAL.is_laundering].astype(int).to_numpy()
    x_test = test.loc[:, list(MODEL_NAMES)].to_numpy(dtype=float)
    y_test = test[CANONICAL.is_laundering].astype(int).to_numpy()

    logistic = fit_oof_fusion(
        oof.loc[:, list(MODEL_NAMES)],
        oof[CANONICAL.is_laundering],
        model_names=MODEL_NAMES,
        random_seed=args.seed,
    )
    models: dict[str, Any] = {
        "logistic": logistic,
        "hist_gbdt": _fit_nonlinear("hist_gbdt", x_oof, y_oof, args.seed),
        "mlp": _fit_nonlinear("mlp", x_oof, y_oof, args.seed),
    }

    summary: dict[str, Any] = {
        "protocol": {
            "components": list(MODEL_NAMES),
            "oof_rows": int(len(oof)),
            "alert_fraction": args.alert_fraction,
            "honest_boundary": (
                "Nonlinear heads fitted on train OOF only; calibration/threshold on validation. "
                "Main-line remains logistic fusion unless this ablation clearly wins. "
                "Synthetic SAML-D."
            ),
        },
        "models": {},
    }
    score_cols = {
        CANONICAL.transaction_id: test[CANONICAL.transaction_id],
        CANONICAL.is_laundering: y_test,
    }

    for name, model in models.items():
        if name == "logistic":
            val_raw = logistic.predict_proba(val.loc[:, list(MODEL_NAMES)])
            test_raw = logistic.predict_proba(test.loc[:, list(MODEL_NAMES)])
        else:
            val_raw = _predict(model, x_val)
            test_raw = _predict(model, x_test)
        cal = fit_validation_calibration_and_threshold(
            val_raw,
            y_val,
            alert_fraction=args.alert_fraction,
            method="isotonic",
        )
        val_cal = cal.predict_proba(val_raw)
        test_cal = cal.predict_proba(test_raw)
        val_metrics = evaluate_binary_risk_scores(y_val, val_cal)
        test_metrics = evaluate_binary_risk_scores(y_test, test_cal)
        alerts = test_cal >= cal.threshold
        summary["models"][name] = {
            "validation": {
                "pr_auc": val_metrics["pr_auc"],
                "ece_10": val_metrics["expected_calibration_error_10_bins"],
            },
            "test": {
                "pr_auc": test_metrics["pr_auc"],
                "roc_auc": test_metrics["roc_auc"],
                "ece_10": test_metrics["expected_calibration_error_10_bins"],
                "alert_budgets": test_metrics["alert_budgets"],
                "alert_rate_at_threshold": float(alerts.mean()),
                "precision_at_threshold": float(y_test[alerts].mean()) if alerts.any() else 0.0,
                "recall_at_threshold": float(y_test[alerts].sum() / y_test.sum()),
            },
            "threshold": cal.threshold,
            "calibration_method": cal.method,
        }
        score_cols[f"{name}_raw"] = test_raw
        score_cols[f"{name}_calibrated"] = test_cal

    logistic_pr = summary["models"]["logistic"]["test"]["pr_auc"]
    best_nonlin = max(
        ("hist_gbdt", "mlp"),
        key=lambda n: summary["models"][n]["test"]["pr_auc"],
    )
    summary["verdict"] = {
        "mainline_logistic_test_pr_auc": logistic_pr,
        "best_nonlinear": best_nonlin,
        "best_nonlinear_test_pr_auc": summary["models"][best_nonlin]["test"]["pr_auc"],
        "beats_logistic": bool(
            summary["models"][best_nonlin]["test"]["pr_auc"] > logistic_pr + 1e-4
        ),
        "beats_gat_alone_0_9483": bool(
            summary["models"][best_nonlin]["test"]["pr_auc"] > 0.9483
        ),
        "reference_mainline_fusion_artifact_pr_auc": 0.9175,
    }

    pd.DataFrame(score_cols).to_parquet(
        args.output_dir / "nonlinear_fusion_test_scores.parquet", index=False
    )
    (args.output_dir / "nonlinear_fusion_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    print(json.dumps(summary["verdict"] | {"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
