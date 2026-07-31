#!/usr/bin/env python3
"""B2: distill GAT OOF scores into CatBoost as an extra feature (train-safe).

Train: join artifacts/graph_oof_gat graphsage column as gat_oof onto PIT train.
Val/Test: join frozen artifacts/gat scores (full-model inference), never test labels
for fitting. Compare to main-line CatBoost 0.809.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.models.tabular import fit_table_models_for_partitions
from aml_evidence_graph.training.table_baseline import (
    deterministic_negative_downsample,
    load_feature_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/pit_features"))
    parser.add_argument(
        "--graph-oof",
        type=Path,
        default=Path("artifacts/graph_oof_gat/graphsage_oof_scores.parquet"),
    )
    parser.add_argument(
        "--gat-validation",
        type=Path,
        default=Path("artifacts/gat/scores/graphsage_validation_scores.parquet"),
    )
    parser.add_argument(
        "--gat-test",
        type=Path,
        default=Path("artifacts/gat/scores/graphsage_test_scores.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/table_baseline_gat_distill"),
    )
    parser.add_argument("--max-train-negatives", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores").mkdir(parents=True, exist_ok=True)

    train = load_feature_split(args.features, TimeSplit.TRAIN)
    val = load_feature_split(args.features, TimeSplit.VALIDATION)
    test = load_feature_split(args.features, TimeSplit.TEST)

    oof = pd.read_parquet(args.graph_oof)[[CANONICAL.transaction_id, "graphsage"]].rename(
        columns={"graphsage": "gat_oof"}
    )
    val_g = pd.read_parquet(args.gat_validation)[[CANONICAL.transaction_id, "graphsage"]].rename(
        columns={"graphsage": "gat_oof"}
    )
    test_g = pd.read_parquet(args.gat_test)[[CANONICAL.transaction_id, "graphsage"]].rename(
        columns={"graphsage": "gat_oof"}
    )

    train = train.merge(oof, on=CANONICAL.transaction_id, how="inner")
    val = val.merge(val_g, on=CANONICAL.transaction_id, how="inner")
    test = test.merge(test_g, on=CANONICAL.transaction_id, how="inner")
    for frame in (train, val, test):
        frame["gat_oof"] = np.clip(frame["gat_oof"].astype(float), 0.0, 1.0)

    train_ds = deterministic_negative_downsample(
        train, maximum_negative_rows=args.max_train_negatives
    )
    models = fit_table_models_for_partitions(
        train_ds,
        val,
        excluded_feature_prefixes=("graph_",),
        random_seed=args.seed,
    )
    preds = models.predict_proba(test)
    metrics = evaluate_binary_risk_scores(
        test[CANONICAL.is_laundering].astype(int), preds["catboost"]
    )
    metrics.pop("curves", None)
    summary = {
        "protocol": {
            "teacher": "GAT OOF on train / frozen GAT on val+test",
            "train_rows": int(len(train_ds)),
            "feature_includes_gat_oof": True,
            "honest_boundary": (
                "Feature distillation only; does not replace GAT alone or main-line CatBoost "
                "unless it clearly wins. Synthetic SAML-D."
            ),
        },
        "test_metrics": metrics,
        "reference_catboost": 0.8092,
        "reference_gat": 0.9483,
        "beats_catboost": bool(metrics["pr_auc"] > 0.8092 + 1e-4),
        "beats_gat": bool(metrics["pr_auc"] > 0.9483),
    }
    output_columns = [
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.is_laundering,
    ]
    out_scores = test[output_columns].copy()
    out_scores["catboost_gat_distill"] = preds["catboost"]
    out_scores.to_parquet(args.output_dir / "scores" / "table_test_scores.parquet", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    print(json.dumps({"test_pr_auc": metrics["pr_auc"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
