"""Table-model attribution artifacts for review; attribution is not causality."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from catboost import Pool

from aml_evidence_graph.compat import to_polars
from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.models.boosting import (
    TrainedLightGBMModel,
    prepare_native_boosting_frame,
)
from aml_evidence_graph.models.tabular import TrainedTableModels, prepare_feature_frames


def write_lightgbm_explanations(
    models: TrainedLightGBMModel,
    frame: pl.DataFrame,
    output_dir: Path,
    *,
    model_name: str = "lightgbm",
    max_rows: int = 200,
) -> dict[str, object]:
    """Write global gain importance and bounded local SHAP contributions."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = to_polars(frame)
    sample = prepare_native_boosting_frame(
        data.head(max_rows),
        models.feature_spec,
        models.category_vocabulary,
    )
    transaction_ids = data.head(len(sample))[CANONICAL.transaction_id].cast(pl.Utf8).to_list()
    contributions = models.model.booster_.predict(
        sample,
        pred_contrib=True,
        num_iteration=models.model.best_iteration_,
    )
    feature_names = list(models.feature_spec.all_columns)
    local = pl.from_numpy(contributions[:, :-1], schema=feature_names).with_columns(
        pl.Series(CANONICAL.transaction_id, transaction_ids),
        pl.Series("base_value", contributions[:, -1]),
    ).select([CANONICAL.transaction_id, *feature_names, "base_value"])
    local_path = output_dir / f"{model_name}_local_shap.parquet"
    local.write_parquet(local_path)

    importance = models.model.booster_.feature_importance(
        importance_type="gain",
        iteration=models.model.best_iteration_,
    )
    global_importance = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(feature_names, importance, strict=True),
            key=lambda item: -item[1],
        )
    ]
    global_path = output_dir / f"{model_name}_global_importance.json"
    global_path.write_text(
        json.dumps(
            {
                "importance_type": "gain",
                "interpretation_limit": (
                    "SHAP contributions describe this model's local attribution; "
                    "they do not establish causal AML behavior."
                ),
                "feature_importance": global_importance,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "local_path": str(local_path),
        "global_path": str(global_path),
        "sample_count": local.height,
    }


def write_catboost_explanations(
    models: TrainedTableModels,
    frame: pl.DataFrame,
    output_dir: Path,
    *,
    model_name: str,
    max_rows: int = 200,
) -> dict[str, object]:
    """Write global importance and bounded local SHAP rows to private artifacts."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = to_polars(frame)
    _, catboost_frame = prepare_feature_frames(data, models.feature_spec)
    sample = catboost_frame.iloc[:max_rows].copy()
    transaction_ids = (
        data.head(len(sample))[CANONICAL.transaction_id].cast(pl.Utf8).to_list()
    )
    pool = Pool(sample, cat_features=list(models.feature_spec.categorical_columns))
    shap_values = models.catboost.get_feature_importance(pool, type="ShapValues")
    feature_names = list(models.feature_spec.all_columns)
    local = pl.from_numpy(shap_values[:, :-1], schema=feature_names).with_columns(
        pl.Series(CANONICAL.transaction_id, transaction_ids),
        pl.Series("base_value", shap_values[:, -1]),
    ).select([CANONICAL.transaction_id, *feature_names, "base_value"])
    local_path = output_dir / f"{model_name}_local_shap.parquet"
    local.write_parquet(local_path)

    importance = models.catboost.get_feature_importance(type="FeatureImportance")
    global_importance = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(feature_names, importance, strict=True),
            key=lambda item: -item[1],
        )
    ]
    global_path = output_dir / f"{model_name}_global_importance.json"
    global_path.write_text(
        json.dumps(
            {
                "interpretation_limit": (
                    "SHAP values describe this model's local attribution; "
                    "they do not establish causal AML behavior."
                ),
                "feature_importance": global_importance,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "local_path": str(local_path),
        "global_path": str(global_path),
        "sample_count": local.height,
    }
