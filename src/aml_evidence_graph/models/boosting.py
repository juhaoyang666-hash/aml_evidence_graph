"""Native categorical preparation for optional LightGBM/XGBoost candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from aml_evidence_graph.compat import to_polars
from aml_evidence_graph.models.tabular import FeatureSpec


@dataclass(frozen=True)
class CategoryVocabulary:
    """Training-period-only category levels used by native boosting models."""

    levels: dict[str, tuple[str, ...]]


@dataclass
class TrainedLightGBMModel:
    """LightGBM primary table component with a frozen feature contract."""

    feature_spec: FeatureSpec
    category_vocabulary: CategoryVocabulary
    model: lgb.LGBMClassifier

    @property
    def primary_score_name(self) -> str:
        return "lightgbm"

    def predict_proba(self, frame: pl.DataFrame | pd.DataFrame) -> dict[str, np.ndarray]:
        prepared = prepare_native_boosting_frame(
            frame,
            self.feature_spec,
            self.category_vocabulary,
        )
        return {self.primary_score_name: self.model.predict_proba(prepared)[:, 1]}


def fit_category_vocabulary(
    frame: pl.DataFrame | pd.DataFrame,
    spec: FeatureSpec,
) -> CategoryVocabulary:
    """Freeze category levels without looking at validation or test values."""
    data = to_polars(frame)
    missing = sorted(set(spec.categorical_columns).difference(data.columns))
    if missing:
        raise ValueError("Training frame is missing categorical columns: " + ", ".join(missing))
    levels = {
        column: tuple(
            sorted(data[column].drop_nulls().cast(pl.Utf8).unique().to_list())
        )
        for column in spec.categorical_columns
    }
    return CategoryVocabulary(levels=levels)


def prepare_native_boosting_frame(
    frame: pl.DataFrame | pd.DataFrame,
    spec: FeatureSpec,
    vocabulary: CategoryVocabulary,
) -> pd.DataFrame:
    """Build a compact pandas frame with frozen categorical dtypes.

    Numeric columns are converted to float32. Categories unseen during training
    become missing values instead of extending the vocabulary from validation/test.
    LightGBM and XGBoost can both consume this pandas categorical representation.
    """
    data = to_polars(frame)
    missing = sorted(set(spec.all_columns).difference(data.columns))
    if missing:
        raise ValueError("Model input is missing columns: " + ", ".join(missing))
    missing_vocab = sorted(set(spec.categorical_columns).difference(vocabulary.levels))
    if missing_vocab:
        raise ValueError("Vocabulary is missing columns: " + ", ".join(missing_vocab))

    expressions: list[pl.Expr] = [
        pl.col(column).cast(pl.Float32, strict=False) for column in spec.numeric_columns
    ]
    expressions.extend(
        pl.col(column).cast(pl.Utf8, strict=False) for column in spec.categorical_columns
    )
    prepared = data.select(expressions).to_pandas()
    for column in spec.categorical_columns:
        prepared[column] = pd.Categorical(
            prepared[column],
            categories=list(vocabulary.levels[column]),
        )
    return prepared


def fit_lightgbm_for_partitions(
    train: pl.DataFrame | pd.DataFrame,
    validation: pl.DataFrame | pd.DataFrame,
    *,
    random_seed: int = 20260722,
    parameters: dict[str, Any] | None = None,
    excluded_feature_prefixes: tuple[str, ...] = ("graph_",),
) -> TrainedLightGBMModel:
    """Fit the promoted table model using train-only categories and validation PR-AUC."""
    from aml_evidence_graph.data.contract import CANONICAL
    from aml_evidence_graph.models.tabular import infer_feature_spec

    train_frame = to_polars(train)
    validation_frame = to_polars(validation)
    if train_frame.is_empty() or validation_frame.is_empty():
        raise ValueError("Both chronological train and validation frames are required.")
    for name, frame in (("train", train_frame), ("validation", validation_frame)):
        if CANONICAL.is_laundering not in frame.columns:
            raise ValueError(f"{name} partition requires canonical is_laundering labels.")
        if frame[CANONICAL.is_laundering].n_unique() < 2:
            raise ValueError(f"{name} partition must contain both label classes.")

    spec = infer_feature_spec(train_frame, excluded_prefixes=excluded_feature_prefixes)
    vocabulary = fit_category_vocabulary(train_frame, spec)
    train_x = prepare_native_boosting_frame(train_frame, spec, vocabulary)
    validation_x = prepare_native_boosting_frame(validation_frame, spec, vocabulary)
    train_y = train_frame[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    validation_y = validation_frame[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    positive_count = int(train_y.sum())
    negative_count = int(len(train_y) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training split must contain both label classes.")

    model_parameters: dict[str, Any] = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.05,
        "n_estimators": 1_200,
        "num_leaves": 63,
        "min_child_samples": 100,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": negative_count / positive_count,
        "random_state": random_seed,
        "n_jobs": 12,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    if parameters:
        forbidden = {"scale_pos_weight", "class_weight", "random_state", "random_seed"}
        if forbidden.intersection(parameters):
            raise ValueError(
                "Class weighting and random seed are centrally controlled for LightGBM."
            )
        model_parameters.update(parameters)
    model_parameters["random_state"] = random_seed
    model_parameters["scale_pos_weight"] = negative_count / positive_count
    model = lgb.LGBMClassifier(**model_parameters)
    model.fit(
        train_x,
        train_y,
        eval_X=validation_x,
        eval_y=validation_y,
        eval_metric="average_precision",
        categorical_feature="auto",
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return TrainedLightGBMModel(
        feature_spec=spec,
        category_vocabulary=vocabulary,
        model=model,
    )


def save_lightgbm_artifacts(models: TrainedLightGBMModel, model_dir: Path) -> None:
    """Persist a complete trusted-local bundle for batch serving."""
    if model_dir.exists():
        raise FileExistsError(f"LightGBM model directory already exists: {model_dir}")
    model_dir.mkdir(parents=True)
    joblib.dump(models.model, model_dir / "lightgbm.joblib", compress=3)
    models.model.booster_.save_model(
        model_dir / "lightgbm.txt",
        num_iteration=int(models.model.best_iteration_ or models.model.n_estimators),
    )
    (model_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "model_family": "lightgbm",
                "primary_score_name": models.primary_score_name,
                "numeric_columns": list(models.feature_spec.numeric_columns),
                "categorical_columns": list(models.feature_spec.categorical_columns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / "category_vocabulary.json").write_text(
        json.dumps(
            {name: list(levels) for name, levels in models.category_vocabulary.levels.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
