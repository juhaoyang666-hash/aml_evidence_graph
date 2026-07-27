"""Chronological Logistic Regression and CatBoost baselines for transaction risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aml_evidence_graph.compat import is_numeric_dtype, to_pandas, to_polars
from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit

LEAKAGE_COLUMNS = {
    CANONICAL.transaction_id,
    CANONICAL.event_ts,
    CANONICAL.sender_account_id,
    CANONICAL.receiver_account_id,
    CANONICAL.is_laundering,
    CANONICAL.laundering_type,
    CANONICAL.source_row_number,
    "split",
    "event_date",
}


@dataclass(frozen=True)
class FeatureSpec:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns


@dataclass
class TrainedTableModels:
    feature_spec: FeatureSpec
    logistic: Pipeline
    catboost: CatBoostClassifier

    def predict_proba(self, frame: pl.DataFrame | pd.DataFrame) -> dict[str, np.ndarray]:
        """Return positive-class probabilities for both deterministic baselines."""
        logistic_frame, catboost_frame = prepare_feature_frames(frame, self.feature_spec)
        return {
            "logistic": self.logistic.predict_proba(logistic_frame)[:, 1],
            "catboost": self.catboost.predict_proba(catboost_frame)[:, 1],
        }


def infer_feature_spec(
    frame: pl.DataFrame | pd.DataFrame,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> FeatureSpec:
    """Select model features while explicitly excluding labels and identifiers."""
    data = to_polars(frame)
    candidates = [
        column
        for column in data.columns
        if column not in LEAKAGE_COLUMNS
        and not any(column.startswith(prefix) for prefix in excluded_prefixes)
    ]
    if not candidates:
        raise ValueError("No model features remain after excluding labels and identifiers.")
    numeric = tuple(
        column for column in candidates if is_numeric_dtype(data.schema[column])
    )
    categorical = tuple(column for column in candidates if column not in numeric)
    return FeatureSpec(numeric_columns=numeric, categorical_columns=categorical)


def prepare_feature_frames(
    frame: pl.DataFrame | pd.DataFrame,
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare matching feature frames without fitting transformations on validation/test.

    Returns pandas DataFrames because sklearn and CatBoost still require them.
    """
    data = to_polars(frame)
    missing = sorted(set(spec.all_columns).difference(data.columns))
    if missing:
        raise ValueError(f"Model input is missing columns: {', '.join(missing)}")
    feature_columns = list(spec.all_columns)
    prepared = data.select(feature_columns)
    for column in spec.categorical_columns:
        prepared = prepared.with_columns(pl.col(column).cast(pl.Utf8))
    logistic_frame = to_pandas(prepared)
    catboost_frame = to_pandas(
        prepared.with_columns(
            [
                pl.col(column).fill_null("__MISSING__")
                for column in spec.categorical_columns
            ]
        )
    )
    return logistic_frame, catboost_frame


def _build_logistic_pipeline(spec: FeatureSpec, *, random_seed: int) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(spec.numeric_columns),
            )
        )
    if spec.categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(spec.categorical_columns),
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _class_weights(labels: pl.Series | pd.Series | np.ndarray) -> list[float]:
    values = np.asarray(labels, dtype=int)
    positive_count = int(values.sum())
    negative_count = int(len(values) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training split must contain both positive and negative examples.")
    return [1.0, negative_count / positive_count]


def fit_table_models(
    frame: pl.DataFrame | pd.DataFrame,
    *,
    random_seed: int = 20260722,
    catboost_params: dict[str, Any] | None = None,
    excluded_feature_prefixes: tuple[str, ...] = (),
) -> TrainedTableModels:
    """Fit table baselines using only the pre-registered training period."""
    data = to_polars(frame)
    if "split" not in data.columns or CANONICAL.is_laundering not in data.columns:
        raise ValueError("Frame requires split and canonical is_laundering columns.")
    train = data.filter(pl.col("split") == TimeSplit.TRAIN.value)
    validation = data.filter(pl.col("split") == TimeSplit.VALIDATION.value)
    if train.is_empty() or validation.is_empty():
        raise ValueError("Both train and validation chronological partitions are required.")
    return fit_table_models_for_partitions(
        train,
        validation,
        random_seed=random_seed,
        catboost_params=catboost_params,
        excluded_feature_prefixes=excluded_feature_prefixes,
    )


def fit_table_models_for_partitions(
    train: pl.DataFrame | pd.DataFrame,
    validation: pl.DataFrame | pd.DataFrame,
    *,
    random_seed: int = 20260722,
    catboost_params: dict[str, Any] | None = None,
    excluded_feature_prefixes: tuple[str, ...] = (),
) -> TrainedTableModels:
    """Fit table models from explicitly supplied chronological train/validation frames."""
    train_frame = to_polars(train)
    validation_frame = to_polars(validation)
    if train_frame.is_empty() or validation_frame.is_empty():
        raise ValueError("Both chronological train and validation frames are required.")
    if (
        CANONICAL.is_laundering not in train_frame.columns
        or CANONICAL.is_laundering not in validation_frame.columns
    ):
        raise ValueError("Both partitions require canonical is_laundering labels.")

    spec = infer_feature_spec(train_frame, excluded_prefixes=excluded_feature_prefixes)
    train_logistic, train_catboost = prepare_feature_frames(train_frame, spec)
    validation_logistic, validation_catboost = prepare_feature_frames(validation_frame, spec)
    labels = train_frame[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    validation_labels = validation_frame[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()

    logistic = _build_logistic_pipeline(spec, random_seed=random_seed)
    logistic.fit(train_logistic, labels)

    params: dict[str, Any] = {
        "iterations": 800,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
        "class_weights": _class_weights(labels),
    }
    if catboost_params:
        forbidden = {"class_weights", "auto_class_weights", "scale_pos_weight"}
        if forbidden.intersection(catboost_params):
            raise ValueError(
                "Class weighting is centrally controlled; "
                "do not override it in catboost_params."
            )
        params.update(catboost_params)
    params["random_seed"] = random_seed
    catboost = CatBoostClassifier(**params)
    catboost.fit(
        train_catboost,
        labels,
        cat_features=list(spec.categorical_columns),
        eval_set=(validation_catboost, validation_labels),
        early_stopping_rounds=80,
    )
    return TrainedTableModels(feature_spec=spec, logistic=logistic, catboost=catboost)
