"""Chronological Logistic Regression and CatBoost baselines for transaction risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    def predict_proba(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return positive-class probabilities for both deterministic baselines."""
        logistic_frame, catboost_frame = prepare_feature_frames(frame, self.feature_spec)
        return {
            "logistic": self.logistic.predict_proba(logistic_frame)[:, 1],
            "catboost": self.catboost.predict_proba(catboost_frame)[:, 1],
        }


def infer_feature_spec(
    frame: pd.DataFrame,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> FeatureSpec:
    """Select model features while explicitly excluding labels and identifiers."""
    candidates = [
        column
        for column in frame.columns
        if column not in LEAKAGE_COLUMNS
        and not any(column.startswith(prefix) for prefix in excluded_prefixes)
    ]
    if not candidates:
        raise ValueError("No model features remain after excluding labels and identifiers.")
    numeric = tuple(
        column for column in candidates if pd.api.types.is_numeric_dtype(frame[column])
    )
    categorical = tuple(column for column in candidates if column not in numeric)
    return FeatureSpec(numeric_columns=numeric, categorical_columns=categorical)


def prepare_feature_frames(
    frame: pd.DataFrame,
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare matching feature frames without fitting transformations on validation/test."""
    missing = sorted(set(spec.all_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Model input is missing columns: {', '.join(missing)}")
    feature_columns = list(spec.all_columns)
    logistic_frame = frame.loc[:, feature_columns].copy()
    catboost_frame = frame.loc[:, feature_columns].copy()
    for column in spec.categorical_columns:
        logistic_frame[column] = logistic_frame[column].astype("string")
        catboost_frame[column] = catboost_frame[column].astype("string").fillna("__MISSING__")
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


def _class_weights(labels: pd.Series) -> list[float]:
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training split must contain both positive and negative examples.")
    return [1.0, negative_count / positive_count]


def fit_table_models(
    frame: pd.DataFrame,
    *,
    random_seed: int = 20260722,
    catboost_params: dict[str, Any] | None = None,
    excluded_feature_prefixes: tuple[str, ...] = (),
) -> TrainedTableModels:
    """Fit table baselines using only the pre-registered training period."""
    if "split" not in frame or CANONICAL.is_laundering not in frame:
        raise ValueError("Frame requires split and canonical is_laundering columns.")
    train = frame.loc[frame["split"].eq(TimeSplit.TRAIN.value)].copy()
    validation = frame.loc[frame["split"].eq(TimeSplit.VALIDATION.value)].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation chronological partitions are required.")
    return fit_table_models_for_partitions(
        train,
        validation,
        random_seed=random_seed,
        catboost_params=catboost_params,
        excluded_feature_prefixes=excluded_feature_prefixes,
    )


def fit_table_models_for_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    random_seed: int = 20260722,
    catboost_params: dict[str, Any] | None = None,
    excluded_feature_prefixes: tuple[str, ...] = (),
) -> TrainedTableModels:
    """Fit table models from explicitly supplied chronological train/validation frames."""
    if train.empty or validation.empty:
        raise ValueError("Both chronological train and validation frames are required.")
    if CANONICAL.is_laundering not in train or CANONICAL.is_laundering not in validation:
        raise ValueError("Both partitions require canonical is_laundering labels.")

    spec = infer_feature_spec(train, excluded_prefixes=excluded_feature_prefixes)
    train_logistic, train_catboost = prepare_feature_frames(train, spec)
    validation_logistic, validation_catboost = prepare_feature_frames(validation, spec)
    labels = train[CANONICAL.is_laundering].astype(int)
    validation_labels = validation[CANONICAL.is_laundering].astype(int)

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
