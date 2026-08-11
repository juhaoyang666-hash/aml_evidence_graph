"""Load persisted private table-model artifacts for controlled batch scoring."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from aml_evidence_graph.models.boosting import CategoryVocabulary, TrainedLightGBMModel
from aml_evidence_graph.models.tabular import FeatureSpec, TrainedTableModels


def _load_lightgbm_artifacts(model_dir: Path) -> TrainedLightGBMModel:
    required = {"lightgbm.joblib", "feature_spec.json", "category_vocabulary.json"}
    present = {path.name for path in model_dir.iterdir()} if model_dir.is_dir() else set()
    missing = sorted(required.difference(present))
    if missing:
        raise FileNotFoundError(
            f"LightGBM artifact directory is incomplete: {', '.join(missing)}"
        )
    spec_document = json.loads(
        (model_dir / "feature_spec.json").read_text(encoding="utf-8")
    )
    if spec_document.get("model_family") != "lightgbm":
        raise ValueError("LightGBM feature contract has an unexpected model_family.")
    vocabulary_document = json.loads(
        (model_dir / "category_vocabulary.json").read_text(encoding="utf-8")
    )
    model = joblib.load(model_dir / "lightgbm.joblib")
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:  # pragma: no cover - primary dependency in release env
        raise RuntimeError("LightGBM is required to load the primary table model.") from error
    if not isinstance(model, LGBMClassifier):
        raise TypeError("Trusted LightGBM artifact must contain an LGBMClassifier.")
    feature_spec = FeatureSpec(
        numeric_columns=tuple(spec_document["numeric_columns"]),
        categorical_columns=tuple(spec_document["categorical_columns"]),
    )
    vocabulary = CategoryVocabulary(
        levels={
            str(name): tuple(str(value) for value in levels)
            for name, levels in vocabulary_document.items()
        }
    )
    if set(vocabulary.levels) != set(feature_spec.categorical_columns):
        raise ValueError("LightGBM category vocabulary does not match the feature contract.")
    return TrainedLightGBMModel(
        feature_spec=feature_spec,
        category_vocabulary=vocabulary,
        model=model,
    )


def _load_legacy_catboost_artifacts(model_dir: Path) -> TrainedTableModels:
    required = {
        "logistic.joblib",
        "catboost.cbm",
        "feature_spec.json",
    }
    present = {path.name for path in model_dir.iterdir()} if model_dir.is_dir() else set()
    missing = sorted(required.difference(present))
    if missing:
        raise FileNotFoundError(
            f"Table model artifact directory is incomplete: {', '.join(missing)}"
        )
    spec_document = json.loads(
        (model_dir / "feature_spec.json").read_text(encoding="utf-8")
    )
    feature_spec = FeatureSpec(
        numeric_columns=tuple(spec_document["numeric_columns"]),
        categorical_columns=tuple(spec_document["categorical_columns"]),
    )
    logistic = joblib.load(model_dir / "logistic.joblib")
    if not isinstance(logistic, Pipeline):
        raise TypeError("Trusted logistic artifact must contain a sklearn Pipeline.")
    catboost = CatBoostClassifier()
    catboost.load_model(model_dir / "catboost.cbm")
    return TrainedTableModels(
        feature_spec=feature_spec,
        logistic=logistic,
        catboost=catboost,
    )


def load_table_model_artifacts(
    model_dir: Path,
) -> TrainedLightGBMModel | TrainedTableModels:
    """Load the primary LightGBM bundle, with explicit legacy CatBoost compatibility."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Table model artifact directory does not exist: {model_dir}")
    if (model_dir / "lightgbm.joblib").is_file():
        return _load_lightgbm_artifacts(model_dir)
    return _load_legacy_catboost_artifacts(model_dir)
