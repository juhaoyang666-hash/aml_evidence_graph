"""Load persisted private table-model artifacts for controlled batch scoring."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from aml_evidence_graph.models.tabular import FeatureSpec, TrainedTableModels


def load_table_model_artifacts(model_dir: Path) -> TrainedTableModels:
    """Load a trusted local artifact bundle; never deserialize untrusted uploads."""
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

