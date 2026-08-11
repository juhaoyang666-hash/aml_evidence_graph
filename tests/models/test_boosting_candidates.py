from pathlib import Path

import pandas as pd
import polars as pl

from aml_evidence_graph.explain.tabular import write_lightgbm_explanations
from aml_evidence_graph.models.boosting import (
    fit_category_vocabulary,
    fit_lightgbm_for_partitions,
    prepare_native_boosting_frame,
    save_lightgbm_artifacts,
)
from aml_evidence_graph.models.loading import load_table_model_artifacts
from aml_evidence_graph.models.tabular import FeatureSpec


def test_native_boosting_vocabulary_is_fit_on_training_only() -> None:
    spec = FeatureSpec(
        numeric_columns=("amount",),
        categorical_columns=("payment_type",),
    )
    training = pl.DataFrame(
        {"amount": [1, 2], "payment_type": ["wire", "cash"]}
    )
    validation = pl.DataFrame(
        {"amount": [3], "payment_type": ["unseen"]}
    )

    vocabulary = fit_category_vocabulary(training, spec)
    prepared = prepare_native_boosting_frame(validation, spec, vocabulary)

    assert vocabulary.levels == {"payment_type": ("cash", "wire")}
    assert str(prepared["amount"].dtype) == "float32"
    assert isinstance(prepared["payment_type"].dtype, pd.CategoricalDtype)
    assert prepared["payment_type"].isna().all()


def test_promoted_lightgbm_bundle_round_trips(tmp_path: Path) -> None:
    train = pl.DataFrame(
        {
            "transaction_id": [f"train-{index}" for index in range(8)],
            "event_ts": [f"2022-10-{index + 1:02d}T00:00:00Z" for index in range(8)],
            "source_row_number": range(8),
            "is_laundering": [0, 1] * 4,
            "amount": [float(index) for index in range(8)],
            "payment_type": ["wire", "cash"] * 4,
        }
    )
    validation = train.with_columns(
        pl.col("transaction_id").str.replace("train", "validation")
    )
    trained = fit_lightgbm_for_partitions(
        train,
        validation,
        parameters={"n_estimators": 10, "num_leaves": 7, "min_child_samples": 1},
    )
    model_dir = tmp_path / "table_primary"
    save_lightgbm_artifacts(trained, model_dir)

    loaded = load_table_model_artifacts(model_dir)
    scores = loaded.predict_proba(validation)["lightgbm"]

    assert len(scores) == validation.height
    assert ((scores >= 0) & (scores <= 1)).all()

    explanation = write_lightgbm_explanations(
        loaded,
        validation,
        tmp_path / "explanations",
        max_rows=4,
    )
    assert explanation["sample_count"] == 4
    assert (tmp_path / "explanations" / "lightgbm_local_shap.parquet").is_file()
    assert (tmp_path / "explanations" / "lightgbm_global_importance.json").is_file()
