"""Promoted chronological LightGBM FE v2 table baseline and frozen evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.drift import feature_drift_report
from aml_evidence_graph.evaluation.metrics import (
    compare_alert_volume_at_fixed_recall,
    evaluate_binary_risk_scores,
)
from aml_evidence_graph.evaluation.monitoring import (
    bootstrap_ranking_intervals,
    categorical_slice_report,
    measure_runtime,
    monthly_stability_report,
    new_account_slice_report,
    paired_categorical_slice_report,
    typology_slice_report,
)
from aml_evidence_graph.explain.tabular import write_lightgbm_explanations
from aml_evidence_graph.models.boosting import (
    TrainedLightGBMModel,
    fit_lightgbm_for_partitions,
    save_lightgbm_artifacts,
)
from aml_evidence_graph.tracking.run import create_run_manifest
from aml_evidence_graph.training.configuration import (
    DEFAULT_MODEL_CONFIG_PATH,
    lightgbm_parameters_from_configuration,
    load_model_configuration,
)
from aml_evidence_graph.training.table_baseline import (
    deterministic_negative_downsample,
    load_feature_split,
    rule_baseline_scores,
)


@dataclass(frozen=True)
class LightGBMBaselineSummary:
    created_at_utc: str
    run_id: str
    input_root: str
    training_rows_before_sampling: int
    training_rows_after_sampling: int
    validation_rows: int
    test_rows: int
    feature_columns: list[str]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    alert_reduction_vs_rules: dict[str, Any]
    test_monthly_stability: dict[str, dict[str, Any]]
    test_typology_slices: dict[str, dict[str, Any]]
    test_new_account_slices: dict[str, dict[str, Any]]
    test_payment_type_slices: dict[str, dict[str, Any]]
    test_location_pair_slices: dict[str, dict[str, Any]]
    test_currency_pair_slices: dict[str, dict[str, Any]]
    test_bootstrap_intervals: dict[str, dict[str, Any]] | None
    feature_drift: dict[str, dict[str, dict[str, Any]]]
    explanation_artifacts: dict[str, object]
    runtime: dict[str, float]


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        if not output_dir.is_dir():
            raise ValueError(f"Output path must be a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _write_scores(
    output_dir: Path,
    *,
    split_name: str,
    frame: pl.DataFrame,
    scores: np.ndarray,
    rule_scores: np.ndarray | None,
) -> Path:
    output = frame.select(
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.is_laundering,
    ).with_columns(pl.Series("lightgbm", scores))
    if rule_scores is not None:
        output = output.with_columns(pl.Series("rules", rule_scores))
    score_dir = output_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    path = score_dir / f"table_{split_name}_scores.parquet"
    output.write_parquet(path)
    return path


def train_and_evaluate_lightgbm(
    feature_root: Path,
    output_dir: Path,
    *,
    maximum_training_negative_rows: int | None = 500_000,
    random_seed: int = 20260722,
    parameters: dict[str, Any] | None = None,
    model_config_path: Path | None = None,
    bootstrap_iterations: int = 0,
    overwrite: bool = False,
) -> LightGBMBaselineSummary:
    """Fit on train, stop on validation PR-AUC, and evaluate frozen test once."""
    started = time.perf_counter()
    _prepare_output_dir(output_dir, overwrite=overwrite)
    full_training = load_feature_split(feature_root, TimeSplit.TRAIN)
    sampled_training = deterministic_negative_downsample(
        full_training,
        maximum_negative_rows=maximum_training_negative_rows,
    )
    validation = load_feature_split(feature_root, TimeSplit.VALIDATION)
    models, fit_resources = measure_runtime(
        lambda: fit_lightgbm_for_partitions(
            sampled_training,
            validation,
            random_seed=random_seed,
            parameters=parameters,
        )
    )
    assert isinstance(models, TrainedLightGBMModel)
    validation_scores, validation_resources = measure_runtime(
        lambda: models.predict_proba(validation)["lightgbm"]
    )
    validation_labels = validation[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    validation_metrics = evaluate_binary_risk_scores(
        validation_labels,
        validation_scores,
    )
    validation_rule_scores = rule_baseline_scores(validation)
    _write_scores(
        output_dir,
        split_name="validation",
        frame=validation,
        scores=validation_scores,
        rule_scores=validation_rule_scores,
    )

    # No test data is loaded until the model and stopping point are frozen.
    test = load_feature_split(feature_root, TimeSplit.TEST)
    test_scores, test_resources = measure_runtime(
        lambda: models.predict_proba(test)["lightgbm"]
    )
    test_labels = test[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    test_metrics = evaluate_binary_risk_scores(test_labels, test_scores)
    test_rule_scores = rule_baseline_scores(test)
    alert_reduction = (
        compare_alert_volume_at_fixed_recall(test_labels, test_scores, test_rule_scores)
        if test_rule_scores is not None
        else {}
    )
    _write_scores(
        output_dir,
        split_name="test",
        frame=test,
        scores=test_scores,
        rule_scores=test_rule_scores,
    )
    save_lightgbm_artifacts(models, output_dir / "models" / "table_primary")
    explanation_artifacts = write_lightgbm_explanations(
        models,
        test,
        output_dir / "explanations",
    )

    training_accounts = set(
        pl.concat(
            [
                full_training.select(pl.col(CANONICAL.sender_account_id).cast(pl.Utf8)),
                full_training.select(
                    pl.col(CANONICAL.receiver_account_id)
                    .cast(pl.Utf8)
                    .alias(CANONICAL.sender_account_id)
                ),
            ]
        )
        .to_series()
        .to_list()
    )
    feature_columns = list(models.feature_spec.all_columns)
    bootstrap = (
        bootstrap_ranking_intervals(
            test_labels,
            test_scores,
            iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        if bootstrap_iterations
        else None
    )
    drift = {
        "validation_vs_train": feature_drift_report(
            full_training,
            validation,
            feature_columns=feature_columns,
        ),
        "test_vs_train": feature_drift_report(
            full_training,
            test,
            feature_columns=feature_columns,
        ),
    }
    manifest = create_run_manifest(
        output_dir=output_dir,
        command="aml-train-table",
        random_seed=random_seed,
        input_paths={"pit_feature_dataset": feature_root},
        config_paths={"model_config": model_config_path or DEFAULT_MODEL_CONFIG_PATH},
        metadata={
            "model_family": "lightgbm",
            "training_sampling": "stable_hash_negative_downsample",
            "maximum_training_negative_rows": maximum_training_negative_rows,
            "selection_split": "validation",
            "test_access": "once_after_model_freeze",
        },
    )
    summary = LightGBMBaselineSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        run_id=manifest.run_id,
        input_root=str(feature_root),
        training_rows_before_sampling=full_training.height,
        training_rows_after_sampling=sampled_training.height,
        validation_rows=validation.height,
        test_rows=test.height,
        feature_columns=feature_columns,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        alert_reduction_vs_rules=alert_reduction,
        test_monthly_stability=monthly_stability_report(test, test_scores),
        test_typology_slices=typology_slice_report(test, test_scores),
        test_new_account_slices=new_account_slice_report(
            test,
            test_scores,
            training_accounts=training_accounts,
        ),
        test_payment_type_slices=categorical_slice_report(
            test,
            test_scores,
            column=CANONICAL.payment_type,
        ),
        test_location_pair_slices=paired_categorical_slice_report(
            test,
            test_scores,
            left_column=CANONICAL.sender_location,
            right_column=CANONICAL.receiver_location,
        ),
        test_currency_pair_slices=paired_categorical_slice_report(
            test,
            test_scores,
            left_column=CANONICAL.payment_currency,
            right_column=CANONICAL.received_currency,
        ),
        test_bootstrap_intervals=bootstrap,
        feature_drift=drift,
        explanation_artifacts=explanation_artifacts,
        runtime={
            "wall_time_seconds": time.perf_counter() - started,
            "fit_wall_time_seconds": fit_resources["wall_time_ms"] / 1_000,
            "validation_inference_wall_time_seconds": validation_resources[
                "wall_time_ms"
            ]
            / 1_000,
            "test_inference_wall_time_seconds": test_resources["wall_time_ms"] / 1_000,
        },
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-training-negative-rows", type=int, default=500_000)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = load_model_configuration(args.model_config)
    summary = train_and_evaluate_lightgbm(
        args.features,
        args.output,
        maximum_training_negative_rows=args.max_training_negative_rows,
        random_seed=args.random_seed,
        parameters=lightgbm_parameters_from_configuration(configuration),
        model_config_path=args.model_config,
        bootstrap_iterations=args.bootstrap_iterations,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "validation_pr_auc": summary.validation_metrics["pr_auc"],
                "test_pr_auc": summary.test_metrics["pr_auc"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
