"""Train and evaluate table baselines without using test labels during fitting."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pyarrow.dataset as ds

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
from aml_evidence_graph.explain.tabular import write_catboost_explanations
from aml_evidence_graph.models.tabular import TrainedTableModels, fit_table_models
from aml_evidence_graph.tracking.run import (
    configure_structured_logger,
    create_run_manifest,
)
from aml_evidence_graph.training.configuration import (
    DEFAULT_MODEL_CONFIG_PATH,
    catboost_parameters_from_configuration,
    load_model_configuration,
)


@dataclass(frozen=True)
class TableBaselineSummary:
    """Metadata and metrics that can be persisted without transaction records."""

    created_at_utc: str
    run_id: str
    input_root: str
    training_rows_before_sampling: int
    training_rows_after_sampling: int
    training_sampling_strategy: str
    validation_rows: int
    test_rows: int
    feature_columns: list[str]
    graph_feature_columns: list[str]
    validation_metrics: dict[str, dict[str, Any]]
    test_metrics: dict[str, dict[str, Any]]
    alert_reduction_vs_rules: dict[str, dict[str, Any]]
    test_monthly_stability: dict[str, dict[str, Any]]
    test_typology_slices: dict[str, dict[str, Any]]
    test_new_account_slices: dict[str, dict[str, Any]]
    test_payment_type_slices: dict[str, dict[str, Any]]
    test_location_pair_slices: dict[str, dict[str, Any]]
    test_currency_pair_slices: dict[str, dict[str, Any]]
    test_bootstrap_intervals: dict[str, dict[str, Any]] | None
    runtime: dict[str, float]
    explanation_artifacts: dict[str, dict[str, object]]
    feature_drift: dict[str, dict[str, dict[str, Any]]]


def load_feature_split(feature_root: Path, split: TimeSplit) -> pd.DataFrame:
    """Load exactly one persisted chronological split without random sampling."""
    if not feature_root.is_dir():
        raise FileNotFoundError(f"Feature dataset does not exist: {feature_root}")
    dataset = ds.dataset(feature_root, format="parquet", partitioning="hive")
    frame = dataset.to_table(
        filter=ds.field("split") == split.value,
    ).to_pandas()
    if frame.empty:
        raise ValueError(f"Feature dataset contains no rows for split={split.value}.")
    return frame


def deterministic_negative_downsample(
    training: pd.DataFrame,
    *,
    maximum_negative_rows: int | None,
) -> pd.DataFrame:
    """Optionally downsample only training negatives via stable row-number hashing."""
    if maximum_negative_rows is None:
        return training.copy()
    if maximum_negative_rows < 1:
        raise ValueError("maximum_negative_rows must be positive when provided.")
    labels = training[CANONICAL.is_laundering].astype(int)
    negatives = training.loc[labels.eq(0)]
    positives = training.loc[labels.eq(1)]
    if len(negatives) <= maximum_negative_rows:
        return training.copy()
    if CANONICAL.source_row_number not in training:
        raise ValueError("Training data requires source_row_number for deterministic sampling.")

    stable_hash = pd.util.hash_pandas_object(
        negatives[CANONICAL.source_row_number],
        index=False,
    )
    selected_negative_index = stable_hash.nsmallest(maximum_negative_rows).index
    sampled = pd.concat(
        [positives, negatives.loc[selected_negative_index]],
        axis=0,
        ignore_index=True,
    )
    return sampled.sort_values(
        [CANONICAL.event_ts, CANONICAL.source_row_number],
        kind="stable",
    ).reset_index(drop=True)


def deterministic_hard_negative_downsample(
    training: pd.DataFrame,
    hard_negative_oof: pd.DataFrame,
    *,
    maximum_negative_rows: int,
    score_column: str = "catboost",
) -> pd.DataFrame:
    """Keep highest temporal-OOF negative scores, filling any remainder deterministically.

    The supplied scores must be OOF predictions created inside the training
    period. The function never fits on labels, scores validation/test rows, or
    resamples a non-training partition.
    """
    if maximum_negative_rows < 1:
        raise ValueError("maximum_negative_rows must be positive.")
    required_training = {
        CANONICAL.transaction_id,
        CANONICAL.source_row_number,
        CANONICAL.is_laundering,
    }
    missing_training = sorted(required_training.difference(training.columns))
    if missing_training:
        raise ValueError("Training data requires: " + ", ".join(missing_training))
    required_oof = {CANONICAL.transaction_id, score_column}
    missing_oof = sorted(required_oof.difference(hard_negative_oof.columns))
    if missing_oof:
        raise ValueError("Hard-negative OOF data requires: " + ", ".join(missing_oof))
    if training[CANONICAL.transaction_id].duplicated().any():
        raise ValueError("Training transaction IDs must be unique.")
    if hard_negative_oof[CANONICAL.transaction_id].duplicated().any():
        raise ValueError("Hard-negative OOF transaction IDs must be unique.")
    raw_scores = pd.to_numeric(hard_negative_oof[score_column], errors="raise")
    if not raw_scores.between(0, 1).all():
        raise ValueError("Hard-negative OOF scores must be probabilities in [0, 1].")
    training_ids = set(training[CANONICAL.transaction_id].astype(str))
    unknown_ids = set(hard_negative_oof[CANONICAL.transaction_id].astype(str)).difference(
        training_ids
    )
    if unknown_ids:
        raise ValueError("Hard-negative OOF scores contain IDs outside the training period.")

    labels = training[CANONICAL.is_laundering].astype(int)
    positives = training.loc[labels.eq(1)]
    negatives = training.loc[labels.eq(0)].copy()
    if len(negatives) <= maximum_negative_rows:
        return training.copy()
    oof_scores = hard_negative_oof.loc[:, [CANONICAL.transaction_id, score_column]].copy()
    oof_scores[CANONICAL.transaction_id] = oof_scores[CANONICAL.transaction_id].astype(str)
    negatives[CANONICAL.transaction_id] = negatives[CANONICAL.transaction_id].astype(str)
    ranked = negatives.merge(
        oof_scores,
        on=CANONICAL.transaction_id,
        how="left",
        validate="one_to_one",
    )
    scored = ranked.loc[ranked[score_column].notna()].sort_values(
        [score_column, CANONICAL.source_row_number],
        ascending=[False, True],
        kind="stable",
    )
    selected = scored.head(maximum_negative_rows)
    remaining_count = maximum_negative_rows - len(selected)
    if remaining_count:
        selected_ids = set(selected[CANONICAL.transaction_id])
        remaining = ranked.loc[~ranked[CANONICAL.transaction_id].isin(selected_ids)].copy()
        remaining["_stable_hash"] = pd.util.hash_pandas_object(
            remaining[CANONICAL.source_row_number],
            index=False,
        )
        selected = pd.concat(
            [
                selected,
                remaining.nsmallest(remaining_count, "_stable_hash"),
            ],
            ignore_index=True,
        )
    sampled = pd.concat(
        [positives, selected.loc[:, training.columns]],
        axis=0,
        ignore_index=True,
    )
    return sampled.sort_values(
        [CANONICAL.event_ts, CANONICAL.source_row_number],
        kind="stable",
    ).reset_index(drop=True)


def rule_baseline_scores(frame: pd.DataFrame) -> pd.Series | None:
    """Return the approved-rule baseline score, or None when no rule features exist."""
    rule_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("rule_") and column.endswith("_hit")
    )
    if not rule_columns:
        return None
    score = frame.loc[:, rule_columns].apply(pd.to_numeric, errors="raise").sum(axis=1)
    return score.clip(lower=0, upper=1).astype(float)


def _write_model_artifacts(
    models: TrainedTableModels,
    output_dir: Path,
    *,
    model_name: str,
) -> None:
    model_dir = output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(models.logistic, model_dir / "logistic.joblib")
    models.catboost.save_model(model_dir / "catboost.cbm")
    (model_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "numeric_columns": list(models.feature_spec.numeric_columns),
                "categorical_columns": list(models.feature_spec.categorical_columns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_component_scores(
    output_dir: Path,
    *,
    split_name: str,
    frame: pd.DataFrame,
    component_scores: dict[str, Any],
) -> Path:
    """Persist private component score rows for later OOF/validation-only fusion."""
    required = [
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.is_laundering,
    ]
    output = frame.loc[:, required].copy()
    for name, scores in component_scores.items():
        if len(scores) != len(output):
            raise ValueError(f"Score length mismatch for component {name}.")
        output[name] = scores
    score_dir = output_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    path = score_dir / f"table_{split_name}_scores.parquet"
    output.to_parquet(path, index=False)
    return path


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Use overwrite=True only for private, regenerable training artifacts."
        )
    if not output_dir.is_dir():
        raise ValueError(f"Training output path must be a directory: {output_dir}")
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def train_and_evaluate_table_baselines(
    feature_root: Path,
    output_dir: Path,
    *,
    maximum_training_negative_rows: int | None = 500_000,
    random_seed: int = 20260722,
    catboost_params: dict[str, Any] | None = None,
    hard_negative_oof: pd.DataFrame | None = None,
    hard_negative_oof_path: Path | None = None,
    hard_negative_score_column: str = "catboost",
    model_config_path: Path | None = None,
    bootstrap_iterations: int = 0,
    overwrite: bool = False,
) -> TableBaselineSummary:
    """Fit on train, select CatBoost stopping point on validation, then test once."""
    started_at = time.perf_counter()
    _prepare_output_dir(output_dir, overwrite=overwrite)
    full_training = load_feature_split(feature_root, TimeSplit.TRAIN)
    if hard_negative_oof is not None:
        if maximum_training_negative_rows is None:
            raise ValueError("Hard-negative sampling requires a negative-row budget.")
        sampled_training = deterministic_hard_negative_downsample(
            full_training,
            hard_negative_oof,
            maximum_negative_rows=maximum_training_negative_rows,
            score_column=hard_negative_score_column,
        )
        sampling_strategy = "temporal_oof_hard_negative"
        if hard_negative_oof_path is None:
            raise ValueError("Hard-negative sampling requires its source artifact path.")
    else:
        sampled_training = deterministic_negative_downsample(
            full_training,
            maximum_negative_rows=maximum_training_negative_rows,
        )
        sampling_strategy = "stable_hash_negative_downsample"
    validation = load_feature_split(feature_root, TimeSplit.VALIDATION)
    fitting_frame = pd.concat([sampled_training, validation], ignore_index=True)
    models, table_fit_resources = measure_runtime(
        lambda: fit_table_models(
            fitting_frame,
            random_seed=random_seed,
            catboost_params=catboost_params,
            excluded_feature_prefixes=("graph_",),
        )
    )
    graph_feature_columns = sorted(
        column for column in fitting_frame.columns if column.startswith("graph_")
    )
    graph_fit_resources: dict[str, float] | None = None
    if graph_feature_columns:
        graph_stat_models, graph_fit_resources = measure_runtime(
            lambda: fit_table_models(
                fitting_frame,
                random_seed=random_seed,
                catboost_params=catboost_params,
            )
        )
    else:
        graph_stat_models = None
    validation_scores, validation_inference_resources = measure_runtime(
        lambda: models.predict_proba(validation)
    )
    validation_labels = validation[CANONICAL.is_laundering].astype(int)
    validation_metrics = {
        name: evaluate_binary_risk_scores(validation_labels, scores)
        for name, scores in validation_scores.items()
    }
    if graph_stat_models is not None:
        graph_stat_scores, validation_graph_inference_resources = measure_runtime(
            lambda: graph_stat_models.predict_proba(validation)["catboost"]
        )
        validation_scores["graph_stats_catboost"] = graph_stat_scores
        validation_metrics["graph_stats_catboost"] = evaluate_binary_risk_scores(
            validation_labels,
            graph_stat_scores,
        )
    validation_rule_scores = rule_baseline_scores(validation)
    if validation_rule_scores is not None:
        validation_metrics["rules"] = evaluate_binary_risk_scores(
            validation_labels,
            validation_rule_scores,
        )
        validation_scores["rules"] = validation_rule_scores.to_numpy()

    test = load_feature_split(feature_root, TimeSplit.TEST)
    test_scores, test_inference_resources = measure_runtime(lambda: models.predict_proba(test))
    test_labels = test[CANONICAL.is_laundering].astype(int)
    test_metrics = {
        name: evaluate_binary_risk_scores(test_labels, scores)
        for name, scores in test_scores.items()
    }
    if graph_stat_models is not None:
        graph_stat_scores, test_graph_inference_resources = measure_runtime(
            lambda: graph_stat_models.predict_proba(test)["catboost"]
        )
        test_scores["graph_stats_catboost"] = graph_stat_scores
        test_metrics["graph_stats_catboost"] = evaluate_binary_risk_scores(
            test_labels,
            graph_stat_scores,
        )
    test_rule_scores = rule_baseline_scores(test)
    if test_rule_scores is not None:
        test_metrics["rules"] = evaluate_binary_risk_scores(test_labels, test_rule_scores)
        test_scores["rules"] = test_rule_scores.to_numpy()

    alert_reduction_vs_rules: dict[str, dict[str, Any]] = {}
    if test_rule_scores is not None:
        alert_reduction_vs_rules["catboost"] = compare_alert_volume_at_fixed_recall(
            test_labels,
            test_scores["catboost"],
            test_rule_scores,
        )
        if "graph_stats_catboost" in test_scores:
            alert_reduction_vs_rules["graph_stats_catboost"] = compare_alert_volume_at_fixed_recall(
                test_labels,
                test_scores["graph_stats_catboost"],
                test_rule_scores,
            )
        if "logistic" in test_scores:
            alert_reduction_vs_rules["logistic"] = compare_alert_volume_at_fixed_recall(
                test_labels,
                test_scores["logistic"],
                test_rule_scores,
            )

    selected_test_scores = test_scores["catboost"]
    training_accounts = set(
        pd.concat(
            [
                full_training[CANONICAL.sender_account_id],
                full_training[CANONICAL.receiver_account_id],
            ],
            ignore_index=True,
        )
        .astype(str)
        .tolist()
    )
    test_monthly_stability = monthly_stability_report(test, selected_test_scores)
    test_typology_slices = typology_slice_report(test, selected_test_scores)
    test_new_account_slices = new_account_slice_report(
        test,
        selected_test_scores,
        training_accounts=training_accounts,
    )
    test_payment_type_slices = categorical_slice_report(
        test,
        selected_test_scores,
        column=CANONICAL.payment_type,
    )
    test_location_pair_slices = paired_categorical_slice_report(
        test,
        selected_test_scores,
        left_column=CANONICAL.sender_location,
        right_column=CANONICAL.receiver_location,
    )
    test_currency_pair_slices = paired_categorical_slice_report(
        test,
        selected_test_scores,
        left_column=CANONICAL.payment_currency,
        right_column=CANONICAL.received_currency,
    )
    test_bootstrap_intervals = (
        bootstrap_ranking_intervals(
            test_labels,
            selected_test_scores,
            iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        if bootstrap_iterations
        else None
    )
    feature_columns_for_drift = list(models.feature_spec.all_columns)
    feature_drift = {
        "validation_vs_train": feature_drift_report(
            full_training,
            validation,
            feature_columns=feature_columns_for_drift,
        ),
        "test_vs_train": feature_drift_report(
            full_training,
            test,
            feature_columns=feature_columns_for_drift,
        ),
    }
    _write_component_scores(
        output_dir,
        split_name="validation",
        frame=validation,
        component_scores=validation_scores,
    )
    _write_component_scores(
        output_dir,
        split_name="test",
        frame=test,
        component_scores=test_scores,
    )
    (output_dir / "models").mkdir(parents=True, exist_ok=False)
    _write_model_artifacts(models, output_dir, model_name="table_baselines")
    if graph_stat_models is not None:
        _write_model_artifacts(
            graph_stat_models,
            output_dir,
            model_name="graph_stats_catboost",
        )
    explanation_dir = output_dir / "explanations"
    explanation_artifacts = {
        "table_baselines": write_catboost_explanations(
            models,
            validation,
            explanation_dir,
            model_name="table_baselines",
        )
    }
    if graph_stat_models is not None:
        explanation_artifacts["graph_stats_catboost"] = write_catboost_explanations(
            graph_stat_models,
            validation,
            explanation_dir,
            model_name="graph_stats_catboost",
        )
    manifest = create_run_manifest(
        output_dir=output_dir,
        command="aml-train-table",
        random_seed=random_seed,
        input_paths={
            "pit_feature_dataset": feature_root,
            **(
                {"hard_negative_oof": hard_negative_oof_path}
                if hard_negative_oof_path is not None
                else {}
            ),
        },
        config_paths={
            "model_config": model_config_path or DEFAULT_MODEL_CONFIG_PATH,
        },
        metadata={
            "maximum_training_negative_rows": maximum_training_negative_rows,
            "training_sampling_strategy": sampling_strategy,
            "hard_negative_score_column": (
                hard_negative_score_column if hard_negative_oof is not None else None
            ),
            "bootstrap_iterations": bootstrap_iterations,
            "model_names": sorted(test_metrics),
        },
    )
    summary = TableBaselineSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        run_id=manifest.run_id,
        input_root=str(feature_root),
        training_rows_before_sampling=len(full_training),
        training_rows_after_sampling=len(sampled_training),
        training_sampling_strategy=sampling_strategy,
        validation_rows=len(validation),
        test_rows=len(test),
        feature_columns=list(models.feature_spec.all_columns),
        graph_feature_columns=graph_feature_columns,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        alert_reduction_vs_rules=alert_reduction_vs_rules,
        test_monthly_stability=test_monthly_stability,
        test_typology_slices=test_typology_slices,
        test_new_account_slices=test_new_account_slices,
        test_payment_type_slices=test_payment_type_slices,
        test_location_pair_slices=test_location_pair_slices,
        test_currency_pair_slices=test_currency_pair_slices,
        test_bootstrap_intervals=test_bootstrap_intervals,
        runtime={
            "wall_time_seconds": time.perf_counter() - started_at,
            **{f"table_fit_{name}": value for name, value in table_fit_resources.items()},
            **{
                f"validation_inference_{name}": value
                for name, value in validation_inference_resources.items()
            },
            **{f"test_inference_{name}": value for name, value in test_inference_resources.items()},
            **(
                {
                    f"graph_stat_fit_{name}": value
                    for name, value in graph_fit_resources.items()
                }
                if graph_fit_resources is not None
                else {}
            ),
            **(
                {
                    f"validation_graph_stat_inference_{name}": value
                    for name, value in validation_graph_inference_resources.items()
                }
                if graph_stat_models is not None
                else {}
            ),
            **(
                {
                    f"test_graph_stat_inference_{name}": value
                    for name, value in test_graph_inference_resources.items()
                }
                if graph_stat_models is not None
                else {}
            ),
            "validation_rows_per_second": len(validation)
            / max(validation_inference_resources["wall_time_ms"] / 1_000, 1e-9),
            "test_rows_per_second": len(test)
            / max(test_inference_resources["wall_time_ms"] / 1_000, 1e-9),
        },
        explanation_artifacts=explanation_artifacts,
        feature_drift=feature_drift,
    )
    configure_structured_logger("aml.table_training", run_id=manifest.run_id).info(
        "table_training_completed",
        extra={
            "model_version": manifest.run_id,
            "duration_ms": summary.runtime["wall_time_seconds"] * 1_000,
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
    parser.add_argument(
        "--hard-negative-oof",
        type=Path,
        help="Training-period OOF Parquet/CSV used only to prioritize negative samples.",
    )
    parser.add_argument("--hard-negative-score-column", default="catboost")
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_configuration = load_model_configuration(args.model_config)
    hard_negative_oof = None
    if args.hard_negative_oof is not None:
        suffix = args.hard_negative_oof.suffix.lower()
        if suffix == ".parquet":
            hard_negative_oof = pd.read_parquet(args.hard_negative_oof)
        elif suffix == ".csv":
            hard_negative_oof = pd.read_csv(args.hard_negative_oof)
        else:
            raise ValueError("Hard-negative OOF input must be Parquet or CSV.")
    summary = train_and_evaluate_table_baselines(
        args.features,
        args.output,
        maximum_training_negative_rows=args.max_training_negative_rows,
        random_seed=args.random_seed,
        catboost_params=catboost_parameters_from_configuration(model_configuration),
        hard_negative_oof=hard_negative_oof,
        hard_negative_oof_path=args.hard_negative_oof,
        hard_negative_score_column=args.hard_negative_score_column,
        model_config_path=args.model_config,
        bootstrap_iterations=args.bootstrap_iterations,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "training_rows_after_sampling": summary.training_rows_after_sampling,
                "validation_rows": summary.validation_rows,
                "test_rows": summary.test_rows,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
