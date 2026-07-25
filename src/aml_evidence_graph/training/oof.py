"""Expanding-time OOF predictions for leakage-safe AML probability fusion."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.graph.snapshots import (
    DailyGraphSnapshotBuilder,
    TemporalNodeIndexer,
    fit_edge_feature_scaler,
    transform_edge_features,
)
from aml_evidence_graph.models.tabular import fit_table_models_for_partitions
from aml_evidence_graph.tracking.run import create_run_manifest
from aml_evidence_graph.training.configuration import (
    DEFAULT_MODEL_CONFIG_PATH,
    catboost_parameters_from_configuration,
    graphsage_parameters_from_configuration,
    load_model_configuration,
)
from aml_evidence_graph.training.graphsage import (
    GraphSAGETrainingConfig,
    fit_graphsage,
    predict_graphsage,
)
from aml_evidence_graph.training.run_graphsage import select_graph_edge_features
from aml_evidence_graph.training.table_baseline import load_feature_split


@dataclass(frozen=True)
class ExpandingTimeFold:
    """One inner fold whose validation period starts strictly after its fit period."""

    fold_id: int
    training_months: tuple[str, ...]
    validation_months: tuple[str, ...]


def make_expanding_time_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    minimum_training_months: int = 2,
) -> list[ExpandingTimeFold]:
    """Split a training-only frame into expanding monthly OOF folds."""
    if CANONICAL.event_ts not in frame:
        raise ValueError("OOF folds require canonical event_ts.")
    if n_splits < 1 or minimum_training_months < 1:
        raise ValueError("n_splits and minimum_training_months must be positive.")
    months = pd.to_datetime(frame[CANONICAL.event_ts], utc=True, errors="raise").dt.strftime(
        "%Y-%m"
    )
    unique_months = tuple(sorted(months.unique()))
    remaining_months = unique_months[minimum_training_months:]
    if len(remaining_months) < n_splits:
        raise ValueError(
            "Not enough monthly periods for the requested expanding OOF folds."
        )
    groups = [tuple(group.tolist()) for group in np.array_split(remaining_months, n_splits)]
    folds: list[ExpandingTimeFold] = []
    for fold_offset, validation_months in enumerate(groups, start=1):
        first_validation_month = validation_months[0]
        training_months = tuple(month for month in unique_months if month < first_validation_month)
        if not training_months:
            raise AssertionError("Expanding OOF fold has no earlier training period.")
        folds.append(
            ExpandingTimeFold(
                fold_id=fold_offset,
                training_months=training_months,
                validation_months=validation_months,
            )
        )
    return folds


def generate_table_oof_predictions(
    training_frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    minimum_training_months: int = 2,
    random_seed: int = 20260722,
    catboost_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate CatBoost-family OOF scores from strictly earlier training months."""
    required = {
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.is_laundering,
    }
    missing = sorted(required.difference(training_frame.columns))
    if missing:
        raise ValueError(f"OOF input requires: {', '.join(missing)}")
    timestamps = pd.to_datetime(training_frame[CANONICAL.event_ts], utc=True, errors="raise")
    month_values = timestamps.dt.strftime("%Y-%m")
    folds = make_expanding_time_folds(
        training_frame,
        n_splits=n_splits,
        minimum_training_months=minimum_training_months,
    )
    graph_columns = tuple(
        column for column in training_frame.columns if column.startswith("graph_")
    )
    results: list[pd.DataFrame] = []
    for fold in folds:
        fit_frame = training_frame.loc[month_values.isin(fold.training_months)].copy()
        validation_frame = training_frame.loc[
            month_values.isin(fold.validation_months)
        ].copy()
        if fit_frame[CANONICAL.is_laundering].nunique() < 2:
            raise ValueError(f"OOF fold {fold.fold_id} training period has one label class.")
        if validation_frame[CANONICAL.is_laundering].nunique() < 2:
            raise ValueError(f"OOF fold {fold.fold_id} validation period has one label class.")
        table_models = fit_table_models_for_partitions(
            fit_frame,
            validation_frame,
            random_seed=random_seed,
            catboost_params=catboost_params,
            excluded_feature_prefixes=("graph_",),
        )
        scores = {
            "catboost": table_models.predict_proba(validation_frame)["catboost"],
        }
        if graph_columns:
            graph_models = fit_table_models_for_partitions(
                fit_frame,
                validation_frame,
                random_seed=random_seed,
                catboost_params=catboost_params,
            )
            scores["graph_stats_catboost"] = graph_models.predict_proba(validation_frame)[
                "catboost"
            ]
        fold_result = validation_frame.loc[
            :,
            [
                CANONICAL.transaction_id,
                CANONICAL.event_ts,
                CANONICAL.is_laundering,
            ],
        ].copy()
        fold_result["oof_fold_id"] = fold.fold_id
        for model_name, model_scores in scores.items():
            fold_result[model_name] = model_scores
        results.append(fold_result)
    output = pd.concat(results, ignore_index=True)
    if output[CANONICAL.transaction_id].duplicated().any():
        raise AssertionError("OOF transaction IDs must not appear in multiple folds.")
    return output.sort_values(CANONICAL.event_ts, kind="stable").reset_index(drop=True)


def generate_graphsage_oof_predictions(
    training_frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    minimum_training_months: int = 2,
    config: GraphSAGETrainingConfig | None = None,
) -> pd.DataFrame:
    """Generate GraphSAGE OOF probabilities using only earlier-month graph history."""
    required = {
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.is_laundering,
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        CANONICAL.source_row_number,
    }
    missing = sorted(required.difference(training_frame.columns))
    if missing:
        raise ValueError(f"Graph OOF input requires: {', '.join(missing)}")
    timestamps = pd.to_datetime(training_frame[CANONICAL.event_ts], utc=True, errors="raise")
    month_values = timestamps.dt.strftime("%Y-%m")
    folds = make_expanding_time_folds(
        training_frame,
        n_splits=n_splits,
        minimum_training_months=minimum_training_months,
    )
    edge_features = select_graph_edge_features(training_frame)
    configuration = config or GraphSAGETrainingConfig()
    results: list[pd.DataFrame] = []
    for fold in folds:
        fit_frame = training_frame.loc[month_values.isin(fold.training_months)].copy()
        validation_frame = training_frame.loc[
            month_values.isin(fold.validation_months)
        ].copy()
        validation_frame = validation_frame.sort_values(
            [CANONICAL.event_ts, CANONICAL.source_row_number],
            kind="stable",
        )
        node_indexer = TemporalNodeIndexer().fit(fit_frame)
        builder = DailyGraphSnapshotBuilder(
            node_indexer,
            edge_feature_columns=edge_features,
            history_window=pd.Timedelta(days=configuration.history_window_days),
        )
        raw_training = builder.build(fit_frame)
        raw_validation = builder.build(validation_frame)
        scaler = fit_edge_feature_scaler(raw_training)
        training_snapshots = transform_edge_features(raw_training, scaler)
        validation_snapshots = transform_edge_features(raw_validation, scaler)
        trained = fit_graphsage(
            training_snapshots,
            validation_snapshots,
            num_nodes=node_indexer.num_nodes,
            config=configuration,
        )
        scores = predict_graphsage(
            trained,
            validation_snapshots,
            num_nodes=node_indexer.num_nodes,
        )
        fold_result = validation_frame.loc[
            :,
            [
                CANONICAL.transaction_id,
                CANONICAL.event_ts,
                CANONICAL.is_laundering,
            ],
        ].copy()
        fold_result["oof_fold_id"] = fold.fold_id
        fold_result["graphsage"] = scores
        results.append(fold_result)
    output = pd.concat(results, ignore_index=True)
    if output[CANONICAL.transaction_id].duplicated().any():
        raise AssertionError("Graph OOF transaction IDs must not appear in multiple folds.")
    return output.sort_values(CANONICAL.event_ts, kind="stable").reset_index(drop=True)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Use overwrite only for private, regenerable OOF artifacts."
        )
    if not output_dir.is_dir():
        raise ValueError(f"OOF output path must be a directory: {output_dir}")
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--minimum-training-months", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--model", choices=("table", "graphsage"), default="table")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=4,
        help="Maximum CUDA devices for GraphSAGE OOF when --device is cuda/auto.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _prepare_output_dir(args.output, overwrite=args.overwrite)
    model_configuration = load_model_configuration(args.model_config)
    training = load_feature_split(args.features, TimeSplit.TRAIN)
    if args.model == "table":
        predictions = generate_table_oof_predictions(
            training,
            n_splits=args.splits,
            minimum_training_months=args.minimum_training_months,
            random_seed=args.random_seed,
            catboost_params=catboost_parameters_from_configuration(model_configuration),
        )
    else:
        graphsage_parameters = graphsage_parameters_from_configuration(model_configuration)
        if args.epochs is not None:
            graphsage_parameters["epochs"] = args.epochs
        predictions = generate_graphsage_oof_predictions(
            training,
            n_splits=args.splits,
            minimum_training_months=args.minimum_training_months,
            config=GraphSAGETrainingConfig(
                device=args.device,
                max_gpus=args.max_gpus,
                random_seed=args.random_seed,
                **graphsage_parameters,
            ),
        )
    output_path = args.output / f"{args.model}_oof_scores.parquet"
    predictions.to_parquet(output_path, index=False)
    manifest = create_run_manifest(
        output_dir=args.output,
        command=f"aml-generate-{args.model}-oof",
        random_seed=args.random_seed,
        input_paths={"pit_feature_dataset": args.features},
        config_paths={"model_config": args.model_config},
        metadata={
            "fold_count": args.splits,
            "minimum_training_months": args.minimum_training_months,
            "model": args.model,
            "oof_rows": len(predictions),
        },
    )
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "oof_rows": len(predictions),
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
