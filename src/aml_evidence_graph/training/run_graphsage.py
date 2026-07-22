"""CLI orchestration for leakage-safe GraphSAGE training and final test scoring."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.evaluation.monitoring import measure_runtime
from aml_evidence_graph.graph.snapshots import (
    DailyGraphSnapshotBuilder,
    TemporalNodeIndexer,
    fit_edge_feature_scaler,
    graph_population_audit,
    transform_edge_features,
)
from aml_evidence_graph.tracking.run import (
    configure_structured_logger,
    create_run_manifest,
)
from aml_evidence_graph.training.configuration import (
    DEFAULT_MODEL_CONFIG_PATH,
    graphsage_parameters_from_configuration,
    load_model_configuration,
)
from aml_evidence_graph.training.graphsage import (
    GraphSAGETrainingConfig,
    fit_graphsage,
    predict_graphsage,
)
from aml_evidence_graph.training.table_baseline import load_feature_split


@dataclass(frozen=True)
class GraphSAGERunSummary:
    """Aggregate graph experiment record without transaction-level predictions."""

    created_at_utc: str
    feature_root: str
    edge_feature_columns: list[str]
    num_nodes: int
    training_snapshot_count: int
    validation_snapshot_count: int
    test_snapshot_count: int
    validation_metrics: dict[str, object]
    test_metrics: dict[str, object]
    validation_population: dict[str, float | int]
    test_population: dict[str, float | int]
    epoch_history: list[dict[str, float]]
    run_id: str
    runtime: dict[str, float]


def select_graph_edge_features(frame: pd.DataFrame) -> tuple[str, ...]:
    """Select numeric current-edge features while excluding labels, IDs, and dates."""
    preferred = (
        CANONICAL.amount,
        "is_cross_border_current_transaction",
    )
    causal_prefixes = ("sender_outgoing_", "receiver_incoming_", "relationship_", "graph_")
    selected = [
        column
        for column in preferred
        if column in frame and pd.api.types.is_numeric_dtype(frame[column])
    ]
    selected.extend(
        sorted(
            column
            for column in frame.columns
            if column.startswith(causal_prefixes)
            and pd.api.types.is_numeric_dtype(frame[column])
            and column not in selected
        )
    )
    if not selected:
        raise ValueError("No numeric causal edge features are available for GraphSAGE.")
    return tuple(selected)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Use overwrite only for private, regenerable graph artifacts."
        )
    if not output_dir.is_dir():
        raise ValueError(f"Graph output path must be a directory: {output_dir}")
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _write_graph_scores(
    output_dir: Path,
    *,
    split_name: str,
    frame: pd.DataFrame,
    scores: object,
) -> pd.DataFrame:
    """Persist scores in the exact chronological snapshot order used for inference."""
    required = {
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.is_laundering,
        CANONICAL.source_row_number,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Graph score artifact requires columns: " + ", ".join(missing)
        )
    ordered = frame.sort_values(
        [CANONICAL.event_ts, CANONICAL.source_row_number],
        kind="stable",
    )
    score_values = np.asarray(scores, dtype=float)
    if len(score_values) != len(ordered):
        raise ValueError("Graph score count does not match the scored partition.")
    output = ordered.loc[
        :,
        [
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.is_laundering,
        ],
    ].copy().reset_index(drop=True)
    output["graphsage"] = score_values
    score_dir = output_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    output.to_parquet(score_dir / f"graphsage_{split_name}_scores.parquet", index=False)
    return output


def train_and_evaluate_graphsage(
    feature_root: Path,
    output_dir: Path,
    *,
    config: GraphSAGETrainingConfig | None = None,
    model_config_path: Path | None = None,
    overwrite: bool = False,
) -> GraphSAGERunSummary:
    """Fit on train, choose by validation, and evaluate the chronological test once."""
    _prepare_output_dir(output_dir, overwrite=overwrite)
    started_at = time.perf_counter()
    configuration = config or GraphSAGETrainingConfig()
    training = load_feature_split(feature_root, TimeSplit.TRAIN)
    validation = load_feature_split(feature_root, TimeSplit.VALIDATION)
    validation_population = graph_population_audit(training, validation)
    edge_feature_columns = select_graph_edge_features(training)
    node_indexer = TemporalNodeIndexer().fit(training)
    builder = DailyGraphSnapshotBuilder(
        node_indexer,
        edge_feature_columns=edge_feature_columns,
        history_window=pd.Timedelta(days=configuration.history_window_days),
    )
    raw_training = builder.build(training)
    raw_validation = builder.build(validation)
    scaler = fit_edge_feature_scaler(raw_training)
    training_snapshots = transform_edge_features(raw_training, scaler)
    validation_snapshots = transform_edge_features(raw_validation, scaler)
    trained, training_resources = measure_runtime(
        lambda: fit_graphsage(
            training_snapshots,
            validation_snapshots,
            num_nodes=node_indexer.num_nodes,
            config=configuration,
        )
    )
    validation_scores, validation_inference_resources = measure_runtime(
        lambda: predict_graphsage(
            trained,
            validation_snapshots,
            num_nodes=node_indexer.num_nodes,
        )
    )
    # The final test split is deliberately read only after validation selection.
    test = load_feature_split(feature_root, TimeSplit.TEST)
    test_population = graph_population_audit(training, test)
    raw_test = builder.build(test, include_labels=False)
    test_snapshots = transform_edge_features(raw_test, scaler)
    test_scores, test_inference_resources = measure_runtime(
        lambda: predict_graphsage(
            trained,
            test_snapshots,
            num_nodes=node_indexer.num_nodes,
        )
    )
    validation_labels = pd.concat(
        [pd.Series(snapshot.labels) for snapshot in validation_snapshots],
        ignore_index=True,
    )
    test_labels = test.sort_values(
        [CANONICAL.event_ts, CANONICAL.source_row_number],
        kind="stable",
    )[CANONICAL.is_laundering].astype(int).reset_index(drop=True)
    _write_graph_scores(
        output_dir,
        split_name="validation",
        frame=validation,
        scores=validation_scores,
    )
    _write_graph_scores(
        output_dir,
        split_name="test",
        frame=test,
        scores=test_scores,
    )
    manifest = create_run_manifest(
        output_dir=output_dir,
        command="aml-train-graphsage",
        random_seed=trained.config.random_seed,
        input_paths={"pit_feature_dataset": feature_root},
        config_paths={
            "model_config": model_config_path or DEFAULT_MODEL_CONFIG_PATH,
        },
        metadata={
            "edge_feature_columns": list(edge_feature_columns),
            "num_nodes": node_indexer.num_nodes,
            "device": str(trained.device),
            "random_seed": trained.config.random_seed,
        },
    )
    summary = GraphSAGERunSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        feature_root=str(feature_root),
        edge_feature_columns=list(edge_feature_columns),
        num_nodes=node_indexer.num_nodes,
        training_snapshot_count=len(training_snapshots),
        validation_snapshot_count=len(validation_snapshots),
        test_snapshot_count=len(test_snapshots),
        validation_metrics=evaluate_binary_risk_scores(validation_labels, validation_scores),
        test_metrics=evaluate_binary_risk_scores(test_labels, test_scores),
        validation_population=validation_population,
        test_population=test_population,
        epoch_history=trained.epoch_history,
        run_id=manifest.run_id,
        runtime={
            "wall_time_seconds": time.perf_counter() - started_at,
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(trained.device) / 1024 / 1024
                if trained.device.type == "cuda"
                else 0.0
            ),
            **{f"training_{name}": value for name, value in training_resources.items()},
            **{
                f"validation_inference_{name}": value
                for name, value in validation_inference_resources.items()
            },
            **{f"test_inference_{name}": value for name, value in test_inference_resources.items()},
            "validation_rows_per_second": len(validation_labels)
            / max(validation_inference_resources["wall_time_ms"] / 1_000, 1e-9),
            "test_rows_per_second": len(test_labels)
            / max(test_inference_resources["wall_time_ms"] / 1_000, 1e-9),
        },
    )
    configure_structured_logger("aml.graphsage_training", run_id=manifest.run_id).info(
        "graphsage_training_completed",
        extra={
            "model_version": manifest.run_id,
            "duration_ms": summary.runtime["wall_time_seconds"] * 1_000,
        },
    )
    torch.save(
        {
            "state_dict": trained.model.state_dict(),
            "config": asdict(trained.config),
            "edge_feature_columns": edge_feature_columns,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "known_node_index": node_indexer._known_nodes,
            "unknown_hash_buckets": node_indexer.unknown_hash_buckets,
        },
        output_dir / "graphsage.pt",
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
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_configuration = load_model_configuration(args.model_config)
    graphsage_parameters = graphsage_parameters_from_configuration(model_configuration)
    if args.epochs is not None:
        graphsage_parameters["epochs"] = args.epochs
    if args.batch_size is not None:
        graphsage_parameters["batch_size"] = args.batch_size
    summary = train_and_evaluate_graphsage(
        args.features,
        args.output,
        config=GraphSAGETrainingConfig(
            device=args.device,
            random_seed=args.random_seed,
            **graphsage_parameters,
        ),
        model_config_path=args.model_config,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation_pr_auc": summary.validation_metrics["pr_auc"],
                "test_pr_auc": summary.test_metrics["pr_auc"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
