#!/usr/bin/env python3
"""Train one configured graph-model candidate using train/validation data only.

The model configuration selects GraphSAGE, GAT, RGCN, or PNA. This runner deliberately
never reads the test split and is intended for controlled candidate comparisons.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.evaluation.monitoring import measure_runtime, measure_runtime_rss_only
from aml_evidence_graph.features.cold_start import COLD_START_FEATURE_FAMILIES
from aml_evidence_graph.graph.snapshots import (
    DailyGraphSnapshotBuilder,
    TemporalNodeIndexer,
    fit_edge_feature_scaler,
    transform_edge_features_in_place,
)
from aml_evidence_graph.training.configuration import (
    graphsage_parameters_from_configuration,
    load_model_configuration,
)
from aml_evidence_graph.training.graphsage import (
    GraphSAGETrainingConfig,
    fit_graphsage,
    predict_graphsage,
)
from aml_evidence_graph.training.run_graphsage import (
    _write_graph_scores,
    select_graph_edge_features,
)
from aml_evidence_graph.training.table_baseline import (
    feature_split_schema,
    load_feature_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-neighbors", type=int, nargs="+", required=True)
    parser.add_argument("--history-window-days", type=int, required=True)
    parser.add_argument(
        "--exclude-feature-family",
        action="append",
        choices=("amount", "temporal_behavior", "relationship", "node_stats"),
        default=[],
        help="Repeat to remove pre-declared feature families before fitting.",
    )
    parser.add_argument(
        "--exclude-cold-start-family",
        action="append",
        choices=tuple(COLD_START_FEATURE_FAMILIES),
        default=[],
        help="Repeat to remove one registered cold-start-v3 candidate family.",
    )
    parser.add_argument(
        "--exclude-feature-column",
        action="append",
        default=[],
        help="Repeat to remove one exact edge-feature column for targeted ablation.",
    )
    parser.add_argument("--random-seed", type=int, default=20260722)
    return parser.parse_args()


def _exclude_feature_families(
    columns: tuple[str, ...], families: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    predicates = {
        "amount": lambda column: column == CANONICAL.amount or "_amount_" in column,
        "temporal_behavior": lambda column: column.startswith(
            ("sender_outgoing_", "receiver_incoming_")
        ),
        "relationship": lambda column: column.startswith("relationship_"),
        "node_stats": lambda column: column.startswith("graph_"),
    }
    excluded = tuple(
        column for column in columns if any(predicates[family](column) for family in families)
    )
    for family in families:
        if not any(predicates[family](column) for column in columns):
            raise ValueError(f"Feature family has no matching columns: {family}")
    retained = tuple(column for column in columns if column not in set(excluded))
    if not retained:
        raise ValueError("Feature-family exclusions removed every graph edge feature.")
    return retained, excluded


def _exclude_cold_start_feature_families(
    columns: tuple[str, ...], families: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    requested_families = tuple(dict.fromkeys(families))
    expected = {
        column for family in requested_families for column in COLD_START_FEATURE_FAMILIES[family]
    }
    missing = sorted(expected.difference(columns))
    if missing:
        raise ValueError(
            "Cold-start family exclusion requires missing feature columns: " + ", ".join(missing)
        )
    excluded = tuple(column for column in columns if column in expected)
    retained = tuple(column for column in columns if column not in expected)
    if not retained:
        raise ValueError("Cold-start family exclusions removed every graph edge feature.")
    return retained, excluded


def _exclude_exact_feature_columns(
    columns: tuple[str, ...], requested: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    requested_columns = tuple(dict.fromkeys(requested))
    missing = sorted(set(requested_columns).difference(columns))
    if missing:
        raise ValueError("Exact feature exclusions are missing: " + ", ".join(missing))
    excluded = tuple(column for column in columns if column in requested_columns)
    retained = tuple(column for column in columns if column not in requested_columns)
    if not retained:
        raise ValueError("Exact feature exclusions removed every graph edge feature.")
    return retained, excluded


def _build_scaled_snapshots(
    training: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    node_indexer: TemporalNodeIndexer,
    edge_feature_columns: tuple[str, ...],
    configuration: GraphSAGETrainingConfig,
) -> tuple[object, object, object]:
    builder = DailyGraphSnapshotBuilder(
        node_indexer,
        edge_feature_columns=edge_feature_columns,
        history_window=timedelta(days=configuration.history_window_days),
        store_relation_types=configuration.num_relations > 1,
    )
    raw_training = builder.build(training)
    raw_validation = builder.build(validation)
    scaler = fit_edge_feature_scaler(raw_training)
    return (
        transform_edge_features_in_place(raw_training, scaler),
        transform_edge_features_in_place(raw_validation, scaler),
        scaler,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate output: {args.output}")
    args.output.mkdir(parents=True)
    started_at = time.perf_counter()

    model_document = load_model_configuration(args.model_config)
    parameters = graphsage_parameters_from_configuration(model_document)
    parameters.update(
        {
            "batch_size": args.batch_size,
            "num_neighbors": tuple(args.num_neighbors),
            "history_window_days": args.history_window_days,
        }
    )
    configuration = GraphSAGETrainingConfig(
        device=args.device,
        max_gpus=1,
        random_seed=args.random_seed,
        **parameters,
    )

    training_schema = feature_split_schema(args.features, TimeSplit.TRAIN)
    schema_frame = pl.DataFrame(schema=training_schema)
    all_edge_feature_columns = select_graph_edge_features(schema_frame)
    edge_feature_columns, excluded_feature_columns = _exclude_feature_families(
        all_edge_feature_columns, args.exclude_feature_family
    )
    edge_feature_columns, excluded_cold_start_columns = _exclude_cold_start_feature_families(
        edge_feature_columns,
        args.exclude_cold_start_family,
    )
    edge_feature_columns, excluded_exact_columns = _exclude_exact_feature_columns(
        edge_feature_columns,
        args.exclude_feature_column,
    )
    required_columns = tuple(
        dict.fromkeys(
            (
                CANONICAL.transaction_id,
                CANONICAL.event_ts,
                CANONICAL.sender_account_id,
                CANONICAL.receiver_account_id,
                CANONICAL.source_row_number,
                CANONICAL.is_laundering,
                *edge_feature_columns,
            )
        )
    )
    (training, validation), load_resources = measure_runtime(
        lambda: (
            load_feature_split(
                args.features,
                TimeSplit.TRAIN,
                columns=required_columns,
            ),
            load_feature_split(
                args.features,
                TimeSplit.VALIDATION,
                columns=required_columns,
            ),
        )
    )
    node_indexer = TemporalNodeIndexer().fit(training)

    (
        (
            training_snapshots,
            validation_snapshots,
            scaler,
        ),
        snapshot_resources,
    ) = measure_runtime_rss_only(
        lambda: _build_scaled_snapshots(
            training,
            validation,
            node_indexer=node_indexer,
            edge_feature_columns=edge_feature_columns,
            configuration=configuration,
        )
    )
    training = None
    gc.collect()
    trained, training_resources = measure_runtime(
        lambda: fit_graphsage(
            training_snapshots,
            validation_snapshots,
            num_nodes=node_indexer.num_nodes,
            config=configuration,
        )
    )
    validation_scores, inference_resources = measure_runtime(
        lambda: predict_graphsage(
            trained,
            validation_snapshots,
            num_nodes=node_indexer.num_nodes,
        )
    )
    validation_labels = np.concatenate(
        [snapshot.labels for snapshot in validation_snapshots], dtype=int
    )
    validation_metrics = evaluate_binary_risk_scores(validation_labels, validation_scores)
    _write_graph_scores(
        args.output,
        split_name="validation",
        frame=validation,
        scores=validation_scores,
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
            "selection_scope": "train_validation_only",
        },
        args.output / "graphsage.pt",
    )
    summary = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_scope": "train_validation_only",
        "test_split_read": False,
        "feature_root": str(args.features),
        "model_config": str(args.model_config),
        "configuration": asdict(configuration),
        "edge_feature_columns": list(edge_feature_columns),
        "excluded_feature_families": args.exclude_feature_family,
        "excluded_feature_columns": list(excluded_feature_columns),
        "excluded_cold_start_feature_families": args.exclude_cold_start_family,
        "excluded_cold_start_feature_columns": list(excluded_cold_start_columns),
        "excluded_exact_feature_columns": list(excluded_exact_columns),
        "num_nodes": node_indexer.num_nodes,
        "training_snapshot_count": len(training_snapshots),
        "validation_snapshot_count": len(validation_snapshots),
        "validation_metrics": validation_metrics,
        "epoch_history": trained.epoch_history,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started_at,
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(trained.device) / 1024 / 1024
                if trained.device.type == "cuda"
                else 0.0
            ),
            **{f"load_{key}": value for key, value in load_resources.items()},
            **{f"snapshot_{key}": value for key, value in snapshot_resources.items()},
            **{f"training_{key}": value for key, value in training_resources.items()},
            **{f"validation_inference_{key}": value for key, value in inference_resources.items()},
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **summary["validation_metrics"]}))


if __name__ == "__main__":
    main()
