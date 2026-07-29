#!/usr/bin/env python3
"""Train one GAT engineering candidate using train/validation data only.

This runner deliberately never reads the test split. It is intended for selecting
batch size, neighbor fanout, and history-window candidates before one frozen test.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.evaluation.monitoring import measure_runtime
from aml_evidence_graph.graph.snapshots import (
    DailyGraphSnapshotBuilder,
    TemporalNodeIndexer,
    fit_edge_feature_scaler,
    transform_edge_features,
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
from aml_evidence_graph.training.table_baseline import load_feature_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-neighbors", type=int, nargs="+", required=True)
    parser.add_argument("--history-window-days", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=20260722)
    return parser.parse_args()


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

    (training, validation), load_resources = measure_runtime(
        lambda: (
            load_feature_split(args.features, TimeSplit.TRAIN),
            load_feature_split(args.features, TimeSplit.VALIDATION),
        )
    )
    edge_feature_columns = select_graph_edge_features(training)
    node_indexer = TemporalNodeIndexer().fit(training)

    def build_snapshots() -> tuple[object, object, object]:
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
            transform_edge_features(raw_training, scaler),
            transform_edge_features(raw_validation, scaler),
            scaler,
        )

    (training_snapshots, validation_snapshots, scaler), snapshot_resources = measure_runtime(
        build_snapshots
    )
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
