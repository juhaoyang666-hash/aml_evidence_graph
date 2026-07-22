"""Optional trusted local loading for persisted GraphSAGE scoring artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aml_evidence_graph.graph.snapshots import TemporalGraphSnapshot, TemporalNodeIndexer
from aml_evidence_graph.models.graphsage import GraphSAGEEdgeClassifier
from aml_evidence_graph.training.graphsage import (
    GraphSAGETrainingConfig,
    TrainedGraphSAGE,
    predict_graphsage,
    resolve_device,
)


@dataclass
class LoadedGraphSAGEArtifact:
    """Trusted local GraphSAGE artifact with its frozen node map and edge scaler."""

    model: GraphSAGEEdgeClassifier
    config: GraphSAGETrainingConfig
    device: torch.device
    edge_feature_columns: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    node_indexer: TemporalNodeIndexer

    @property
    def num_nodes(self) -> int:
        return self.node_indexer.num_nodes

    def predict(self, snapshots: list[TemporalGraphSnapshot]) -> np.ndarray:
        """Score historical-only snapshots using the persisted scaling parameters."""
        scaled_snapshots = [
            TemporalGraphSnapshot(
                event_date=snapshot.event_date,
                history_edge_index=snapshot.history_edge_index,
                scoring_edge_index=snapshot.scoring_edge_index,
                edge_features=(
                    (snapshot.edge_features - self.scaler_mean) / self.scaler_scale
                ).astype(np.float32),
                labels=snapshot.labels,
                transaction_ids=snapshot.transaction_ids,
            )
            for snapshot in snapshots
        ]
        return predict_graphsage(
            TrainedGraphSAGE(model=self.model, config=self.config, device=self.device),
            scaled_snapshots,
            num_nodes=self.num_nodes,
        )


def load_graphsage_artifact(
    artifact_path: Path,
    *,
    device: str = "auto",
) -> LoadedGraphSAGEArtifact:
    """Load a trusted local GraphSAGE checkpoint without accepting uploaded files."""
    if not artifact_path.is_file():
        raise FileNotFoundError(f"GraphSAGE artifact does not exist: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    required = {
        "state_dict",
        "config",
        "edge_feature_columns",
        "scaler_mean",
        "scaler_scale",
        "known_node_index",
        "unknown_hash_buckets",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("GraphSAGE artifact is missing required checkpoint fields.")
    raw_config = dict(payload["config"])
    raw_config["num_neighbors"] = tuple(raw_config["num_neighbors"])
    raw_config["device"] = device
    config = GraphSAGETrainingConfig(**raw_config)
    if config.history_window_days < 1:
        raise ValueError("GraphSAGE artifact has an invalid history_window_days value.")
    edge_feature_columns = tuple(str(value) for value in payload["edge_feature_columns"])
    if not edge_feature_columns:
        raise ValueError("GraphSAGE artifact must declare at least one edge feature.")
    scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
    if (
        scaler_mean.ndim != 1
        or scaler_scale.ndim != 1
        or len(scaler_mean) != len(edge_feature_columns)
        or len(scaler_scale) != len(edge_feature_columns)
        or not np.isfinite(scaler_mean).all()
        or not np.isfinite(scaler_scale).all()
        or (scaler_scale <= 0).any()
    ):
        raise ValueError("GraphSAGE edge scaler is invalid for the saved feature schema.")
    unknown_hash_buckets = int(payload["unknown_hash_buckets"])
    raw_known_nodes = payload["known_node_index"]
    if not isinstance(raw_known_nodes, dict):
        raise ValueError("GraphSAGE known node map must be a dictionary.")
    known_nodes = {str(identifier): int(index) for identifier, index in raw_known_nodes.items()}
    if len(set(known_nodes.values())) != len(known_nodes) or any(
        index < 1 for index in known_nodes.values()
    ):
        raise ValueError("GraphSAGE known node map contains invalid indices.")
    indexer = TemporalNodeIndexer(unknown_hash_buckets=unknown_hash_buckets)
    indexer._known_nodes = known_nodes
    resolved_device = resolve_device(device)
    model = GraphSAGEEdgeClassifier(
        num_nodes=indexer.num_nodes,
        edge_feature_dim=len(edge_feature_columns),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(resolved_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedGraphSAGEArtifact(
        model=model,
        config=config,
        device=resolved_device,
        edge_feature_columns=edge_feature_columns,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        node_indexer=indexer,
    )
