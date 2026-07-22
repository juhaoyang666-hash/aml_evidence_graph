"""Historical-neighbor GraphSAGE training for AML transaction edge classification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.graph.snapshots import TemporalGraphSnapshot
from aml_evidence_graph.models.graphsage import GraphSAGEEdgeClassifier


@dataclass(frozen=True)
class GraphSAGETrainingConfig:
    """Conservative defaults for a 6GB GPU; batches only sample past neighbors."""

    hidden_dim: int = 64
    num_layers: int = 2
    num_neighbors: tuple[int, ...] = (15, 10)
    batch_size: int = 2_048
    epochs: int = 12
    learning_rate: float = 0.001
    dropout: float = 0.15
    patience: int = 3
    history_window_days: int = 30
    device: str = "auto"
    random_seed: int = 20260722


@dataclass
class TrainedGraphSAGE:
    """The selected graph model and validation-only training history."""

    model: GraphSAGEEdgeClassifier
    config: GraphSAGETrainingConfig
    device: torch.device
    epoch_history: list[dict[str, float]] = field(default_factory=list)


def resolve_device(requested: str) -> torch.device:
    """Resolve explicit CPU/CUDA selection without silently falling back from CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return device


def _make_loader(
    snapshot: TemporalGraphSnapshot,
    *,
    num_nodes: int,
    config: GraphSAGETrainingConfig,
    shuffle: bool,
) -> LinkNeighborLoader:
    data = Data(
        x=torch.arange(num_nodes, dtype=torch.long),
        edge_index=torch.from_numpy(
            np.ascontiguousarray(snapshot.history_edge_index)
        ).long(),
    )
    return LinkNeighborLoader(
        data,
        num_neighbors=list(config.num_neighbors),
        edge_label_index=torch.from_numpy(
            np.ascontiguousarray(snapshot.scoring_edge_index)
        ).long(),
        edge_label=torch.from_numpy(np.array(snapshot.labels, copy=True)).long(),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        neg_sampling=None,
    )


def _positive_class_weight(snapshots: list[TemporalGraphSnapshot]) -> torch.Tensor:
    labels = np.concatenate([snapshot.labels for snapshot in snapshots])
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Graph training needs both label classes in the training period.")
    return torch.tensor(negative_count / positive_count, dtype=torch.float32)


def _batch_logits(
    model: GraphSAGEEdgeClassifier,
    batch: Data,
    edge_features: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch.input_id.to("cpu")
    batch = batch.to(device)
    batch_edge_features = edge_features[input_ids].to(device)
    logits = model(
        batch.x,
        batch.edge_index,
        batch.edge_label_index,
        batch_edge_features,
    )
    return logits, batch.edge_label.float()


@torch.no_grad()
def predict_graphsage(
    trained: TrainedGraphSAGE,
    snapshots: list[TemporalGraphSnapshot],
    *,
    num_nodes: int,
) -> np.ndarray:
    """Score every supplied edge exactly once; no test-period edge is resampled."""
    trained.model.eval()
    probabilities: list[np.ndarray] = []
    for snapshot in snapshots:
        loader = _make_loader(
            snapshot,
            num_nodes=num_nodes,
            config=trained.config,
            shuffle=False,
        )
        snapshot_scores = np.empty(len(snapshot.labels), dtype=np.float64)
        edge_features = torch.from_numpy(snapshot.edge_features).float()
        for batch in loader:
            input_ids = batch.input_id.detach().cpu().numpy()
            logits, _ = _batch_logits(
                trained.model,
                batch,
                edge_features,
                device=trained.device,
            )
            snapshot_scores[input_ids] = torch.sigmoid(logits).detach().cpu().numpy()
        probabilities.append(snapshot_scores)
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float64)


def fit_graphsage(
    training_snapshots: list[TemporalGraphSnapshot],
    validation_snapshots: list[TemporalGraphSnapshot],
    *,
    num_nodes: int,
    config: GraphSAGETrainingConfig | None = None,
) -> TrainedGraphSAGE:
    """Train GraphSAGE only on train snapshots and select epochs using validation PR-AUC."""
    if not training_snapshots or not validation_snapshots:
        raise ValueError("Both training and validation graph snapshots are required.")
    configuration = config or GraphSAGETrainingConfig()
    if len(configuration.num_neighbors) != configuration.num_layers:
        raise ValueError("num_neighbors length must equal num_layers.")
    if configuration.history_window_days < 1:
        raise ValueError("history_window_days must be positive.")
    device = resolve_device(configuration.device)
    torch.manual_seed(configuration.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configuration.random_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    edge_feature_dim = training_snapshots[0].edge_features.shape[1]
    model = GraphSAGEEdgeClassifier(
        num_nodes=num_nodes,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=configuration.hidden_dim,
        num_layers=configuration.num_layers,
        dropout=configuration.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=configuration.learning_rate)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=_positive_class_weight(training_snapshots).to(device)
    )
    validation_labels = np.concatenate(
        [snapshot.labels for snapshot in validation_snapshots]
    )

    best_state: dict[str, Any] | None = None
    best_pr_auc = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, configuration.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for snapshot in training_snapshots:
            loader = _make_loader(
                snapshot,
                num_nodes=num_nodes,
                config=configuration,
                shuffle=True,
            )
            edge_features = torch.from_numpy(snapshot.edge_features).float()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits, labels = _batch_logits(
                    model,
                    batch,
                    edge_features,
                    device=device,
                )
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

        candidate = TrainedGraphSAGE(
            model=model,
            config=configuration,
            device=device,
        )
        validation_scores = predict_graphsage(
            candidate,
            validation_snapshots,
            num_nodes=num_nodes,
        )
        validation_metrics = evaluate_binary_risk_scores(
            validation_labels,
            validation_scores,
        )
        pr_auc = float(validation_metrics["pr_auc"])
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(epoch_losses)),
                "validation_pr_auc": pr_auc,
            }
        )
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= configuration.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainedGraphSAGE(
        model=model,
        config=configuration,
        device=device,
        epoch_history=history,
    )
