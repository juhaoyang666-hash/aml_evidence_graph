"""Historical-neighbor GraphSAGE training for AML transaction edge classification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import parallel_apply, replicate
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.graph.snapshots import TemporalGraphSnapshot
from aml_evidence_graph.models.edge_classifiers import build_edge_classifier

# Cap multi-GPU fan-out even when the host exposes more cards.
MAX_GRAPHSAGE_GPUS = 4


@dataclass(frozen=True)
class GraphSAGETrainingConfig:
    """Conservative defaults for a 6GB GPU; batches only sample past neighbors."""

    architecture: str = "graphsage"
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
    max_gpus: int = MAX_GRAPHSAGE_GPUS
    random_seed: int = 20260722
    num_relations: int = 1


@dataclass
class TrainedGraphSAGE:
    """The selected graph model and validation-only training history."""

    model: nn.Module
    config: GraphSAGETrainingConfig
    device: torch.device
    device_ids: tuple[int, ...] = ()
    epoch_history: list[dict[str, float]] = field(default_factory=list)


def resolve_device(requested: str) -> torch.device:
    """Resolve explicit CPU/CUDA selection without silently falling back from CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return device


def resolve_cuda_device_ids(
    requested: str,
    *,
    max_gpus: int = MAX_GRAPHSAGE_GPUS,
) -> tuple[torch.device, tuple[int, ...]]:
    """Pick a primary device and optional CUDA ids (at most ``max_gpus``).

    Explicit ``cuda:N`` stays single-GPU. Plain ``cuda`` / ``auto`` (when CUDA is
    available) uses ``min(max_gpus, torch.cuda.device_count())`` visible devices,
    so ``CUDA_VISIBLE_DEVICES=0,1,2,3`` caps usage even on 8-GPU hosts.
    """
    if max_gpus < 1:
        raise ValueError("max_gpus must be at least 1.")
    device = resolve_device(requested)
    if device.type != "cuda":
        return device, ()
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("CUDA was requested but no visible CUDA devices are available.")
    if device.index is not None:
        if device.index < 0 or device.index >= available:
            raise RuntimeError(
                f"Requested {device} but only {available} visible CUDA device(s) exist."
            )
        return device, (device.index,)
    device_ids = tuple(range(min(max_gpus, available)))
    return torch.device(f"cuda:{device_ids[0]}"), device_ids


class _EdgeSplitDataParallel(nn.Module):
    """DataParallel-style helper for neighbor-sampled edge classifiers.

    Standard ``nn.DataParallel`` scatters every tensor on dim-0 and breaks
    ``edge_index``. This wrapper replicates the sampled subgraph on each GPU and
    only splits the scoring edges / edge features.
    """

    def __init__(self, module: nn.Module, device_ids: tuple[int, ...]) -> None:
        super().__init__()
        if len(device_ids) < 2:
            raise ValueError("Edge-split DataParallel requires at least two device ids.")
        self.module = module
        self.device_ids = tuple(int(device_id) for device_id in device_ids)
        self.output_device = torch.device(f"cuda:{self.device_ids[0]}")

    def forward(
        self,
        node_ids: torch.Tensor,
        history_edge_index: torch.Tensor,
        scoring_edge_index: torch.Tensor,
        scoring_edge_features: torch.Tensor,
        history_edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_edges = int(scoring_edge_features.size(0))
        if num_edges <= 1:
            return self.module(
                node_ids.to(self.output_device),
                history_edge_index.to(self.output_device),
                scoring_edge_index.to(self.output_device),
                scoring_edge_features.to(self.output_device),
                None
                if history_edge_type is None
                else history_edge_type.to(self.output_device),
            )
        used_ids = self.device_ids[: min(len(self.device_ids), num_edges)]
        replicas = replicate(self.module, used_ids)
        chunk_sizes = [num_edges // len(used_ids)] * len(used_ids)
        for index in range(num_edges % len(used_ids)):
            chunk_sizes[index] += 1
        inputs: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []
        offset = 0
        for device_id, chunk_size in zip(used_ids, chunk_sizes, strict=True):
            device = torch.device(f"cuda:{device_id}")
            end = offset + chunk_size
            inputs.append(
                (
                    node_ids.to(device, non_blocking=True),
                    history_edge_index.to(device, non_blocking=True),
                    scoring_edge_index[:, offset:end].to(device, non_blocking=True),
                    scoring_edge_features[offset:end].to(device, non_blocking=True),
                    None
                    if history_edge_type is None
                    else history_edge_type.to(device, non_blocking=True),
                )
            )
            offset = end
        outputs = parallel_apply(replicas, inputs)
        return torch.cat([output.to(self.output_device) for output in outputs], dim=0)


def _wrap_for_devices(model: nn.Module, device_ids: tuple[int, ...]) -> nn.Module:
    if len(device_ids) > 1:
        return _EdgeSplitDataParallel(model, device_ids)
    return model


def _make_loader(
    snapshot: TemporalGraphSnapshot,
    *,
    num_nodes: int,
    config: GraphSAGETrainingConfig,
    shuffle: bool,
    batch_size: int | None = None,
) -> LinkNeighborLoader:
    edge_index = torch.from_numpy(np.ascontiguousarray(snapshot.history_edge_index)).long()
    data_kwargs: dict[str, Any] = {
        "x": torch.arange(num_nodes, dtype=torch.long),
        "edge_index": edge_index,
    }
    if snapshot.history_edge_type is not None and snapshot.history_edge_type.size:
        # Store relation ids in edge_attr so LinkNeighborLoader keeps them on sampled edges.
        data_kwargs["edge_attr"] = torch.from_numpy(
            np.ascontiguousarray(snapshot.history_edge_type)
        ).long().view(-1, 1)
    data = Data(**data_kwargs)
    return LinkNeighborLoader(
        data,
        num_neighbors=list(config.num_neighbors),
        edge_label_index=torch.from_numpy(
            np.ascontiguousarray(snapshot.scoring_edge_index)
        ).long(),
        edge_label=torch.from_numpy(np.array(snapshot.labels, copy=True)).long(),
        batch_size=config.batch_size if batch_size is None else batch_size,
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


def _effective_batch_size(config: GraphSAGETrainingConfig, device_ids: tuple[int, ...]) -> int:
    """Keep per-GPU scoring load near the configured batch size when using multi-GPU."""
    gpu_count = max(len(device_ids), 1)
    return int(config.batch_size) * gpu_count


def _batch_logits(
    model: nn.Module,
    batch: Data,
    edge_features: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch.input_id.to("cpu")
    batch_edge_features = edge_features[input_ids]
    history_edge_type = None
    if getattr(batch, "edge_attr", None) is not None:
        history_edge_type = batch.edge_attr.view(-1)
    if isinstance(model, _EdgeSplitDataParallel):
        # Keep the neighbor-sampled tensors on CPU; the wrapper places replicas.
        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_label_index,
            batch_edge_features,
            history_edge_type,
        )
        labels = batch.edge_label.float().to(device)
        return logits, labels
    batch = batch.to(device)
    batch_edge_features = batch_edge_features.to(device)
    if history_edge_type is not None:
        history_edge_type = history_edge_type.to(device)
    logits = model(
        batch.x,
        batch.edge_index,
        batch.edge_label_index,
        batch_edge_features,
        history_edge_type,
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
    rng_devices = list(trained.device_ids)
    if trained.device.type == "cuda" and not rng_devices:
        rng_devices = [
            trained.device.index
            if trained.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(trained.config.random_seed)
        if trained.device.type == "cuda":
            torch.cuda.manual_seed_all(trained.config.random_seed)
        inference_model = _wrap_for_devices(trained.model, trained.device_ids)
        inference_model.eval()
        probabilities: list[np.ndarray] = []
        for snapshot in snapshots:
            loader = _make_loader(
                snapshot,
                num_nodes=num_nodes,
                config=trained.config,
                shuffle=False,
                batch_size=_effective_batch_size(trained.config, trained.device_ids),
            )
            snapshot_scores = np.empty(len(snapshot.labels), dtype=np.float64)
            edge_features = torch.from_numpy(snapshot.edge_features).float()
            for batch in loader:
                input_ids = batch.input_id.detach().cpu().numpy()
                logits, _ = _batch_logits(
                    inference_model,
                    batch,
                    edge_features,
                    device=trained.device,
                )
                snapshot_scores[input_ids] = torch.sigmoid(logits).detach().cpu().numpy()
            probabilities.append(snapshot_scores)
        return (
            np.concatenate(probabilities)
            if probabilities
            else np.empty(0, dtype=np.float64)
        )


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
    device, device_ids = resolve_cuda_device_ids(
        configuration.device,
        max_gpus=configuration.max_gpus,
    )
    torch.manual_seed(configuration.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configuration.random_seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    edge_feature_dim = training_snapshots[0].edge_features.shape[1]
    model = build_edge_classifier(
        configuration.architecture,
        num_nodes=num_nodes,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=configuration.hidden_dim,
        num_layers=configuration.num_layers,
        dropout=configuration.dropout,
        num_relations=configuration.num_relations,
    ).to(device)
    train_model = _wrap_for_devices(model, device_ids)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=configuration.learning_rate)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=_positive_class_weight(training_snapshots).to(device)
    )
    validation_labels = np.concatenate(
        [snapshot.labels for snapshot in validation_snapshots]
    )
    loader_batch_size = _effective_batch_size(configuration, device_ids)

    best_state: dict[str, Any] | None = None
    best_pr_auc = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, configuration.epochs + 1):
        train_model.train()
        epoch_losses: list[float] = []
        for snapshot in training_snapshots:
            loader = _make_loader(
                snapshot,
                num_nodes=num_nodes,
                config=configuration,
                shuffle=True,
                batch_size=loader_batch_size,
            )
            edge_features = torch.from_numpy(snapshot.edge_features).float()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits, labels = _batch_logits(
                    train_model,
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
            device_ids=device_ids,
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
        device_ids=device_ids,
        epoch_history=history,
    )
