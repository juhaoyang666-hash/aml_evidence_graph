"""Shared transaction-edge GNN classifiers (GraphSAGE / GAT / RGCN / PNA)."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import GATConv, PNAConv, RGCNConv, SAGEConv

SUPPORTED_EDGE_ARCHITECTURES = ("graphsage", "gat", "rgcn", "pna")
EdgeArchitecture = Literal["graphsage", "gat", "rgcn", "pna"]


class _BaseEdgeClassifier(nn.Module):
    """Score current transaction edges from historical sampled neighborhoods."""

    def __init__(
        self,
        *,
        num_nodes: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if num_nodes < 2 or edge_feature_dim < 1 or num_layers < 1:
            raise ValueError("num_nodes, edge_feature_dim, and num_layers must be positive.")
        self.architecture = "base"
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.convolutions = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _propagate(
        self,
        node_state: torch.Tensor,
        history_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        node_ids: torch.Tensor,
        history_edge_index: torch.Tensor,
        scoring_edge_index: torch.Tensor,
        scoring_edge_features: torch.Tensor,
        history_edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for scoring edges; history_edge_index must contain only past edges."""
        del history_edge_type  # used by RGCN subclass override
        node_state = self.node_embedding(node_ids)
        node_state = self._propagate(node_state, history_edge_index)
        source_state = node_state[scoring_edge_index[0]]
        receiver_state = node_state[scoring_edge_index[1]]
        edge_state = self.edge_encoder(scoring_edge_features)
        combined = torch.cat(
            [
                source_state,
                receiver_state,
                torch.abs(source_state - receiver_state),
                edge_state,
            ],
            dim=-1,
        )
        return self.classifier(combined).squeeze(-1)


class GraphSAGEEdgeClassifier(_BaseEdgeClassifier):
    """GraphSAGE message-passing edge classifier."""

    def __init__(self, **kwargs: int | float) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.architecture = "graphsage"
        hidden_dim = int(kwargs.get("hidden_dim", 64))
        num_layers = int(kwargs.get("num_layers", 2))
        self.convolutions = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def _propagate(
        self,
        node_state: torch.Tensor,
        history_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for convolution in self.convolutions:
            node_state = convolution(node_state, history_edge_index)
            node_state = self.dropout(torch.relu(node_state))
        return node_state


class GATEdgeClassifier(_BaseEdgeClassifier):
    """GAT message-passing edge classifier."""

    def __init__(self, **kwargs: int | float) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.architecture = "gat"
        hidden_dim = int(kwargs.get("hidden_dim", 64))
        num_layers = int(kwargs.get("num_layers", 2))
        self.convolutions = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim, heads=1, concat=False) for _ in range(num_layers)]
        )

    def _propagate(
        self,
        node_state: torch.Tensor,
        history_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for convolution in self.convolutions:
            node_state = convolution(node_state, history_edge_index)
            node_state = self.dropout(torch.relu(node_state))
        return node_state


class RGCNEdgeClassifier(_BaseEdgeClassifier):
    """RGCN edge classifier with discrete relation types on historical edges."""

    def __init__(self, *, num_relations: int = 1, **kwargs: int | float) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if num_relations < 1:
            raise ValueError("num_relations must be positive.")
        self.architecture = "rgcn"
        self.num_relations = int(num_relations)
        hidden_dim = int(kwargs.get("hidden_dim", 64))
        num_layers = int(kwargs.get("num_layers", 2))
        self.convolutions = nn.ModuleList(
            [
                RGCNConv(hidden_dim, hidden_dim, num_relations=self.num_relations)
                for _ in range(num_layers)
            ]
        )

    def _propagate(
        self,
        node_state: torch.Tensor,
        history_edge_index: torch.Tensor,
        history_edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_edge_type is None:
            edge_type = torch.zeros(
                history_edge_index.size(1),
                dtype=torch.long,
                device=history_edge_index.device,
            )
        else:
            edge_type = history_edge_type.long().view(-1)
            if edge_type.numel() != history_edge_index.size(1):
                raise ValueError("history_edge_type length must match history edges.")
            if (edge_type < 0).any() or (edge_type >= self.num_relations).any():
                raise ValueError("history_edge_type contains out-of-range relation ids.")
        for convolution in self.convolutions:
            node_state = convolution(node_state, history_edge_index, edge_type)
            node_state = self.dropout(torch.relu(node_state))
        return node_state

    def forward(
        self,
        node_ids: torch.Tensor,
        history_edge_index: torch.Tensor,
        scoring_edge_index: torch.Tensor,
        scoring_edge_features: torch.Tensor,
        history_edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_state = self.node_embedding(node_ids)
        node_state = self._propagate(node_state, history_edge_index, history_edge_type)
        source_state = node_state[scoring_edge_index[0]]
        receiver_state = node_state[scoring_edge_index[1]]
        edge_state = self.edge_encoder(scoring_edge_features)
        combined = torch.cat(
            [
                source_state,
                receiver_state,
                torch.abs(source_state - receiver_state),
                edge_state,
            ],
            dim=-1,
        )
        return self.classifier(combined).squeeze(-1)


class PNAEdgeClassifier(_BaseEdgeClassifier):
    """PNA edge classifier with a stable degree histogram placeholder for scaffolding."""

    def __init__(self, **kwargs: int | float) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.architecture = "pna"
        hidden_dim = int(kwargs.get("hidden_dim", 64))
        num_layers = int(kwargs.get("num_layers", 2))
        # Scaffold-only degree histogram; full training can replace with empirical degrees.
        self.register_buffer("deg", torch.ones(32))
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.convolutions = nn.ModuleList(
            [
                PNAConv(
                    hidden_dim,
                    hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=self.deg,
                )
                for _ in range(num_layers)
            ]
        )

    def _propagate(
        self,
        node_state: torch.Tensor,
        history_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for convolution in self.convolutions:
            node_state = convolution(node_state, history_edge_index)
            node_state = self.dropout(torch.relu(node_state))
        return node_state


def build_edge_classifier(
    architecture: str,
    *,
    num_nodes: int,
    edge_feature_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.15,
    num_relations: int = 1,
) -> _BaseEdgeClassifier:
    """Build a supported edge classifier while keeping a shared forward signature."""
    normalized = architecture.strip().lower()
    if normalized not in SUPPORTED_EDGE_ARCHITECTURES:
        raise ValueError(
            "Unsupported edge GNN architecture: "
            f"{architecture}. Expected one of {SUPPORTED_EDGE_ARCHITECTURES}."
        )
    kwargs = {
        "num_nodes": num_nodes,
        "edge_feature_dim": edge_feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    if normalized == "graphsage":
        return GraphSAGEEdgeClassifier(**kwargs)
    if normalized == "gat":
        return GATEdgeClassifier(**kwargs)
    if normalized == "rgcn":
        return RGCNEdgeClassifier(num_relations=num_relations, **kwargs)
    return PNAEdgeClassifier(**kwargs)
