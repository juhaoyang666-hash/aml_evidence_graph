"""GraphSAGE transaction-edge classifier with an edge-feature MLP."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEdgeClassifier(nn.Module):
    """Score current transaction edges from historical sampled graph neighborhoods."""

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
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.convolutions = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
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

    def forward(
        self,
        node_ids: torch.Tensor,
        history_edge_index: torch.Tensor,
        scoring_edge_index: torch.Tensor,
        scoring_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for scoring edges; history_edge_index must contain only past edges."""
        node_state = self.node_embedding(node_ids)
        for convolution in self.convolutions:
            node_state = convolution(node_state, history_edge_index)
            node_state = self.dropout(torch.relu(node_state))
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
