import torch

from aml_evidence_graph.models.edge_classifiers import (
    SUPPORTED_EDGE_ARCHITECTURES,
    build_edge_classifier,
)


def test_edge_classifier_architectures_share_forward_signature() -> None:
    num_nodes = 8
    edge_feature_dim = 3
    node_ids = torch.arange(num_nodes, dtype=torch.long)
    history_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    scoring_edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    scoring_edge_features = torch.randn(2, edge_feature_dim)

    for architecture in SUPPORTED_EDGE_ARCHITECTURES:
        model = build_edge_classifier(
            architecture,
            num_nodes=num_nodes,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=16,
            num_layers=2,
            dropout=0.0,
        )
        logits = model(
            node_ids,
            history_edge_index,
            scoring_edge_index,
            scoring_edge_features,
        )
        assert logits.shape == (2,)
        assert model.architecture == architecture


def test_unsupported_edge_architecture_is_rejected() -> None:
    try:
        build_edge_classifier(
            "transformer",
            num_nodes=4,
            edge_feature_dim=2,
        )
    except ValueError as exc:
        assert "Unsupported edge GNN architecture" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported architecture.")
