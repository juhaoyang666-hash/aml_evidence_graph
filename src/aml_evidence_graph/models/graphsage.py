"""Backward-compatible GraphSAGE export; prefer models.edge_classifiers for new code."""

from aml_evidence_graph.models.edge_classifiers import (
    GraphSAGEEdgeClassifier,
    build_edge_classifier,
)

__all__ = ["GraphSAGEEdgeClassifier", "build_edge_classifier"]
