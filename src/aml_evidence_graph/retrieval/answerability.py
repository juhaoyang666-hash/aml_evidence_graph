"""Frozen feature contract for the retrieval answerability gate."""

from __future__ import annotations

import numpy as np

from aml_evidence_graph.retrieval.contracts import TypologyScoredRetriever
from aml_evidence_graph.retrieval.evaluation import RetrievalCase
from aml_evidence_graph.retrieval.retrievers import (
    BM25TypologyRetriever,
    DenseTypologyRetriever,
    SentenceTransformerEncoder,
)

ANSWERABILITY_FEATURE_CONTRACT = (
    "normalized_embedding+dense_top+dense_margin+bm25_top+bm25_margin+"
    "query_length+non_ascii_ratio"
)


def build_answerability_features(
    cases: list[RetrievalCase],
    encoder: SentenceTransformerEncoder,
    dense: DenseTypologyRetriever,
    bm25: BM25TypologyRetriever,
) -> np.ndarray:
    embeddings = encoder.encode([case.query for case in cases])
    signals: list[list[float]] = []
    for case in cases:
        dense_hits = dense.search(case.query, limit=3)
        bm25_hits = bm25.search(case.query, limit=3)
        dense_scores = [hit.score for hit in dense_hits]
        bm25_scores = [hit.score for hit in bm25_hits]
        non_ascii = sum(ord(char) > 127 for char in case.query)
        signals.append(
            [
                dense_scores[0] if dense_scores else 0.0,
                dense_scores[0] - dense_scores[1] if len(dense_scores) > 1 else 0.0,
                bm25_scores[0] if bm25_scores else 0.0,
                bm25_scores[0] - bm25_scores[1] if len(bm25_scores) > 1 else 0.0,
                float(len(case.query)),
                non_ascii / max(len(case.query), 1),
            ]
        )
    return np.column_stack([embeddings, np.asarray(signals, dtype=np.float32)])


class ProbabilityGatedRetriever:
    """Reject queries below a precomputed answerability probability threshold."""

    def __init__(
        self,
        base: TypologyScoredRetriever,
        probabilities: dict[str, float],
        threshold: float,
    ) -> None:
        self.base = base
        self.probabilities = probabilities
        self.threshold = threshold
        self.name = "hybrid-rrf+answerability-gate"

    def search(self, query: str, *, limit: int = 3) -> list[object]:
        if self.probabilities[query] < self.threshold:
            return []
        return self.base.search(query, limit=limit)
