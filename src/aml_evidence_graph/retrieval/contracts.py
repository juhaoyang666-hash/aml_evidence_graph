"""Small contracts shared by lexical, dense, hybrid, and reranking stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from aml_evidence_graph.evidence.typology import TypologyDocument


@dataclass(frozen=True)
class ScoredTypologyDocument:
    """A ranked retrieval candidate with explicit score provenance."""

    document: TypologyDocument
    score: float
    rank: int
    retriever: str


class TypologyScoredRetriever(Protocol):
    """Retriever interface used by evaluation and reciprocal-rank fusion."""

    name: str

    def search(self, query: str, *, limit: int = 3) -> list[ScoredTypologyDocument]: ...


class TextEncoder(Protocol):
    """Pluggable text encoder; remote calls are intentionally unsupported."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class TypologyReranker(Protocol):
    """Reranker that only receives local query text and curated documents."""

    name: str

    def rerank(
        self,
        query: str,
        candidates: list[ScoredTypologyDocument],
        *,
        limit: int,
    ) -> list[ScoredTypologyDocument]: ...
