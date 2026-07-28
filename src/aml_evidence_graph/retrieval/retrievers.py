"""Deterministic lexical, dense, hybrid, and reranking implementations."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from aml_evidence_graph.evidence.typology import TypologyDocument
from aml_evidence_graph.retrieval.contracts import (
    ScoredTypologyDocument,
    TextEncoder,
    TypologyReranker,
    TypologyScoredRetriever,
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower())


def _document_text(document: TypologyDocument) -> str:
    return f"{document.title} {document.body}"


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("limit must be positive.")


class BM25TypologyRetriever:
    """BM25 baseline that suppresses all-zero, no-evidence result sets."""

    name = "bm25"

    def __init__(
        self,
        documents: Sequence[TypologyDocument],
        *,
        minimum_score: float = 0.0,
    ) -> None:
        if not documents:
            raise ValueError("At least one Typology document is required.")
        self.documents = list(documents)
        self.minimum_score = minimum_score
        self._bm25 = BM25Okapi([_tokens(_document_text(doc)) for doc in self.documents])

    def search(self, query: str, *, limit: int = 3) -> list[ScoredTypologyDocument]:
        _validate_limit(limit)
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ordered = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), self.documents[item[0]].typology_id),
        )
        retained = [item for item in ordered if float(item[1]) > self.minimum_score][:limit]
        return [
            ScoredTypologyDocument(
                document=self.documents[index],
                score=float(score),
                rank=rank,
                retriever=self.name,
            )
            for rank, (index, score) in enumerate(retained, start=1)
        ]

    def retrieve(self, query: str, *, limit: int = 3) -> list[TypologyDocument]:
        """Compatibility method for the existing investigation workflow."""
        return [item.document for item in self.search(query, limit=limit)]


class TfidfDenseEncoder:
    """Dependency-light local dense baseline; not a semantic foundation model."""

    name = "tfidf"

    def __init__(self, corpus: Sequence[str]) -> None:
        if not corpus:
            raise ValueError("TF-IDF encoder requires a non-empty corpus.")
        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self._vectorizer.fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.transform(texts).toarray().astype(np.float32)


class SentenceTransformerEncoder:
    """Optional local sentence-transformers adapter loaded only when requested."""

    def __init__(self, model_name: str, *, revision: str) -> None:
        if not revision.strip():
            raise ValueError("SentenceTransformer revision must be an exact non-empty revision.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the 'retrieval' extra before using sentence-transformers."
            ) from error
        self.name = f"sentence-transformers:{model_name}@{revision}"
        self._model = SentenceTransformer(model_name, revision=revision)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class DenseTypologyRetriever:
    """In-memory cosine retriever over a pluggable local encoder."""

    name = "dense"

    def __init__(
        self,
        documents: Sequence[TypologyDocument],
        encoder: TextEncoder,
        *,
        minimum_score: float = 1e-12,
    ) -> None:
        if not documents:
            raise ValueError("At least one Typology document is required.")
        self.documents = list(documents)
        self.encoder = encoder
        self.minimum_score = minimum_score
        self._document_vectors = _normalize_rows(
            encoder.encode([_document_text(doc) for doc in self.documents])
        )

    def search(self, query: str, *, limit: int = 3) -> list[ScoredTypologyDocument]:
        _validate_limit(limit)
        if not query.strip():
            return []
        query_vector = _normalize_rows(self.encoder.encode([query]))[0]
        scores = self._document_vectors @ query_vector
        ordered = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), self.documents[item[0]].typology_id),
        )
        retained = [item for item in ordered if float(item[1]) >= self.minimum_score][:limit]
        return [
            ScoredTypologyDocument(
                document=self.documents[index],
                score=float(score),
                rank=rank,
                retriever=f"dense:{self.encoder.name}",
            )
            for rank, (index, score) in enumerate(retained, start=1)
        ]

    def retrieve(self, query: str, *, limit: int = 3) -> list[TypologyDocument]:
        return [item.document for item in self.search(query, limit=limit)]


class LexicalOverlapReranker:
    """Transparent local reranker used as a deterministic baseline."""

    name = "lexical-overlap"

    def rerank(
        self,
        query: str,
        candidates: list[ScoredTypologyDocument],
        *,
        limit: int,
    ) -> list[ScoredTypologyDocument]:
        _validate_limit(limit)
        query_tokens = set(_tokens(query))

        def score(item: ScoredTypologyDocument) -> tuple[float, float, str]:
            document_tokens = set(_tokens(_document_text(item.document)))
            overlap = len(query_tokens.intersection(document_tokens)) / max(len(query_tokens), 1)
            return (-overlap, -item.score, item.document.typology_id)

        ordered = sorted(candidates, key=score)[:limit]
        return [
            ScoredTypologyDocument(
                document=item.document,
                score=item.score,
                rank=rank,
                retriever=f"{item.retriever}+{self.name}",
            )
            for rank, item in enumerate(ordered, start=1)
        ]


class HybridTypologyRetriever:
    """Reciprocal-rank fusion across local retrievers with optional reranking."""

    name = "hybrid-rrf"

    def __init__(
        self,
        retrievers: Sequence[TypologyScoredRetriever],
        *,
        reranker: TypologyReranker | None = None,
        rrf_constant: int = 60,
        fetch_limit: int = 20,
        required_source_prefixes: Sequence[str] = (),
    ) -> None:
        if not retrievers:
            raise ValueError("Hybrid retrieval requires at least one component.")
        if rrf_constant < 1 or fetch_limit < 1:
            raise ValueError("rrf_constant and fetch_limit must be positive.")
        self.retrievers = list(retrievers)
        self.reranker = reranker
        self.name = f"hybrid-rrf+{reranker.name}" if reranker is not None else "hybrid-rrf"
        self.rrf_constant = rrf_constant
        self.fetch_limit = fetch_limit
        self.required_source_prefixes = tuple(required_source_prefixes)

    def search(self, query: str, *, limit: int = 3) -> list[ScoredTypologyDocument]:
        _validate_limit(limit)
        fused: dict[str, tuple[TypologyDocument, float, list[str]]] = {}
        for retriever in self.retrievers:
            for item in retriever.search(query, limit=self.fetch_limit):
                document, score, sources = fused.get(
                    item.document.typology_id,
                    (item.document, 0.0, []),
                )
                fused[item.document.typology_id] = (
                    document,
                    score + 1.0 / (self.rrf_constant + item.rank),
                    [*sources, item.retriever],
                )
        eligible = [
            item
            for item in fused.values()
            if not self.required_source_prefixes
            or all(
                any(source.startswith(prefix) for source in item[2])
                for prefix in self.required_source_prefixes
            )
        ]
        ordered = sorted(eligible, key=lambda item: (-item[1], item[0].typology_id))
        candidates = [
            ScoredTypologyDocument(
                document=document,
                score=score,
                rank=rank,
                retriever=f"{self.name}({'+'.join(sources)})",
            )
            for rank, (document, score, sources) in enumerate(ordered, start=1)
        ]
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, limit=limit)
        return candidates[:limit]

    def retrieve(self, query: str, *, limit: int = 3) -> list[TypologyDocument]:
        return [item.document for item in self.search(query, limit=limit)]
