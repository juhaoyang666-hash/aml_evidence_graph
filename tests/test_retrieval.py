from __future__ import annotations

from aml_evidence_graph.evidence.typology import TypologyDocument
from aml_evidence_graph.retrieval import (
    BM25TypologyRetriever,
    DenseTypologyRetriever,
    HybridTypologyRetriever,
    LexicalOverlapReranker,
    RetrievalCase,
    TfidfDenseEncoder,
    evaluate_retriever,
)


def _documents() -> list[TypologyDocument]:
    return [
        TypologyDocument("structuring", "1", "Structuring", "split repeated payments", "test"),
        TypologyDocument("cycle", "1", "Cycle", "circular closed account flow", "test"),
        TypologyDocument("cash", "1", "Cash out", "rapid cash withdrawal", "test"),
    ]


def test_hybrid_retrieval_is_ranked_and_compatible() -> None:
    documents = _documents()
    corpus = [f"{document.title} {document.body}" for document in documents]
    bm25 = BM25TypologyRetriever(documents)
    dense = DenseTypologyRetriever(documents, TfidfDenseEncoder(corpus))
    hybrid = HybridTypologyRetriever(
        [bm25, dense],
        reranker=LexicalOverlapReranker(),
    )

    results = hybrid.search("repeated split payments", limit=2)

    assert results[0].document.typology_id == "structuring"
    assert [item.rank for item in results] == list(range(1, len(results) + 1))
    assert hybrid.retrieve("rapid cash withdrawal", limit=1)[0].typology_id == "cash"


def test_retrieval_evaluation_separates_no_answer_false_positives() -> None:
    retriever = BM25TypologyRetriever(_documents())
    cases = [
        RetrievalCase(
            case_id="answerable",
            query="circular closed flow",
            relevant_typology_ids=["cycle"],
        ),
        RetrievalCase(
            case_id="no-answer",
            query="weather restaurant",
            expect_no_answer=True,
        ),
    ]

    summary = evaluate_retriever(retriever, cases)

    assert summary.recall_at_1 == 1.0
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.no_answer_false_positive_rate == 0.0
