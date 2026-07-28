from __future__ import annotations

import json
from pathlib import Path

import pytest

from aml_evidence_graph.evidence.typology import (
    TypologyDocument,
    load_typology_documents,
)
from aml_evidence_graph.retrieval import (
    BM25TypologyRetriever,
    DenseTypologyRetriever,
    HybridTypologyRetriever,
    LexicalOverlapReranker,
    RetrievalCase,
    TfidfDenseEncoder,
    evaluate_retriever,
    evaluate_retriever_cases,
    summarize_retrieval_results,
)


def test_versioned_typology_corpus_and_golden_references_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = load_typology_documents(root / "knowledge" / "typologies")
    document_ids = [document.typology_id for document in documents]
    raw_cases: list[dict[str, object]] = []
    for name in ("retrieval_queries_v2.json", "retrieval_queries_v3_additions.json"):
        raw_cases.extend(
            json.loads((root / "golden" / name).read_text(encoding="utf-8"))
        )
    cases = [RetrievalCase.model_validate(item) for item in raw_cases]

    assert len(documents) == 11
    assert len(cases) == 130
    assert len(document_ids) == len(set(document_ids))
    assert len({case.case_id for case in cases}) == len(cases)
    assert {
        typology_id
        for case in cases
        for typology_id in case.relevant_typology_ids
    }.issubset(document_ids)
    official_documents = [
        document
        for document in documents
        if document.source.startswith(("FATF", "FinCEN"))
    ]
    assert len(official_documents) == 6
    assert all("https://" in document.source for document in official_documents)


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


def test_retrieval_case_results_support_tag_slices_and_bad_cases() -> None:
    retriever = BM25TypologyRetriever(_documents())
    cases = [
        RetrievalCase(
            case_id="direct",
            query="circular closed flow",
            relevant_typology_ids=["cycle"],
            tags=["direct"],
        ),
        RetrievalCase(
            case_id="hard-negative",
            query="cash flow statement tutorial",
            expect_no_answer=True,
            tags=["hard-negative", "no-answer"],
        ),
    ]

    results = evaluate_retriever_cases(retriever, cases)
    hard_negative = [result for result in results if "hard-negative" in result.tags]
    sliced = summarize_retrieval_results(retriever.name, hard_negative)

    assert results[0].passed
    assert not results[1].passed
    assert sliced.cases == 1
    assert sliced.no_answer_false_positive_rate == 1.0


def test_sentence_transformer_requires_exact_revision() -> None:
    from aml_evidence_graph.retrieval import SentenceTransformerEncoder

    with pytest.raises(ValueError, match="revision"):
        SentenceTransformerEncoder("unused-in-this-test", revision="")


def test_hybrid_can_require_dense_evidence_before_rrf() -> None:
    documents = _documents()
    corpus = [f"{document.title} {document.body}" for document in documents]
    bm25 = BM25TypologyRetriever(documents)
    dense = DenseTypologyRetriever(
        documents,
        TfidfDenseEncoder(corpus),
        minimum_score=0.9,
    )
    hybrid = HybridTypologyRetriever(
        [bm25, dense],
        required_source_prefixes=("dense:",),
    )

    assert bm25.search("cash flow statement", limit=3)
    assert not hybrid.search("cash flow statement", limit=3)
