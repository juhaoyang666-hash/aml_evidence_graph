"""Retrieval-only metrics kept separate from LLM generation evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from aml_evidence_graph.retrieval.contracts import TypologyScoredRetriever


class RetrievalCase(BaseModel):
    """A versioned query judgment for local Typology retrieval."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    query: str
    relevant_typology_ids: list[str] = Field(default_factory=list)
    expect_no_answer: bool = False
    tags: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """Aggregate ranking and no-answer metrics for one retriever version."""

    retriever: str
    cases: int
    answerable_cases: int
    no_answer_cases: int
    recall_at_1: float
    recall_at_3: float
    mean_reciprocal_rank: float
    ndcg_at_3: float
    no_answer_false_positive_rate: float


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Auditable per-query outcome used for tag slices and Bad Case review."""

    case_id: str
    tags: tuple[str, ...]
    expect_no_answer: bool
    relevant_typology_ids: tuple[str, ...]
    retrieved_typology_ids: tuple[str, ...]
    recall_at_1: float | None
    recall_at_3: float | None
    reciprocal_rank: float | None
    ndcg_at_3: float | None
    no_answer_false_positive: bool

    @property
    def passed(self) -> bool:
        if self.expect_no_answer:
            return not self.no_answer_false_positive
        return self.recall_at_3 == 1.0


def _recall_at(ids: list[str], relevant: set[str], k: int) -> float:
    return len(set(ids[:k]).intersection(relevant)) / len(relevant) if relevant else 0.0


def _reciprocal_rank(ids: list[str], relevant: set[str]) -> float:
    for rank, typology_id in enumerate(ids, start=1):
        if typology_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at(ids: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if typology_id in relevant else 0.0 for typology_id in ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def evaluate_retriever_cases(
    retriever: TypologyScoredRetriever,
    cases: list[RetrievalCase],
    *,
    limit: int = 3,
) -> list[RetrievalCaseResult]:
    """Return per-case judgments without report generation or case conclusions."""
    if not cases:
        raise ValueError("At least one retrieval case is required.")
    results: list[RetrievalCaseResult] = []
    for case in cases:
        retrieved = retriever.search(case.query, limit=limit)
        ids = [result.document.typology_id for result in retrieved]
        if case.expect_no_answer:
            results.append(
                RetrievalCaseResult(
                    case_id=case.case_id,
                    tags=tuple(case.tags),
                    expect_no_answer=True,
                    relevant_typology_ids=(),
                    retrieved_typology_ids=tuple(ids),
                    recall_at_1=None,
                    recall_at_3=None,
                    reciprocal_rank=None,
                    ndcg_at_3=None,
                    no_answer_false_positive=bool(ids),
                )
            )
            continue
        relevant = set(case.relevant_typology_ids)
        if not relevant:
            raise ValueError(f"Answerable case {case.case_id} has no relevant ids.")
        results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                tags=tuple(case.tags),
                expect_no_answer=False,
                relevant_typology_ids=tuple(sorted(relevant)),
                retrieved_typology_ids=tuple(ids),
                recall_at_1=_recall_at(ids, relevant, 1),
                recall_at_3=_recall_at(ids, relevant, 3),
                reciprocal_rank=_reciprocal_rank(ids, relevant),
                ndcg_at_3=_ndcg_at(ids, relevant, 3),
                no_answer_false_positive=False,
            )
        )
    return results


def summarize_retrieval_results(
    retriever_name: str,
    results: list[RetrievalCaseResult],
) -> RetrievalEvaluationSummary:
    """Aggregate a complete set or a tag slice of per-query outcomes."""
    if not results:
        raise ValueError("At least one retrieval result is required.")
    answerable = [result for result in results if not result.expect_no_answer]
    no_answer = [result for result in results if result.expect_no_answer]

    def mean(values: list[float | None]) -> float:
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else 0.0

    return RetrievalEvaluationSummary(
        retriever=retriever_name,
        cases=len(results),
        answerable_cases=len(answerable),
        no_answer_cases=len(no_answer),
        recall_at_1=mean([result.recall_at_1 for result in answerable]),
        recall_at_3=mean([result.recall_at_3 for result in answerable]),
        mean_reciprocal_rank=mean([result.reciprocal_rank for result in answerable]),
        ndcg_at_3=mean([result.ndcg_at_3 for result in answerable]),
        no_answer_false_positive_rate=(
            sum(result.no_answer_false_positive for result in no_answer) / len(no_answer)
            if no_answer
            else 0.0
        ),
    )


def evaluate_retriever(
    retriever: TypologyScoredRetriever,
    cases: list[RetrievalCase],
    *,
    limit: int = 3,
) -> RetrievalEvaluationSummary:
    """Evaluate ranking separately from report generation or case conclusions."""
    return summarize_retrieval_results(
        retriever.name,
        evaluate_retriever_cases(retriever, cases, limit=limit),
    )
