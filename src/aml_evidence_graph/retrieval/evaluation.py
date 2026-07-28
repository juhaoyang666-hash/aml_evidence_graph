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


def evaluate_retriever(
    retriever: TypologyScoredRetriever,
    cases: list[RetrievalCase],
    *,
    limit: int = 3,
) -> RetrievalEvaluationSummary:
    """Evaluate ranking separately from report generation or case conclusions."""
    if not cases:
        raise ValueError("At least one retrieval case is required.")
    answerable = [case for case in cases if not case.expect_no_answer]
    no_answer = [case for case in cases if case.expect_no_answer]
    recall_1: list[float] = []
    recall_3: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    false_positives = 0
    for case in cases:
        results = retriever.search(case.query, limit=limit)
        ids = [result.document.typology_id for result in results]
        if case.expect_no_answer:
            false_positives += bool(ids)
            continue
        relevant = set(case.relevant_typology_ids)
        if not relevant:
            raise ValueError(f"Answerable case {case.case_id} has no relevant ids.")
        recall_1.append(_recall_at(ids, relevant, 1))
        recall_3.append(_recall_at(ids, relevant, 3))
        reciprocal_ranks.append(_reciprocal_rank(ids, relevant))
        ndcgs.append(_ndcg_at(ids, relevant, 3))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return RetrievalEvaluationSummary(
        retriever=retriever.name,
        cases=len(cases),
        answerable_cases=len(answerable),
        no_answer_cases=len(no_answer),
        recall_at_1=mean(recall_1),
        recall_at_3=mean(recall_3),
        mean_reciprocal_rank=mean(reciprocal_ranks),
        ndcg_at_3=mean(ndcgs),
        no_answer_false_positive_rate=(
            false_positives / len(no_answer) if no_answer else 0.0
        ),
    )
