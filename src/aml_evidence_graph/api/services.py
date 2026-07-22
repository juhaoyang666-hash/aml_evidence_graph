"""Opaque-reference service contracts for AML API endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from aml_evidence_graph.evidence.package import RiskEvidencePackage


class EvidenceStore(Protocol):
    """Private storage keyed by opaque alert IDs."""

    def get(self, alert_id: str) -> RiskEvidencePackage | None: ...

    def put(self, evidence: RiskEvidencePackage) -> None: ...


@dataclass
class InMemoryEvidenceStore:
    """Local demo/testing store; production adapters must enforce access control."""

    _items: dict[str, RiskEvidencePackage]

    def __init__(self) -> None:
        self._items = {}

    def get(self, alert_id: str) -> RiskEvidencePackage | None:
        return self._items.get(alert_id)

    def put(self, evidence: RiskEvidencePackage) -> None:
        self._items[evidence.alert_id] = evidence


@dataclass(frozen=True)
class ScoreBatchResult:
    """Safe aggregate response for an internal controlled-partition scoring operation."""

    partition_ref: str
    alert_ids: list[str]
    model_version: str


class PartitionScoringService(Protocol):
    """Score a pre-authorized private partition without accepting transaction payloads."""

    def score_partition(self, partition_ref: str) -> ScoreBatchResult: ...


ReviewDecision = Literal["confirmed", "dismissed", "needs_more_evidence"]


@dataclass(frozen=True)
class HumanReviewRecord:
    """Immutable investigator feedback; it never triggers online model training."""

    review_id: str
    alert_id: str
    reviewer_reference: str
    decision: ReviewDecision
    note: str | None
    submitted_at: datetime


class ReviewStore(Protocol):
    """Private audit store for human review outcomes."""

    def append(self, record: HumanReviewRecord) -> HumanReviewRecord: ...


@dataclass
class InMemoryReviewStore:
    """Local/testing review store; a production adapter must enforce retention rules."""

    _items: list[HumanReviewRecord]

    def __init__(self) -> None:
        self._items = []

    def append(self, record: HumanReviewRecord) -> HumanReviewRecord:
        self._items.append(record)
        return record


class MockPartitionScoringService:
    """Demo-only partition scorer that registers fictional evidence."""

    def __init__(self, store: EvidenceStore, mock_evidence: RiskEvidencePackage) -> None:
        self.store = store
        self.mock_evidence = mock_evidence

    def score_partition(self, partition_ref: str) -> ScoreBatchResult:
        if partition_ref != "mock-partition":
            raise ValueError("Demo service only recognizes mock-partition.")
        self.store.put(self.mock_evidence)
        return ScoreBatchResult(
            partition_ref=partition_ref,
            alert_ids=[self.mock_evidence.alert_id],
            model_version="mock-model-v1",
        )
