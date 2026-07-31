"""Opaque-reference service contracts for AML API endpoints."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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


class SQLiteEvidenceStore:
    """Shared local evidence storage for multi-worker engineering validation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_evidence_packages (
                    alert_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def get(self, alert_id: str) -> RiskEvidencePackage | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM risk_evidence_packages WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()
        if row is None:
            return None
        return RiskEvidencePackage.model_validate_json(row["payload_json"])

    def put(self, evidence: RiskEvidencePackage) -> None:
        payload = json.dumps(
            evidence.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO risk_evidence_packages (
                    alert_id, schema_version, payload_json, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    evidence.alert_id,
                    evidence.schema_version,
                    payload,
                    datetime.now(UTC).isoformat(),
                ),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT payload_json FROM risk_evidence_packages WHERE alert_id = ?",
                    (evidence.alert_id,),
                ).fetchone()
                if row is None or row["payload_json"] != payload:
                    raise ValueError(
                        "Evidence alert_id already exists with a different immutable payload."
                    )


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
