"""Independent, value-minimized audit storage for controlled investigations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class InvestigationAuditRecord(BaseModel):
    """Append-only metadata; evidence bodies and feature values are deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=128)
    alert_id: str = Field(min_length=1, max_length=256)
    category: Literal["tool", "node", "review"]
    name: str = Field(min_length=1, max_length=128)
    status: Literal["complete", "failed"]
    timestamp_utc: datetime
    duration_ms: float | None = Field(default=None, ge=0)
    argument_names: list[str] = Field(default_factory=list)
    output_keys: list[str] = Field(default_factory=list)
    state_change: str | None = Field(default=None, max_length=128)
    error_code: str | None = Field(default=None, max_length=128)
    review_action: Literal["approve", "edit", "reject"] | None = None
    reviewer_reference: str | None = Field(default=None, max_length=128)
    note_present: bool | None = None


class InvestigationAuditStore(Protocol):
    """Storage contract kept separate from LangGraph checkpoint persistence."""

    def append_many(self, records: Iterable[InvestigationAuditRecord]) -> int: ...

    def list_by_thread(self, thread_id: str) -> list[InvestigationAuditRecord]: ...


class InMemoryInvestigationAuditStore:
    """Test/demo implementation with the same event-id idempotency as SQLite."""

    def __init__(self) -> None:
        self._records: dict[str, InvestigationAuditRecord] = {}

    def append_many(self, records: Iterable[InvestigationAuditRecord]) -> int:
        inserted = 0
        for record in records:
            if record.event_id not in self._records:
                self._records[record.event_id] = record
                inserted += 1
        return inserted

    def list_by_thread(self, thread_id: str) -> list[InvestigationAuditRecord]:
        return sorted(
            (record for record in self._records.values() if record.thread_id == thread_id),
            key=lambda record: (record.timestamp_utc, record.event_id),
        )


class SQLiteInvestigationAuditStore:
    """Local durable prototype using append-only, idempotent SQLite inserts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_audit_events (
                    event_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    alert_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    duration_ms REAL,
                    argument_names_json TEXT NOT NULL,
                    output_keys_json TEXT NOT NULL,
                    state_change TEXT,
                    error_code TEXT,
                    review_action TEXT,
                    reviewer_reference TEXT,
                    note_present INTEGER
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_investigation_audit_thread_time
                ON investigation_audit_events(thread_id, timestamp_utc, event_id)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def append_many(self, records: Iterable[InvestigationAuditRecord]) -> int:
        payload = [self._to_row(record) for record in records]
        if not payload:
            return 0
        with self._connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO investigation_audit_events (
                    event_id, thread_id, alert_id, category, name, status,
                    timestamp_utc, duration_ms, argument_names_json, output_keys_json,
                    state_change, error_code, review_action, reviewer_reference, note_present
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            return connection.total_changes - before

    def list_by_thread(self, thread_id: str) -> list[InvestigationAuditRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM investigation_audit_events
                WHERE thread_id = ?
                ORDER BY timestamp_utc, event_id
                """,
                (thread_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _to_row(record: InvestigationAuditRecord) -> tuple[object, ...]:
        return (
            record.event_id,
            record.thread_id,
            record.alert_id,
            record.category,
            record.name,
            record.status,
            record.timestamp_utc.astimezone(UTC).isoformat(),
            record.duration_ms,
            json.dumps(record.argument_names, ensure_ascii=True, separators=(",", ":")),
            json.dumps(record.output_keys, ensure_ascii=True, separators=(",", ":")),
            record.state_change,
            record.error_code,
            record.review_action,
            record.reviewer_reference,
            None if record.note_present is None else int(record.note_present),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> InvestigationAuditRecord:
        return InvestigationAuditRecord(
            event_id=row["event_id"],
            thread_id=row["thread_id"],
            alert_id=row["alert_id"],
            category=row["category"],
            name=row["name"],
            status=row["status"],
            timestamp_utc=row["timestamp_utc"],
            duration_ms=row["duration_ms"],
            argument_names=json.loads(row["argument_names_json"]),
            output_keys=json.loads(row["output_keys_json"]),
            state_change=row["state_change"],
            error_code=row["error_code"],
            review_action=row["review_action"],
            reviewer_reference=row["reviewer_reference"],
            note_present=(
                None if row["note_present"] is None else bool(row["note_present"])
            ),
        )


def _stable_event_id(prefix: str, payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}-{digest}"


def audit_records_from_state(
    thread_id: str,
    state: dict[str, object],
) -> list[InvestigationAuditRecord]:
    """Project checkpoint metadata into a separate, value-minimized audit stream."""
    evidence = state.get("evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("alert_id"), str):
        raise ValueError("Controlled state is missing its alert_id.")
    alert_id = evidence["alert_id"]
    records: list[InvestigationAuditRecord] = []

    for raw_event in state.get("audit_events", []):
        if not isinstance(raw_event, dict):
            continue
        records.append(
            InvestigationAuditRecord(
                event_id=str(raw_event["event_id"]),
                thread_id=thread_id,
                alert_id=alert_id,
                category="tool",
                name=str(raw_event["tool_name"]),
                status=str(raw_event["status"]),
                timestamp_utc=raw_event["timestamp_utc"],
                duration_ms=raw_event.get("duration_ms"),
                argument_names=list(raw_event.get("argument_names", [])),
                output_keys=list(raw_event.get("output_keys", [])),
                error_code=raw_event.get("error_code"),
            )
        )

    for raw_event in state.get("node_timeline", []):
        if not isinstance(raw_event, dict):
            continue
        identity = {
            "thread_id": thread_id,
            "node": raw_event.get("node"),
            "timestamp_utc": raw_event.get("timestamp_utc"),
            "state_change": raw_event.get("state_change"),
        }
        records.append(
            InvestigationAuditRecord(
                event_id=_stable_event_id("node", identity),
                thread_id=thread_id,
                alert_id=alert_id,
                category="node",
                name=str(raw_event["node"]),
                status=str(raw_event["status"]),
                timestamp_utc=raw_event["timestamp_utc"],
                duration_ms=raw_event.get("duration_ms"),
                state_change=str(raw_event["state_change"]),
            )
        )

    decision = state.get("review_decision")
    if isinstance(decision, dict):
        review_timeline = next(
            (
                event
                for event in reversed(state.get("node_timeline", []))
                if isinstance(event, dict) and event.get("node") == "human_review"
            ),
            None,
        )
        timestamp = (
            review_timeline.get("timestamp_utc")
            if isinstance(review_timeline, dict)
            else datetime.now(UTC).isoformat()
        )
        identity = {
            "thread_id": thread_id,
            "timestamp_utc": timestamp,
            "action": decision.get("action"),
        }
        records.append(
            InvestigationAuditRecord(
                event_id=_stable_event_id("review", identity),
                thread_id=thread_id,
                alert_id=alert_id,
                category="review",
                name="human_review_decision",
                status="complete",
                timestamp_utc=timestamp,
                review_action=decision.get("action"),
                reviewer_reference=decision.get("reviewer_reference"),
                note_present=bool(decision.get("note")),
            )
        )
    return records


def persist_state_audit(
    store: InvestigationAuditStore,
    *,
    thread_id: str,
    state: dict[str, object],
) -> int:
    """Persist all visible state events; event IDs make repeated snapshots idempotent."""
    return store.append_many(audit_records_from_state(thread_id, state))
