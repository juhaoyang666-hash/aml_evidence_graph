from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from aml_evidence_graph.investigation.audit_store import (
    InvestigationAuditRecord,
    SQLiteInvestigationAuditStore,
)


def _record() -> InvestigationAuditRecord:
    return InvestigationAuditRecord(
        event_id="tool-event-1",
        thread_id="thread-1",
        alert_id="mock-alert-1",
        category="tool",
        name="get_feature_snapshot",
        status="complete",
        timestamp_utc=datetime(2026, 7, 28, tzinfo=UTC),
        duration_ms=2.5,
        argument_names=["alert_id"],
        output_keys=["features", "source_versions"],
    )


def test_sqlite_audit_store_is_durable_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "audit.sqlite"
    first = SQLiteInvestigationAuditStore(path)

    assert first.append_many([_record()]) == 1
    assert first.append_many([_record()]) == 0

    reopened = SQLiteInvestigationAuditStore(path)
    records = reopened.list_by_thread("thread-1")

    assert records == [_record()]


def test_sqlite_schema_cannot_store_evidence_or_note_bodies(tmp_path: Path) -> None:
    path = tmp_path / "audit.sqlite"
    store = SQLiteInvestigationAuditStore(path)
    store.append_many([_record()])

    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(investigation_audit_events)"
            ).fetchall()
        }
        serialized = json.dumps(
            connection.execute(
                "SELECT * FROM investigation_audit_events"
            ).fetchall(),
            ensure_ascii=False,
        )

    assert "evidence" not in columns
    assert "feature_values" not in columns
    assert "note" not in columns
    assert "transaction" not in serialized.lower()
