#!/usr/bin/env python3
"""Validate shared-store multi-worker evidence and exactly-once probe artifacts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/multiworker_shared_v1"),
    )
    return parser.parse_args()


def _scalar(path: Path, query: str) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute(query).fetchone()
    return int(row[0])


def main() -> None:
    args = parse_args()
    evidence = json.loads((args.root / "evidence_probe.json").read_text(encoding="utf-8"))
    benchmark = json.loads(
        (args.root / "agent_benchmark" / "metrics.json").read_text(encoding="utf-8")
    )
    requests = int(benchmark["summary"]["requests"])
    duplicate_reviews = int(benchmark["duplicate_reviews_per_thread"])
    review_events = _scalar(
        args.root / "audit.sqlite",
        "SELECT COUNT(*) FROM investigation_audit_events WHERE category = 'review'",
    )
    finalize_events = _scalar(
        args.root / "audit.sqlite",
        "SELECT COUNT(*) FROM investigation_audit_events WHERE name = 'finalize'",
    )
    active_leases = _scalar(
        args.root / "coordination.sqlite",
        "SELECT COUNT(*) FROM investigation_thread_leases",
    )
    checks = {
        "all_evidence_reads_visible": evidence["status_counts"] == {"200": evidence["attempts"]},
        "zero_agent_cycle_errors": benchmark["summary"]["errors"] == 0,
        "one_review_execution_per_thread": benchmark["review_execution_count"] == requests,
        "all_duplicate_reviews_replayed": benchmark["idempotent_replay_count"]
        == requests * (duplicate_reviews - 1),
        "one_review_audit_event_per_thread": review_events == requests,
        "one_finalize_audit_event_per_thread": finalize_events == requests,
        "all_thread_leases_released": active_leases == 0,
    }
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "local two-worker shared SQLite engineering validation; not production SLA",
        "evidence_attempts": evidence["attempts"],
        "evidence_status_counts": evidence["status_counts"],
        "investigation_threads": requests,
        "review_requests": benchmark["review_request_count"],
        "review_executions": benchmark["review_execution_count"],
        "idempotent_replays": benchmark["idempotent_replay_count"],
        "review_audit_events": review_events,
        "finalize_audit_events": finalize_events,
        "active_thread_leases": active_leases,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = args.root / "validation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
