from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aml_evidence_graph.api.app import create_app
from aml_evidence_graph.evidence.typology import (
    LocalBM25TypologyRetriever,
    load_typology_documents,
)
from aml_evidence_graph.investigation.coordination import (
    LeaseLostError,
    SQLiteThreadLockRegistry,
)


def _retriever(tmp_path: Path) -> LocalBM25TypologyRetriever:
    typology_root = tmp_path / "typologies"
    typology_root.mkdir()
    (typology_root / "test.yaml").write_text(
        """
typology_id: "TYPOLOGY-TEST"
version: "1"
title: "Test Typology"
source: "Test"
body: "Transaction risk investigation."
""".strip(),
        encoding="utf-8",
    )
    return LocalBM25TypologyRetriever(load_typology_documents(typology_root))


def test_heartbeat_renewal_keeps_a_slow_holder_exclusive(tmp_path: Path) -> None:
    """A request slower than lease_seconds must not let a rival claim the thread."""
    path = tmp_path / "coordination.sqlite"
    holder = SQLiteThreadLockRegistry(
        path,
        acquire_timeout_seconds=1.0,
        lease_seconds=0.3,
        renew_interval_seconds=0.05,
    )
    rival = SQLiteThreadLockRegistry(
        path,
        acquire_timeout_seconds=0.2,
        lease_seconds=0.3,
        renew_interval_seconds=0.05,
    )
    entered = threading.Event()
    finish = threading.Event()
    failures: list[BaseException] = []

    def slow_holder() -> None:
        try:
            with holder.hold("thread-slow"):
                entered.set()
                finish.wait(timeout=5.0)
        except BaseException as error:  # noqa: BLE001 - recorded and re-asserted below
            failures.append(error)

    worker = threading.Thread(target=slow_holder, name="slow-holder")
    worker.start()
    try:
        assert entered.wait(timeout=5.0)
        # Three full lease periods; without renewal the lease would already be reaped.
        time.sleep(0.9)
        with pytest.raises(TimeoutError):
            with rival.hold("thread-slow"):
                pass
    finally:
        finish.set()
        worker.join(timeout=5.0)

    assert failures == []
    assert holder.active_lease_count() == 0


def test_expired_lease_from_a_crashed_worker_is_taken_over(tmp_path: Path) -> None:
    """A worker that dies mid-hold leaves a lease that must expire and be reclaimed."""
    path = tmp_path / "coordination.sqlite"
    crashed = SQLiteThreadLockRegistry(
        path,
        lease_seconds=0.2,
        renew_interval_seconds=0.05,
    )
    # A crash is exactly this: claimed, never renewed, never released.
    assert crashed._claim("thread-crashed", "dead-worker-token")
    assert crashed.active_lease_count() == 1

    survivor = SQLiteThreadLockRegistry(
        path,
        acquire_timeout_seconds=5.0,
        lease_seconds=0.5,
        renew_interval_seconds=0.1,
    )
    with survivor.hold("thread-crashed"):
        assert survivor.active_lease_count() == 1
    assert survivor.active_lease_count() == 0


def test_holder_that_lost_its_lease_raises_and_spares_the_new_owner(
    tmp_path: Path,
) -> None:
    """Fencing: a stale holder must report the loss and must not delete the new lease."""
    path = tmp_path / "coordination.sqlite"
    registry = SQLiteThreadLockRegistry(
        path,
        lease_seconds=0.2,
        renew_interval_seconds=0.05,
    )
    with pytest.raises(LeaseLostError, match="not provably exclusive"):
        with registry.hold("thread-stolen"):
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """
                    UPDATE investigation_thread_leases
                    SET owner_token = ?, expires_at = ?
                    WHERE thread_id = ?
                    """,
                    ("rival-worker-token", time.time() + 60.0, "thread-stolen"),
                )
                connection.commit()
            finally:
                connection.close()

    assert registry.active_lease_count() == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"renew_interval_seconds": 0.0},
        {"renew_interval_seconds": -1.0},
        {"lease_seconds": 1.0, "renew_interval_seconds": 1.0},
        {"lease_seconds": 1.0, "renew_interval_seconds": 2.0},
    ],
)
def test_invalid_renew_interval_is_rejected(
    tmp_path: Path,
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        SQLiteThreadLockRegistry(tmp_path / "coordination.sqlite", **overrides)


def test_default_renew_interval_is_shorter_than_the_lease(tmp_path: Path) -> None:
    registry = SQLiteThreadLockRegistry(tmp_path / "coordination.sqlite")
    assert 0 < registry.renew_interval_seconds < registry.lease_seconds


class _LostLeaseRegistry:
    """Stands in for a registry whose lease was taken over mid-request."""

    def hold(self, thread_id: str) -> None:
        raise LeaseLostError(f"simulated takeover of {thread_id!r}")


def test_lost_lease_is_reported_as_retryable_503(tmp_path: Path) -> None:
    client = TestClient(
        create_app(_retriever(tmp_path), thread_lock_registry=_LostLeaseRegistry())
    )
    response = client.post(
        "/v1/controlled-investigations/mock-alert-0001",
        json={"thread_id": "thread-lost"},
    )
    assert response.status_code == 503
    assert "simulated takeover" in response.json()["detail"]
