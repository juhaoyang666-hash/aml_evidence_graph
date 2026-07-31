"""Cross-process mutation coordination for controlled investigation threads."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LeaseLostError(RuntimeError):
    """Raised when a holder can no longer prove it owned the lease throughout.

    A lost lease means the critical section was not provably exclusive, so the
    caller must treat the mutation as unconfirmed rather than successful.
    """


class SQLiteThreadLockRegistry:
    """A small SQLite lease lock that serializes one thread across API workers.

    SQLite transactions are held only while claiming, renewing, or releasing a
    lease. Different thread IDs can therefore progress concurrently. The durable
    LangGraph checkpoint remains the source of truth after a worker crash.

    A background heartbeat renews the lease while it is held, so a slow request
    cannot silently let the lease expire under a second worker. If a renewal or
    the final release cannot confirm ownership, ``hold`` raises
    :class:`LeaseLostError` instead of reporting a clean exit.
    """

    def __init__(
        self,
        path: Path,
        *,
        acquire_timeout_seconds: float = 30.0,
        lease_seconds: float = 600.0,
        poll_interval_seconds: float = 0.02,
        renew_interval_seconds: float | None = None,
    ) -> None:
        if acquire_timeout_seconds <= 0 or lease_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("SQLite thread-lock timing values must be positive.")
        renew_interval = (
            lease_seconds / 3.0 if renew_interval_seconds is None else renew_interval_seconds
        )
        if renew_interval <= 0:
            raise ValueError("renew_interval_seconds must be positive.")
        if renew_interval >= lease_seconds:
            raise ValueError("renew_interval_seconds must be shorter than lease_seconds.")
        self.path = path
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.lease_seconds = lease_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.renew_interval_seconds = renew_interval
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_thread_leases (
                    thread_id TEXT PRIMARY KEY,
                    owner_token TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=max(self.acquire_timeout_seconds, 1.0),
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _claim(self, thread_id: str, owner_token: str) -> bool:
        now = time.time()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM investigation_thread_leases WHERE expires_at <= ?",
                (now,),
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO investigation_thread_leases (
                    thread_id, owner_token, acquired_at, expires_at
                ) VALUES (?, ?, ?, ?)
                """,
                (thread_id, owner_token, now, now + self.lease_seconds),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def _renew(self, thread_id: str, owner_token: str) -> bool:
        """Extend an owned, still-valid lease. An expired lease is never resurrected."""
        now = time.time()
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                UPDATE investigation_thread_leases
                SET expires_at = ?
                WHERE thread_id = ? AND owner_token = ? AND expires_at > ?
                """,
                (now + self.lease_seconds, thread_id, owner_token, now),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def _release(self, thread_id: str, owner_token: str) -> bool:
        """Drop only our own lease; return whether we still owned it."""
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                DELETE FROM investigation_thread_leases
                WHERE thread_id = ? AND owner_token = ?
                """,
                (thread_id, owner_token),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def _heartbeat(
        self,
        thread_id: str,
        owner_token: str,
        stop: threading.Event,
        lost: threading.Event,
    ) -> None:
        while not stop.wait(self.renew_interval_seconds):
            if not self._renew(thread_id, owner_token):
                lost.set()
                return

    def active_lease_count(self) -> int:
        """Count leases currently recorded, including any not yet reaped."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) FROM investigation_thread_leases"
            ).fetchone()
        finally:
            connection.close()
        return int(row[0])

    @contextmanager
    def hold(self, thread_id: str) -> Iterator[None]:
        if not thread_id.strip():
            raise ValueError("thread_id must be non-empty.")
        owner_token = uuid.uuid4().hex
        deadline = time.monotonic() + self.acquire_timeout_seconds
        while not self._claim(thread_id, owner_token):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring shared lock for thread {thread_id!r}.")
            time.sleep(self.poll_interval_seconds)

        stop = threading.Event()
        lost = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(thread_id, owner_token, stop, lost),
            name=f"lease-heartbeat-{thread_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            yield
        finally:
            stop.set()
            heartbeat.join(timeout=10.0)
            if not self._release(thread_id, owner_token):
                lost.set()
        if lost.is_set():
            raise LeaseLostError(
                f"Lease for thread {thread_id!r} expired or was taken over while held; "
                "the mutation is not provably exclusive."
            )
