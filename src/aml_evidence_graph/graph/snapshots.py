"""Leakage-safe daily graph snapshots with historical-neighbor-only edges."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aml_evidence_graph.data.contract import CANONICAL


def graph_population_audit(
    reference_transactions: pd.DataFrame,
    scored_transactions: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize account overlap and cold starts without emitting identifiers."""
    required = {CANONICAL.sender_account_id, CANONICAL.receiver_account_id}
    for name, frame in (
        ("reference", reference_transactions),
        ("scored", scored_transactions),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} transactions are missing: " + ", ".join(missing))
    reference_accounts = set(
        pd.concat(
            [
                reference_transactions[CANONICAL.sender_account_id],
                reference_transactions[CANONICAL.receiver_account_id],
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
    )
    sender_accounts = scored_transactions[CANONICAL.sender_account_id].astype(str)
    receiver_accounts = scored_transactions[CANONICAL.receiver_account_id].astype(str)
    scored_accounts = set(sender_accounts).union(receiver_accounts)
    shared_accounts = scored_accounts.intersection(reference_accounts)
    sender_cold = ~sender_accounts.isin(reference_accounts)
    receiver_cold = ~receiver_accounts.isin(reference_accounts)
    scored_account_count = len(scored_accounts)
    return {
        "reference_account_count": len(reference_accounts),
        "scored_transaction_count": len(scored_transactions),
        "scored_account_count": scored_account_count,
        "shared_account_count": len(shared_accounts),
        "account_overlap_rate": len(shared_accounts) / scored_account_count
        if scored_account_count
        else 0.0,
        "cold_start_account_count": len(scored_accounts.difference(reference_accounts)),
        "cold_start_account_rate": len(scored_accounts.difference(reference_accounts))
        / scored_account_count
        if scored_account_count
        else 0.0,
        "sender_cold_start_transaction_count": int(sender_cold.sum()),
        "receiver_cold_start_transaction_count": int(receiver_cold.sum()),
        "either_endpoint_cold_start_transaction_count": int((sender_cold | receiver_cold).sum()),
        "both_endpoints_seen_transaction_count": int((~sender_cold & ~receiver_cold).sum()),
    }


@dataclass(frozen=True)
class TemporalGraphSnapshot:
    """One scoring day and its strictly earlier directed transaction graph."""

    event_date: str
    history_edge_index: np.ndarray
    scoring_edge_index: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray
    transaction_ids: tuple[str, ...]
    history_edge_type: np.ndarray | None = None


def relation_id_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Map cross-border × currency-conversion into {0,1,2,3} (train-period vocabulary)."""
    cross = (
        frame["is_cross_border_current_transaction"].to_numpy(dtype=np.float32)
        if "is_cross_border_current_transaction" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    convert = (
        frame["is_currency_conversion"].to_numpy(dtype=np.float32)
        if "is_currency_conversion" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    return (2 * (cross > 0.5).astype(np.int64) + (convert > 0.5).astype(np.int64)).astype(
        np.int64
    )


class TemporalNodeIndexer:
    """Fit known account nodes on training data and hash unseen accounts into buckets."""

    def __init__(self, *, unknown_hash_buckets: int = 65_536) -> None:
        if unknown_hash_buckets < 1:
            raise ValueError("unknown_hash_buckets must be positive.")
        self.unknown_hash_buckets = unknown_hash_buckets
        self._known_nodes: dict[str, int] = {}

    @property
    def num_nodes(self) -> int:
        return 1 + len(self._known_nodes) + self.unknown_hash_buckets

    def fit(self, transactions: pd.DataFrame) -> TemporalNodeIndexer:
        required = {
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
        }
        missing = sorted(required.difference(transactions.columns))
        if missing:
            raise ValueError(f"Node index fit requires columns: {', '.join(missing)}")
        identifiers = pd.concat(
            [
                transactions[CANONICAL.sender_account_id],
                transactions[CANONICAL.receiver_account_id],
            ],
            ignore_index=True,
        ).astype("string")
        self._known_nodes = {
            value: offset
            for offset, value in enumerate(
                sorted(identifiers.dropna().unique().tolist()),
                start=1,
            )
        }
        return self

    def transform_identifier(self, identifier: str) -> int:
        known = self._known_nodes.get(identifier)
        if known is not None:
            return known
        digest = hashlib.blake2b(identifier.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % self.unknown_hash_buckets
        return 1 + len(self._known_nodes) + bucket

    def transform(self, identifiers: Iterable[str]) -> np.ndarray:
        return np.asarray(
            [self.transform_identifier(str(identifier)) for identifier in identifiers],
            dtype=np.int64,
        )


class DailyGraphSnapshotBuilder:
    """Construct daily scores against a bounded graph containing earlier days only."""

    def __init__(
        self,
        node_indexer: TemporalNodeIndexer,
        *,
        edge_feature_columns: tuple[str, ...],
        history_window: pd.Timedelta | None = None,
        store_relation_types: bool = False,
    ) -> None:
        history_window = history_window or pd.Timedelta(days=30)
        if history_window <= pd.Timedelta(0):
            raise ValueError("history_window must be positive.")
        self.node_indexer = node_indexer
        self.edge_feature_columns = edge_feature_columns
        self.history_window = history_window
        self.store_relation_types = store_relation_types
        # (event_ts, sender, receiver[, relation_id])
        self._history_edges: deque[tuple] = deque()
        self._last_processed_date: pd.Timestamp | None = None

    def build(
        self,
        transactions: pd.DataFrame,
        *,
        include_labels: bool = True,
    ) -> list[TemporalGraphSnapshot]:
        """Build snapshots while retaining history from prior calls and periods.

        In scoring mode, labels are deliberately not read from the input. The
        zero placeholders preserve the common snapshot shape while ensuring
        that inference cannot consume future outcomes.
        """
        required = {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.source_row_number,
            *self.edge_feature_columns,
        }
        if include_labels:
            required.add(CANONICAL.is_laundering)
        missing = sorted(required.difference(transactions.columns))
        if missing:
            raise ValueError(f"Graph snapshot input is missing columns: {', '.join(missing)}")
        ordered = transactions.sort_values(
            [CANONICAL.event_ts, CANONICAL.source_row_number],
            kind="stable",
        ).copy()
        ordered[CANONICAL.event_ts] = pd.to_datetime(
            ordered[CANONICAL.event_ts],
            utc=True,
            errors="raise",
        )
        ordered["_graph_event_date"] = ordered[CANONICAL.event_ts].dt.normalize()

        snapshots: list[TemporalGraphSnapshot] = []
        for event_day, current in ordered.groupby("_graph_event_date", sort=False):
            if self._last_processed_date is not None and event_day <= self._last_processed_date:
                raise ValueError(
                    "Graph snapshots must be built in strictly increasing event dates."
                )
            cutoff = event_day - self.history_window
            while self._history_edges and self._history_edges[0][0] < cutoff:
                self._history_edges.popleft()

            sender_nodes = self.node_indexer.transform(current[CANONICAL.sender_account_id])
            receiver_nodes = self.node_indexer.transform(current[CANONICAL.receiver_account_id])
            scoring_edges = np.vstack([sender_nodes, receiver_nodes])
            relation_ids = (
                relation_id_from_frame(current)
                if self.store_relation_types
                else np.zeros(len(current), dtype=np.int64)
            )
            if self._history_edges:
                if self.store_relation_types:
                    history_edges = np.asarray(
                        [(sender, receiver) for _, sender, receiver, _ in self._history_edges],
                        dtype=np.int64,
                    ).T
                    history_types = np.asarray(
                        [relation for _, _, _, relation in self._history_edges],
                        dtype=np.int64,
                    )
                else:
                    history_edges = np.asarray(
                        [(sender, receiver) for _, sender, receiver in self._history_edges],
                        dtype=np.int64,
                    ).T
                    history_types = None
            else:
                history_edges = np.empty((2, 0), dtype=np.int64)
                history_types = (
                    np.empty(0, dtype=np.int64) if self.store_relation_types else None
                )
            edge_features = (
                current.loc[:, self.edge_feature_columns]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=np.float32)
            )
            snapshots.append(
                TemporalGraphSnapshot(
                    event_date=event_day.date().isoformat(),
                    history_edge_index=history_edges,
                    scoring_edge_index=scoring_edges,
                    edge_features=edge_features,
                    labels=(
                        current[CANONICAL.is_laundering].to_numpy(dtype=np.int64)
                        if include_labels
                        else np.zeros(len(current), dtype=np.int64)
                    ),
                    transaction_ids=tuple(
                        current[CANONICAL.transaction_id].astype(str).tolist()
                    ),
                    history_edge_type=history_types,
                )
            )
            if self.store_relation_types:
                for event_ts, sender, receiver, relation in zip(
                    current[CANONICAL.event_ts],
                    sender_nodes,
                    receiver_nodes,
                    relation_ids,
                    strict=True,
                ):
                    self._history_edges.append(
                        (event_ts, int(sender), int(receiver), int(relation))
                    )
            else:
                for event_ts, sender, receiver in zip(
                    current[CANONICAL.event_ts],
                    sender_nodes,
                    receiver_nodes,
                    strict=True,
                ):
                    self._history_edges.append((event_ts, int(sender), int(receiver)))
            self._last_processed_date = event_day
        return snapshots


def fit_edge_feature_scaler(snapshots: Iterable[TemporalGraphSnapshot]) -> StandardScaler:
    """Fit edge scaling on training snapshots only without concatenating all rows."""
    scaler = StandardScaler()
    seen_rows = 0
    for snapshot in snapshots:
        if len(snapshot.edge_features):
            scaler.partial_fit(snapshot.edge_features)
            seen_rows += len(snapshot.edge_features)
    if seen_rows == 0:
        raise ValueError("At least one training graph edge is required to fit a scaler.")
    return scaler


def transform_edge_features(
    snapshots: Iterable[TemporalGraphSnapshot],
    scaler: StandardScaler,
) -> list[TemporalGraphSnapshot]:
    """Scale a snapshot collection using an already fitted training-only scaler."""
    transformed: list[TemporalGraphSnapshot] = []
    for snapshot in snapshots:
        transformed.append(
            TemporalGraphSnapshot(
                event_date=snapshot.event_date,
                history_edge_index=snapshot.history_edge_index,
                scoring_edge_index=snapshot.scoring_edge_index,
                edge_features=scaler.transform(snapshot.edge_features).astype(
                    np.float32
                ),
                labels=snapshot.labels,
                transaction_ids=snapshot.transaction_ids,
                history_edge_type=snapshot.history_edge_type,
            )
        )
    return transformed
