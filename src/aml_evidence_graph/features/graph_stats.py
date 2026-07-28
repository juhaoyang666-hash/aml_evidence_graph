"""Causal directed-graph statistics for transaction edge classification."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime

import polars as pl

from aml_evidence_graph.compat import to_polars
from aml_evidence_graph.data.contract import CANONICAL


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    parsed = pl.Series([value]).cast(pl.Datetime(time_zone="UTC"), strict=True)[0]
    assert isinstance(parsed, datetime)
    return parsed


class CausalGraphStatisticsBuilder:
    """Maintain directed graph structure using only strictly earlier transactions."""

    def __init__(self) -> None:
        self._outgoing_neighbors: dict[str, set[str]] = defaultdict(set)
        self._incoming_neighbors: dict[str, set[str]] = defaultdict(set)
        self._edge_counts: Counter[tuple[str, str]] = Counter()
        self._last_processed_ts: datetime | None = None

    def transform_partition(self, transactions: pl.DataFrame | object) -> pl.DataFrame:
        """Return graph features for a complete chronological event-date partition."""
        frame = to_polars(transactions)
        required = {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.source_row_number,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Transactions are missing required graph-stat columns: {', '.join(missing)}"
            )
        ordered = frame.sort(
            [CANONICAL.event_ts, CANONICAL.source_row_number],
            maintain_order=True,
        ).with_columns(
            pl.col(CANONICAL.event_ts).cast(pl.Datetime(time_zone="UTC"), strict=True)
        )
        first_timestamp = _as_utc_datetime(ordered[CANONICAL.event_ts].min())
        if self._last_processed_ts is not None and first_timestamp <= self._last_processed_ts:
            raise ValueError(
                "Graph-stat partitions must not split or revisit an event timestamp."
            )

        rows: list[dict[str, float | str]] = []
        for batch in ordered.partition_by(CANONICAL.event_ts, maintain_order=True):
            event_ts = _as_utc_datetime(batch[CANONICAL.event_ts][0])
            pending_edges: list[tuple[str, str]] = []
            for row in batch.iter_rows(named=True):
                sender = str(row[CANONICAL.sender_account_id])
                receiver = str(row[CANONICAL.receiver_account_id])
                directed_edge = (sender, receiver)
                reverse_edge = (receiver, sender)
                rows.append(
                    {
                        CANONICAL.transaction_id: str(row[CANONICAL.transaction_id]),
                        "graph_sender_historical_out_degree": float(
                            len(self._outgoing_neighbors[sender])
                        ),
                        "graph_sender_historical_in_degree": float(
                            len(self._incoming_neighbors[sender])
                        ),
                        "graph_receiver_historical_out_degree": float(
                            len(self._outgoing_neighbors[receiver])
                        ),
                        "graph_receiver_historical_in_degree": float(
                            len(self._incoming_neighbors[receiver])
                        ),
                        "graph_directed_edge_prior_count": float(
                            self._edge_counts[directed_edge]
                        ),
                        "graph_reverse_edge_prior_count": float(
                            self._edge_counts[reverse_edge]
                        ),
                        "graph_prior_reciprocal_relationship": float(
                            self._edge_counts[reverse_edge] > 0
                        ),
                    }
                )
                pending_edges.append(directed_edge)
            for sender, receiver in pending_edges:
                self._outgoing_neighbors[sender].add(receiver)
                self._incoming_neighbors[receiver].add(sender)
                self._edge_counts[(sender, receiver)] += 1
            self._last_processed_ts = event_ts

        return pl.DataFrame(rows)
