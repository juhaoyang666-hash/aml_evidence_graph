"""Causal directed-graph statistics for transaction edge classification."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL


class CausalGraphStatisticsBuilder:
    """Maintain directed graph structure using only strictly earlier transactions."""

    def __init__(self) -> None:
        self._outgoing_neighbors: dict[str, set[str]] = defaultdict(set)
        self._incoming_neighbors: dict[str, set[str]] = defaultdict(set)
        self._edge_counts: Counter[tuple[str, str]] = Counter()
        self._last_processed_ts: pd.Timestamp | None = None

    def transform_partition(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Return graph features for a complete chronological event-date partition."""
        required = {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.source_row_number,
        }
        missing = sorted(required.difference(transactions.columns))
        if missing:
            raise ValueError(
                f"Transactions are missing required graph-stat columns: {', '.join(missing)}"
            )
        ordered = transactions.sort_values(
            [CANONICAL.event_ts, CANONICAL.source_row_number],
            kind="stable",
        ).copy()
        ordered[CANONICAL.event_ts] = pd.to_datetime(
            ordered[CANONICAL.event_ts],
            utc=True,
            errors="raise",
        )
        first_timestamp = ordered[CANONICAL.event_ts].min()
        if self._last_processed_ts is not None and first_timestamp <= self._last_processed_ts:
            raise ValueError(
                "Graph-stat partitions must not split or revisit an event timestamp."
            )

        rows: list[dict[str, float | str]] = []
        for event_ts, batch in ordered.groupby(CANONICAL.event_ts, sort=False):
            pending_edges: list[tuple[str, str]] = []
            for row in batch.itertuples(index=False):
                sender = str(getattr(row, CANONICAL.sender_account_id))
                receiver = str(getattr(row, CANONICAL.receiver_account_id))
                directed_edge = (sender, receiver)
                reverse_edge = (receiver, sender)
                rows.append(
                    {
                        CANONICAL.transaction_id: str(
                            getattr(row, CANONICAL.transaction_id)
                        ),
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

        return pd.DataFrame(rows)
