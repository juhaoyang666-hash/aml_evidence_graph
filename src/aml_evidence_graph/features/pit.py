"""Exact Point-in-Time account and relationship features.

The builder deliberately delays state updates for every transaction sharing an
event timestamp. Therefore no transaction can use another same-second event as
history when the source system does not provide an ordering guarantee.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Final

import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL

WINDOWS: Final[dict[str, pd.Timedelta]] = {
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
    "7d": pd.Timedelta(days=7),
    "14d": pd.Timedelta(days=14),
    "30d": pd.Timedelta(days=30),
}
MAX_WINDOW: Final[pd.Timedelta] = max(WINDOWS.values())


@dataclass(frozen=True, slots=True)
class HistoryEvent:
    event_ts: pd.Timestamp
    amount: float
    currency: str
    counterparty_id: str
    cross_border: bool


class RollingHistory:
    """Bounded event history for one account direction or account pair."""

    def __init__(self) -> None:
        self.events: deque[HistoryEvent] = deque()

    def _prune(self, as_of: pd.Timestamp) -> None:
        cutoff = as_of - MAX_WINDOW
        while self.events and self.events[0].event_ts < cutoff:
            self.events.popleft()

    def summary(self, as_of: pd.Timestamp, *, current_currency: str) -> dict[str, float]:
        """Return window aggregates using only events strictly before as_of."""
        self._prune(as_of)
        result: dict[str, float] = {}
        counters = {
            name: {
                "count": 0,
                "same_currency_amount": 0.0,
                "counterparties": set(),
                "cross_border_count": 0,
            }
            for name in WINDOWS
        }

        for event in reversed(self.events):
            if event.event_ts >= as_of:
                raise AssertionError("Future or same-timestamp history was inserted.")
            for name, window in WINDOWS.items():
                if event.event_ts < as_of - window:
                    continue
                counter = counters[name]
                counter["count"] += 1
                counter["counterparties"].add(event.counterparty_id)
                if event.currency == current_currency:
                    counter["same_currency_amount"] += event.amount
                if event.cross_border:
                    counter["cross_border_count"] += 1

        for name, counter in counters.items():
            result[f"count_{name}"] = float(counter["count"])
            result[f"same_currency_amount_sum_{name}"] = float(counter["same_currency_amount"])
            result[f"unique_counterparties_{name}"] = float(len(counter["counterparties"]))
            result[f"cross_border_count_{name}"] = float(counter["cross_border_count"])
        return result

    def append(self, event: HistoryEvent) -> None:
        if self.events and event.event_ts < self.events[-1].event_ts:
            raise ValueError("History events must be appended in non-decreasing event-time order.")
        self.events.append(event)


class PITFeatureBuilder:
    """Build exact causal features over sequential chronological partitions."""

    def __init__(self) -> None:
        self._outgoing: dict[str, RollingHistory] = defaultdict(RollingHistory)
        self._incoming: dict[str, RollingHistory] = defaultdict(RollingHistory)
        self._relationships: dict[tuple[str, str], RollingHistory] = defaultdict(RollingHistory)
        self._last_processed_ts: pd.Timestamp | None = None

    @staticmethod
    def _feature_prefix(prefix: str, summary: dict[str, float]) -> dict[str, float]:
        return {f"{prefix}_{name}": value for name, value in summary.items()}

    def transform_partition(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Transform one complete chronological partition while retaining history.

        Partitions must be supplied in non-decreasing event-time order. A full
        event date is the recommended unit, so no timestamp is split between
        calls. The input source row gives deterministic ordering for output only;
        it does not make same-timestamp records visible to one another.
        """
        required = {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.amount,
            CANONICAL.payment_currency,
            CANONICAL.sender_location,
            CANONICAL.receiver_location,
            CANONICAL.source_row_number,
        }
        missing = sorted(required.difference(transactions.columns))
        if missing:
            raise ValueError(f"Transactions are missing required PIT columns: {', '.join(missing)}")
        if transactions[CANONICAL.transaction_id].duplicated().any():
            raise ValueError("transaction_id must be unique within a PIT partition.")

        ordered = transactions.sort_values(
            [CANONICAL.event_ts, CANONICAL.source_row_number],
            kind="stable",
        ).copy()
        ordered[CANONICAL.event_ts] = pd.to_datetime(
            ordered[CANONICAL.event_ts],
            utc=True,
            errors="raise",
        )
        first_partition_timestamp = ordered[CANONICAL.event_ts].min()
        if (
            self._last_processed_ts is not None
            and first_partition_timestamp <= self._last_processed_ts
        ):
            raise ValueError(
                "PIT partitions must not split or revisit an event timestamp; "
                "process complete chronological partitions."
            )

        feature_rows: list[dict[str, float | str]] = []
        for event_ts, batch in ordered.groupby(CANONICAL.event_ts, sort=False):
            pending_events: list[tuple[str, str, HistoryEvent, HistoryEvent]] = []
            for row in batch.itertuples(index=False):
                sender = str(getattr(row, CANONICAL.sender_account_id))
                receiver = str(getattr(row, CANONICAL.receiver_account_id))
                currency = str(getattr(row, CANONICAL.payment_currency))
                received_currency = (
                    str(getattr(row, CANONICAL.received_currency))
                    if CANONICAL.received_currency in ordered.columns
                    else currency
                )
                amount = float(getattr(row, CANONICAL.amount))
                cross_border = str(getattr(row, CANONICAL.sender_location)) != str(
                    getattr(row, CANONICAL.receiver_location)
                )

                sender_summary = self._outgoing[sender].summary(
                    event_ts,
                    current_currency=currency,
                )
                receiver_summary = self._incoming[receiver].summary(
                    event_ts,
                    current_currency=currency,
                )
                relationship_summary = self._relationships[(sender, receiver)].summary(
                    event_ts,
                    current_currency=currency,
                )

                output: dict[str, float | str] = {
                    CANONICAL.transaction_id: str(getattr(row, CANONICAL.transaction_id)),
                    "is_new_sender_account": float(not self._outgoing[sender].events),
                    "is_new_receiver_account": float(not self._incoming[receiver].events),
                    "is_cross_border_current_transaction": float(cross_border),
                    "amount_log1p": math.log1p(amount),
                    "is_currency_conversion": float(currency != received_currency),
                }
                output.update(self._feature_prefix("sender_outgoing", sender_summary))
                output.update(self._feature_prefix("receiver_incoming", receiver_summary))
                output.update(self._feature_prefix("relationship", relationship_summary))
                feature_rows.append(output)

                outgoing_event = HistoryEvent(
                    event_ts=event_ts,
                    amount=amount,
                    currency=currency,
                    counterparty_id=receiver,
                    cross_border=cross_border,
                )
                incoming_event = HistoryEvent(
                    event_ts=event_ts,
                    amount=amount,
                    currency=currency,
                    counterparty_id=sender,
                    cross_border=cross_border,
                )
                pending_events.append((sender, receiver, outgoing_event, incoming_event))

            for sender, receiver, outgoing_event, incoming_event in pending_events:
                self._outgoing[sender].append(outgoing_event)
                self._incoming[receiver].append(incoming_event)
                self._relationships[(sender, receiver)].append(outgoing_event)
            self._last_processed_ts = event_ts

        features = pd.DataFrame(feature_rows)
        return ordered.merge(
            features,
            on=CANONICAL.transaction_id,
            how="left",
            validate="one_to_one",
        )
