"""Exact Point-in-Time account and relationship features.

The builder deliberately delays state updates for every transaction sharing an
event timestamp. Therefore no transaction can use another same-second event as
history when the source system does not provide an ordering guarantee.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import polars as pl

from aml_evidence_graph.compat import to_polars
from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.engineering_config import FeatureEngineeringConfig

WINDOWS: Final[dict[str, timedelta]] = {
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
    "7d": timedelta(days=7),
    "14d": timedelta(days=14),
    "30d": timedelta(days=30),
}
MAX_WINDOW: Final[timedelta] = max(WINDOWS.values())
_EPS: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class HistoryEvent:
    event_ts: datetime
    amount: float
    currency: str
    counterparty_id: str
    cross_border: bool


class RollingHistory:
    """Bounded event history for one account direction or account pair."""

    def __init__(self) -> None:
        self.events: deque[HistoryEvent] = deque()

    def _prune(self, as_of: datetime) -> None:
        cutoff = as_of - MAX_WINDOW
        while self.events and self.events[0].event_ts < cutoff:
            self.events.popleft()

    def summary(
        self,
        as_of: datetime,
        *,
        current_currency: str,
        small_amount_threshold: float,
    ) -> dict[str, float]:
        """Return window aggregates using only events strictly before as_of."""
        self._prune(as_of)
        result: dict[str, float] = {}
        counters = {
            name: {
                "count": 0,
                "same_currency_amount": 0.0,
                "amount_sum": 0.0,
                "amount_sum_sq": 0.0,
                "counterparties": set(),
                "small_amount_counterparties": set(),
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
                counterparties = counter["counterparties"]
                assert isinstance(counterparties, set)
                counterparties.add(event.counterparty_id)
                counter["amount_sum"] += event.amount
                counter["amount_sum_sq"] += event.amount * event.amount
                if event.currency == current_currency:
                    counter["same_currency_amount"] += event.amount
                if event.cross_border:
                    counter["cross_border_count"] += 1
                if event.amount < small_amount_threshold:
                    small_counterparties = counter["small_amount_counterparties"]
                    assert isinstance(small_counterparties, set)
                    small_counterparties.add(event.counterparty_id)

        for name, counter in counters.items():
            counterparties = counter["counterparties"]
            small_counterparties = counter["small_amount_counterparties"]
            assert isinstance(counterparties, set)
            assert isinstance(small_counterparties, set)
            result[f"count_{name}"] = float(counter["count"])
            result[f"same_currency_amount_sum_{name}"] = float(counter["same_currency_amount"])
            result[f"unique_counterparties_{name}"] = float(len(counterparties))
            result[f"cross_border_count_{name}"] = float(counter["cross_border_count"])
            result[f"amount_sum_{name}"] = float(counter["amount_sum"])
            result[f"amount_sum_sq_{name}"] = float(counter["amount_sum_sq"])
            result[f"small_amount_unique_counterparties_{name}"] = float(
                len(small_counterparties)
            )
        return result

    def append(self, event: HistoryEvent) -> None:
        if self.events and event.event_ts < self.events[-1].event_ts:
            raise ValueError("History events must be appended in non-decreasing event-time order.")
        self.events.append(event)


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    parsed = pl.Series([value]).cast(pl.Datetime(time_zone="UTC"), strict=True)[0]
    assert isinstance(parsed, datetime)
    return parsed


def _seconds_since(
    event_ts: datetime,
    previous: datetime | None,
    *,
    missing_recency_seconds: float,
) -> float:
    if previous is None:
        return float(missing_recency_seconds)
    delta = (event_ts - previous).total_seconds()
    if delta < 0:
        raise AssertionError("History timestamp must be strictly before the current event.")
    return float(delta)


def _amount_mean_ratio(amount: float, amount_sum: float, count: float) -> float:
    mean = amount_sum / max(count, 1.0)
    return float(amount / (1.0 + mean))


def _amount_zscore(amount: float, amount_sum: float, amount_sum_sq: float, count: float) -> float:
    if count < 2.0:
        return 0.0
    mean = amount_sum / count
    variance = max(amount_sum_sq / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    return float((amount - mean) / (std + _EPS))


def _short_over_long(short: float, long: float) -> float:
    return float(short / (1.0 + long))


class PITFeatureBuilder:
    """Build exact causal features over sequential chronological partitions."""

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self._config = config or FeatureEngineeringConfig.defaults()
        self._outgoing: dict[str, RollingHistory] = defaultdict(RollingHistory)
        self._incoming: dict[str, RollingHistory] = defaultdict(RollingHistory)
        self._relationships: dict[tuple[str, str], RollingHistory] = defaultdict(RollingHistory)
        self._last_outgoing_ts: dict[str, datetime] = {}
        self._last_incoming_ts: dict[str, datetime] = {}
        self._last_cash_incoming_ts: dict[str, datetime] = {}
        self._last_processed_ts: datetime | None = None

    @staticmethod
    def _feature_prefix(prefix: str, summary: dict[str, float]) -> dict[str, float]:
        skip_suffixes = ("amount_sum_", "amount_sum_sq_", "small_amount_unique_counterparties_")
        return {
            f"{prefix}_{name}": value
            for name, value in summary.items()
            if not name.startswith(skip_suffixes)
        }

    def transform_partition(self, transactions: pl.DataFrame | object) -> pl.DataFrame:
        """Transform one complete chronological partition while retaining history.

        Partitions must be supplied in non-decreasing event-time order. A full
        event date is the recommended unit, so no timestamp is split between
        calls. The input source row gives deterministic ordering for output only;
        it does not make same-timestamp records visible to one another.
        """
        frame = to_polars(transactions)
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
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Transactions are missing required PIT columns: {', '.join(missing)}")
        if frame[CANONICAL.transaction_id].is_duplicated().any():
            raise ValueError("transaction_id must be unique within a PIT partition.")

        ordered = frame.sort(
            [CANONICAL.event_ts, CANONICAL.source_row_number],
            maintain_order=True,
        ).with_columns(
            pl.col(CANONICAL.event_ts).cast(pl.Datetime(time_zone="UTC"), strict=True)
        )
        first_partition_timestamp = _as_utc_datetime(ordered[CANONICAL.event_ts].min())
        if (
            self._last_processed_ts is not None
            and first_partition_timestamp <= self._last_processed_ts
        ):
            raise ValueError(
                "PIT partitions must not split or revisit an event timestamp; "
                "process complete chronological partitions."
            )

        has_received_currency = CANONICAL.received_currency in ordered.columns
        has_payment_type = CANONICAL.payment_type in ordered.columns
        config = self._config
        cash_window = timedelta(hours=config.cash_in_then_out_window_hours)
        just_below_floor = (
            config.reporting_threshold * config.just_below_reporting_threshold_ratio
        )
        feature_rows: list[dict[str, float | str]] = []
        for batch in ordered.partition_by(CANONICAL.event_ts, maintain_order=True):
            event_ts = _as_utc_datetime(batch[CANONICAL.event_ts][0])
            pending_events: list[
                tuple[str, str, HistoryEvent, HistoryEvent, bool]
            ] = []
            for row in batch.iter_rows(named=True):
                sender = str(row[CANONICAL.sender_account_id])
                receiver = str(row[CANONICAL.receiver_account_id])
                currency = str(row[CANONICAL.payment_currency])
                received_currency = (
                    str(row[CANONICAL.received_currency])
                    if has_received_currency
                    else currency
                )
                amount = float(row[CANONICAL.amount])
                payment_type = (
                    str(row[CANONICAL.payment_type]) if has_payment_type else ""
                )
                sender_location = str(row[CANONICAL.sender_location])
                receiver_location = str(row[CANONICAL.receiver_location])
                cross_border = sender_location != receiver_location
                is_cash_like = payment_type in config.cash_like_payment_types
                is_high_risk_sender = float(sender_location in config.high_risk_locations)
                is_high_risk_receiver = float(
                    receiver_location in config.high_risk_locations
                )

                sender_summary = self._outgoing[sender].summary(
                    event_ts,
                    current_currency=currency,
                    small_amount_threshold=config.small_amount_threshold,
                )
                receiver_summary = self._incoming[receiver].summary(
                    event_ts,
                    current_currency=currency,
                    small_amount_threshold=config.small_amount_threshold,
                )
                relationship_summary = self._relationships[(sender, receiver)].summary(
                    event_ts,
                    current_currency=currency,
                    small_amount_threshold=config.small_amount_threshold,
                )

                last_cash_in = self._last_cash_incoming_ts.get(sender)
                cash_in_then_out = float(
                    last_cash_in is not None and event_ts - last_cash_in <= cash_window
                )

                output: dict[str, float | str] = {
                    CANONICAL.transaction_id: str(row[CANONICAL.transaction_id]),
                    "is_new_sender_account": float(not self._outgoing[sender].events),
                    "is_new_receiver_account": float(not self._incoming[receiver].events),
                    "is_cross_border_current_transaction": float(cross_border),
                    "amount_log1p": math.log1p(amount),
                    "is_currency_conversion": float(currency != received_currency),
                    "is_high_risk_sender_location": is_high_risk_sender,
                    "is_high_risk_receiver_location": is_high_risk_receiver,
                    "is_high_risk_corridor": float(
                        is_high_risk_sender or is_high_risk_receiver
                    ),
                    "is_cash_like_payment": float(is_cash_like),
                    "is_cross_border_payment_type": float(
                        payment_type in config.cross_border_payment_types
                    ),
                    "is_round_amount": float(
                        abs(amount - round(amount)) <= config.round_amount_tolerance
                    ),
                    "is_just_below_reporting_threshold": float(
                        just_below_floor <= amount < config.reporting_threshold
                    ),
                    "sender_small_amount_unique_receivers_7d": float(
                        sender_summary["small_amount_unique_counterparties_7d"]
                    ),
                    "receiver_small_amount_unique_senders_7d": float(
                        receiver_summary["small_amount_unique_counterparties_7d"]
                    ),
                    "amount_to_sender_outgoing_mean_ratio_30d": _amount_mean_ratio(
                        amount,
                        sender_summary["amount_sum_30d"],
                        sender_summary["count_30d"],
                    ),
                    "amount_zscore_vs_sender_outgoing_30d": _amount_zscore(
                        amount,
                        sender_summary["amount_sum_30d"],
                        sender_summary["amount_sum_sq_30d"],
                        sender_summary["count_30d"],
                    ),
                    "seconds_since_last_outgoing": _seconds_since(
                        event_ts,
                        self._last_outgoing_ts.get(sender),
                        missing_recency_seconds=config.missing_recency_seconds,
                    ),
                    "seconds_since_last_incoming": _seconds_since(
                        event_ts,
                        self._last_incoming_ts.get(sender),
                        missing_recency_seconds=config.missing_recency_seconds,
                    ),
                    "cash_in_then_out_within_window": cash_in_then_out,
                    "sender_outgoing_count_1d_over_30d": _short_over_long(
                        sender_summary["count_1d"],
                        sender_summary["count_30d"],
                    ),
                    "receiver_incoming_count_1d_over_30d": _short_over_long(
                        receiver_summary["count_1d"],
                        receiver_summary["count_30d"],
                    ),
                    "sender_outgoing_unique_counterparties_1d_over_30d": _short_over_long(
                        sender_summary["unique_counterparties_1d"],
                        sender_summary["unique_counterparties_30d"],
                    ),
                    "receiver_incoming_unique_counterparties_1d_over_30d": _short_over_long(
                        receiver_summary["unique_counterparties_1d"],
                        receiver_summary["unique_counterparties_30d"],
                    ),
                    "hour_of_day": float(event_ts.hour),
                    "day_of_week": float(event_ts.weekday()),
                    "is_weekend": float(event_ts.weekday() >= 5),
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
                pending_events.append(
                    (sender, receiver, outgoing_event, incoming_event, is_cash_like)
                )

            for sender, receiver, outgoing_event, incoming_event, is_cash_like in pending_events:
                self._outgoing[sender].append(outgoing_event)
                self._incoming[receiver].append(incoming_event)
                self._relationships[(sender, receiver)].append(outgoing_event)
                self._last_outgoing_ts[sender] = event_ts
                self._last_incoming_ts[receiver] = event_ts
                if is_cash_like:
                    self._last_cash_incoming_ts[receiver] = event_ts
            self._last_processed_ts = event_ts

        features = pl.DataFrame(feature_rows)
        return ordered.join(features, on=CANONICAL.transaction_id, how="left")
