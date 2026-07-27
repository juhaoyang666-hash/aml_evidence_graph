"""Canonical private transaction schema and chunk-level validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from aml_evidence_graph.compat import to_polars

RAW_ENGLISH_COLUMNS: Final[tuple[str, ...]] = (
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Is_laundering",
    "Laundering_type",
)


@dataclass(frozen=True)
class CanonicalColumns:
    """Names used by all newly-created pipeline modules."""

    transaction_id: str = "transaction_id"
    event_ts: str = "event_ts"
    sender_account_id: str = "sender_account_id"
    receiver_account_id: str = "receiver_account_id"
    amount: str = "amount"
    payment_currency: str = "payment_currency"
    received_currency: str = "received_currency"
    sender_location: str = "sender_location"
    receiver_location: str = "receiver_location"
    payment_type: str = "payment_type"
    is_laundering: str = "is_laundering"
    laundering_type: str = "laundering_type"
    source_row_number: str = "source_row_number"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.transaction_id,
            self.event_ts,
            self.sender_account_id,
            self.receiver_account_id,
            self.amount,
            self.payment_currency,
            self.received_currency,
            self.sender_location,
            self.receiver_location,
            self.payment_type,
            self.is_laundering,
            self.laundering_type,
            self.source_row_number,
        )


CANONICAL = CanonicalColumns()


class DataContractError(ValueError):
    """Raised when a source chunk cannot meet the canonical transaction contract."""


def validate_raw_columns(columns: list[str] | tuple[str, ...] | pl.Series) -> None:
    """Validate the supported source schema before data is read at scale."""
    column_names = list(columns) if not isinstance(columns, pl.Series) else columns.to_list()
    missing = sorted(set(RAW_ENGLISH_COLUMNS).difference(column_names))
    if missing:
        raise DataContractError(f"Input is missing required columns: {', '.join(missing)}")


def _validate_chunk_values(frame: pl.DataFrame) -> None:
    required = (
        CANONICAL.event_ts,
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        CANONICAL.amount,
        CANONICAL.is_laundering,
    )
    failures: dict[str, int] = {}
    for name in required:
        null_count = int(frame[name].null_count())
        if null_count:
            failures[name] = null_count
    if failures:
        raise DataContractError(f"Required canonical values are null: {failures}")

    observed_labels = set(frame[CANONICAL.is_laundering].unique().to_list())
    if not observed_labels.issubset({0, 1}):
        raise DataContractError(
            f"Expected binary is_laundering labels {{0, 1}}, received {sorted(observed_labels)}"
        )

    if (frame[CANONICAL.amount] < 0).any():
        raise DataContractError("Amounts must be non-negative in the canonical transaction table.")


def normalize_transaction_chunk(
    raw: pl.DataFrame | object,
    *,
    source_row_start: int,
    timezone: str = "UTC",
) -> pl.DataFrame:
    """Map a validated raw transaction chunk to the canonical schema.

    A deterministic transaction_id is derived from the immutable source row number.
    It is an internal pipeline identifier, not a customer or transaction identifier.
    """
    frame = to_polars(raw)
    validate_raw_columns(frame.columns)
    if source_row_start < 1:
        raise ValueError("source_row_start must be one-based and positive.")

    source_rows = list(range(source_row_start, source_row_start + frame.height))
    event_text = (
        pl.col("Date").cast(pl.Utf8).str.strip_chars()
        + pl.lit(" ")
        + pl.col("Time").cast(pl.Utf8).str.strip_chars()
    )
    if timezone == "UTC":
        event_ts_expr = event_text.str.to_datetime(time_zone="UTC", strict=False)
    else:
        event_ts_expr = (
            event_text.str.to_datetime(time_zone=None, strict=False)
            .dt.replace_time_zone(timezone)
            .dt.convert_time_zone("UTC")
        )

    result = frame.with_columns(
        pl.Series(CANONICAL.source_row_number, source_rows),
        pl.Series(
            CANONICAL.transaction_id,
            [f"txn-row-{source_row:012d}" for source_row in source_rows],
        ),
        event_ts_expr.alias(CANONICAL.event_ts),
        pl.col("Sender_account").cast(pl.Utf8).str.strip_chars().alias(CANONICAL.sender_account_id),
        pl.col("Receiver_account")
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias(CANONICAL.receiver_account_id),
        pl.col("Amount").cast(pl.Float64, strict=False).alias(CANONICAL.amount),
        pl.col("Payment_currency")
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias(CANONICAL.payment_currency),
        pl.col("Received_currency")
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias(CANONICAL.received_currency),
        pl.col("Sender_bank_location")
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias(CANONICAL.sender_location),
        pl.col("Receiver_bank_location")
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias(CANONICAL.receiver_location),
        pl.col("Payment_type").cast(pl.Utf8).str.strip_chars().alias(CANONICAL.payment_type),
        pl.col("Is_laundering").cast(pl.Int64, strict=False).alias(CANONICAL.is_laundering),
        pl.col("Laundering_type").cast(pl.Utf8).str.strip_chars().alias(CANONICAL.laundering_type),
    ).select(list(CANONICAL.required_columns))
    _validate_chunk_values(result)
    return result
