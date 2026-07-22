"""Canonical private transaction schema and chunk-level validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

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


def validate_raw_columns(columns: pd.Index) -> None:
    """Validate the supported source schema before data is read at scale."""
    missing = sorted(set(RAW_ENGLISH_COLUMNS).difference(columns))
    if missing:
        raise DataContractError(f"Input is missing required columns: {', '.join(missing)}")


def _validate_chunk_values(frame: pd.DataFrame) -> None:
    required = (
        CANONICAL.event_ts,
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        CANONICAL.amount,
        CANONICAL.is_laundering,
    )
    null_counts = frame.loc[:, required].isna().sum()
    failures = {name: int(count) for name, count in null_counts.items() if count}
    if failures:
        raise DataContractError(f"Required canonical values are null: {failures}")

    observed_labels = set(frame[CANONICAL.is_laundering].unique())
    if not observed_labels.issubset({0, 1}):
        raise DataContractError(
            f"Expected binary is_laundering labels {{0, 1}}, received {sorted(observed_labels)}"
        )

    if (frame[CANONICAL.amount] < 0).any():
        raise DataContractError("Amounts must be non-negative in the canonical transaction table.")


def normalize_transaction_chunk(
    raw: pd.DataFrame,
    *,
    source_row_start: int,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Map a validated raw transaction chunk to the canonical schema.

    A deterministic transaction_id is derived from the immutable source row number.
    It is an internal pipeline identifier, not a customer or transaction identifier.
    """
    validate_raw_columns(raw.columns)
    if source_row_start < 1:
        raise ValueError("source_row_start must be one-based and positive.")

    result = pd.DataFrame(index=raw.index)
    source_rows = list(range(source_row_start, source_row_start + len(raw)))
    result[CANONICAL.source_row_number] = source_rows
    result[CANONICAL.transaction_id] = [
        f"txn-row-{source_row:012d}" for source_row in source_rows
    ]

    event_text = (
        raw["Date"].astype("string").str.strip()
        + " "
        + raw["Time"].astype("string").str.strip()
    )
    event_ts = pd.to_datetime(event_text, errors="coerce", utc=True)
    if timezone != "UTC":
        event_ts = (
            pd.to_datetime(event_text, errors="coerce")
            .dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )

    result[CANONICAL.event_ts] = event_ts
    result[CANONICAL.sender_account_id] = raw["Sender_account"].astype("string").str.strip()
    result[CANONICAL.receiver_account_id] = raw["Receiver_account"].astype("string").str.strip()
    result[CANONICAL.amount] = pd.to_numeric(raw["Amount"], errors="coerce")
    result[CANONICAL.payment_currency] = raw["Payment_currency"].astype("string").str.strip()
    result[CANONICAL.received_currency] = raw["Received_currency"].astype("string").str.strip()
    result[CANONICAL.sender_location] = raw["Sender_bank_location"].astype("string").str.strip()
    result[CANONICAL.receiver_location] = raw["Receiver_bank_location"].astype("string").str.strip()
    result[CANONICAL.payment_type] = raw["Payment_type"].astype("string").str.strip()
    result[CANONICAL.is_laundering] = pd.to_numeric(
        raw["Is_laundering"], errors="coerce"
    ).astype("Int64")
    result[CANONICAL.laundering_type] = raw["Laundering_type"].astype("string").str.strip()
    _validate_chunk_values(result)
    return result.loc[:, CANONICAL.required_columns]
