"""Create an aggregate-only manifest for an AML CSV without exposing records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from aml_evidence_graph.data.configuration import (
    DEFAULT_DATA_CONFIG_PATH,
    DataConfiguration,
    load_data_configuration,
)
from aml_evidence_graph.data.contract import RAW_ENGLISH_COLUMNS, validate_raw_columns
from aml_evidence_graph.data.splits import assign_time_split, split_bounds_as_iso

PROFILE_COLUMNS = RAW_ENGLISH_COLUMNS
QUALITY_REQUIRED_RAW_COLUMNS = (
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Is_laundering",
)


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file fingerprint without retaining transaction contents."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _merge_counts(target: Counter[str], values: pl.Series) -> None:
    value_counts = values.value_counts(sort=False)
    name = values.name or "values"
    for label, count in zip(
        value_counts[name].to_list(),
        value_counts["count"].to_list(),
        strict=True,
    ):
        target[str(label)] += int(count)


def _validate_null_rates(
    null_counts: Counter[str],
    *,
    row_count: int,
    configuration: DataConfiguration,
) -> dict[str, float]:
    """Apply configured gates only to canonical required source values."""
    rates = {
        column: null_counts[column] / row_count if row_count else 0.0
        for column in QUALITY_REQUIRED_RAW_COLUMNS
    }
    violations = {
        column: rate
        for column, rate in rates.items()
        if rate > configuration.quality.max_null_rate_for_required_columns
    }
    if violations:
        rendered = ", ".join(f"{name}={rate:.6f}" for name, rate in sorted(violations.items()))
        raise ValueError("Required source-column null rate exceeds configured limit: " + rendered)
    return rates


def build_manifest(
    input_path: Path,
    *,
    chunk_size: int = 250_000,
    data_config_path: Path = DEFAULT_DATA_CONFIG_PATH,
) -> dict[str, Any]:
    """Profile a raw CSV in chunks and return only aggregate quality metadata."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    configuration = load_data_configuration(data_config_path)

    header = pl.read_csv(input_path, n_rows=0)
    validate_raw_columns(header.columns)

    row_count = 0
    positive_count = 0
    min_timestamp: datetime | None = None
    max_timestamp: datetime | None = None
    null_counts: Counter[str] = Counter()
    monthly_rows: Counter[str] = Counter()
    monthly_positives: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_positives: Counter[str] = Counter()

    reader_batches = pl.scan_csv(str(input_path)).collect_batches(chunk_size=chunk_size)
    for batch in reader_batches:
        chunk = batch.select(list(PROFILE_COLUMNS))
        row_count += chunk.height
        null_row = chunk.null_count().row(0, named=True)
        null_counts.update({name: int(value) for name, value in null_row.items()})
        event_ts = (
            chunk["Date"].cast(pl.Utf8).str.strip_chars()
            + " "
            + chunk["Time"].cast(pl.Utf8).str.strip_chars()
        ).str.to_datetime(time_zone="UTC", strict=False)
        if event_ts.null_count() > 0:
            raise ValueError("Found invalid event timestamps while building manifest.")
        current_min = event_ts.min()
        current_max = event_ts.max()
        min_timestamp = current_min if min_timestamp is None else min(min_timestamp, current_min)
        max_timestamp = current_max if max_timestamp is None else max(max_timestamp, current_max)

        labels = chunk["Is_laundering"].cast(pl.Float64, strict=False)
        if configuration.quality.require_binary_label:
            invalid_labels = set(labels.unique().to_list()).difference({0.0, 1.0})
            if invalid_labels:
                raise ValueError(f"Found non-binary labels: {sorted(invalid_labels)}")
        positive_mask = labels.eq(1)
        positive_count += int(positive_mask.sum())
        months = event_ts.dt.strftime("%Y-%m")
        _merge_counts(monthly_rows, months)
        _merge_counts(monthly_positives, months.filter(positive_mask))
        _merge_counts(
            type_counts,
            chunk.filter(positive_mask)["Laundering_type"].cast(pl.Utf8),
        )

        splits = assign_time_split(event_ts)
        _merge_counts(split_rows, splits)
        _merge_counts(split_positives, splits.filter(positive_mask))

    assert min_timestamp is not None
    assert max_timestamp is not None
    required_null_rates = _validate_null_rates(
        null_counts,
        row_count=row_count,
        configuration=configuration,
    )
    return {
        "manifest_schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": {
            "file_name": input_path.name,
            "sha256": file_sha256(input_path),
            "raw_columns": list(header.columns),
        },
        "aggregate_profile": {
            "row_count": row_count,
            "positive_count": positive_count,
            "positive_rate": positive_count / row_count if row_count else 0.0,
            "event_timestamp_min": min_timestamp.isoformat(),
            "event_timestamp_max": max_timestamp.isoformat(),
            "null_counts": dict(sorted(null_counts.items())),
            "positive_laundering_type_counts": dict(sorted(type_counts.items())),
            "monthly": {
                month: {
                    "row_count": monthly_rows[month],
                    "positive_count": monthly_positives[month],
                    "positive_rate": monthly_positives[month] / monthly_rows[month],
                }
                for month in sorted(monthly_rows)
            },
            "time_split_counts": {
                split: {
                    "row_count": split_rows[split],
                    "positive_count": split_positives[split],
                    "positive_rate": split_positives[split] / split_rows[split],
                }
                for split in sorted(split_rows)
            },
        },
        "pre_registered_time_splits": split_bounds_as_iso(),
        "data_configuration": {
            "version": configuration.version,
            "required_column_null_rates": required_null_rates,
            "max_null_rate_for_required_columns": (
                configuration.quality.max_null_rate_for_required_columns
            ),
        },
        "privacy_notice": (
            "This file intentionally contains aggregate counts and a source fingerprint only. "
            "It must not include account identifiers, transaction identifiers, "
            "or transaction records."
        ),
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Atomically write a private aggregate manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Private source CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Aggregate manifest JSON path.")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.input,
        chunk_size=args.chunk_size,
        data_config_path=args.data_config,
    )
    write_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "row_count": manifest["aggregate_profile"]["row_count"],
                "positive_count": manifest["aggregate_profile"]["positive_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
