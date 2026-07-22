"""Convert a private AML CSV into tokenized, date-partitioned Parquet."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aml_evidence_graph.data.configuration import DEFAULT_DATA_CONFIG_PATH, load_data_configuration
from aml_evidence_graph.data.contract import CANONICAL, normalize_transaction_chunk
from aml_evidence_graph.data.privacy import tokenise_identifier
from aml_evidence_graph.data.splits import assign_time_split
from aml_evidence_graph.settings import Settings
from aml_evidence_graph.tracking.run import create_run_manifest


@dataclass(frozen=True)
class ConversionSummary:
    """Aggregate-only trace of a private CSV-to-Parquet conversion."""

    created_at_utc: str
    run_id: str
    input_name: str
    output_root: str
    row_count: int
    chunk_count: int
    partition_count: int
    event_date_min: str
    event_date_max: str


def _tokenise_accounts(frame: pd.DataFrame, *, secret: str) -> pd.DataFrame:
    result = frame.copy()
    result[CANONICAL.sender_account_id] = result[CANONICAL.sender_account_id].map(
        lambda value: tokenise_identifier(str(value), secret=secret, namespace="account")
    )
    result[CANONICAL.receiver_account_id] = result[CANONICAL.receiver_account_id].map(
        lambda value: tokenise_identifier(str(value), secret=secret, namespace="account")
    )
    return result


def convert_csv_to_parquet(
    input_path: Path,
    output_root: Path,
    *,
    tokenization_secret: str,
    chunk_size: int = 250_000,
    timezone: str = "UTC",
    data_config_path: Path = DEFAULT_DATA_CONFIG_PATH,
    overwrite: bool = False,
) -> ConversionSummary:
    """Convert the raw source to a private tokenized Parquet dataset.

    Source account identifiers exist only in the in-memory chunk before being
    deterministically tokenized. The output is partitioned by event_date and
    chronological split; no label field is used to make the partition.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    configuration = load_data_configuration(data_config_path)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. "
                "Use overwrite=True only for private artefacts."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    source_row_start = 1
    row_count = 0
    chunk_count = 0
    event_dates: set[str] = set()
    for raw_chunk in pd.read_csv(input_path, chunksize=chunk_size):
        normalized = normalize_transaction_chunk(
            raw_chunk,
            source_row_start=source_row_start,
            timezone=timezone,
        )
        source_row_start += len(normalized)
        row_count += len(normalized)
        chunk_count += 1
        tokenized = _tokenise_accounts(normalized, secret=tokenization_secret)
        tokenized["event_date"] = tokenized[CANONICAL.event_ts].dt.strftime("%Y-%m-%d")
        event_dates.update(tokenized["event_date"].unique().tolist())
        tokenized["split"] = assign_time_split(tokenized[CANONICAL.event_ts])
        table = pa.Table.from_pandas(tokenized, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(output_root),
            partition_cols=["event_date", "split"],
            compression="zstd",
        )
    if not event_dates:
        raise ValueError("Input CSV contains no transactions after schema validation.")
    manifest = create_run_manifest(
        output_dir=output_root,
        command="aml-convert-private-data",
        random_seed=0,
        input_paths={"raw_csv": input_path},
        config_paths={"data_config": data_config_path},
        metadata={
            "chunk_size": chunk_size,
            "timezone": timezone,
            "data_configuration_version": configuration.version,
            "row_count": row_count,
            "partition_count": len(event_dates),
        },
        filename="_run_manifest.json",
    )
    summary = ConversionSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        run_id=manifest.run_id,
        input_name=input_path.name,
        output_root=str(output_root),
        row_count=row_count,
        chunk_count=chunk_count,
        partition_count=len(event_dates),
        event_date_min=min(event_dates),
        event_date_max=max(event_dates),
    )
    (output_root / "_conversion_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    summary = convert_csv_to_parquet(
        args.input,
        args.output,
        tokenization_secret=settings.require_tokenization_secret(),
        chunk_size=args.chunk_size,
        timezone=args.timezone,
        data_config_path=args.data_config,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {"run_id": summary.run_id, "rows": summary.row_count, "output": str(args.output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
