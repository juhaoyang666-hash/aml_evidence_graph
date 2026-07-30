"""Build a causal cold-start/low-degree feature sidecar from frozen PIT rows.

The input is an existing date-partitioned PIT feature dataset.  The builder
copies every input column and appends candidate graph-edge features.  Account
state is updated only after all transactions sharing one timestamp have been
scored, so same-timestamp edges never become history for one another.

``observed_age_days`` means time since the account first appeared in the
available dataset.  It is not customer tenure or account-opening age.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from aml_evidence_graph.compat import to_polars
from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.tracking.run import create_run_manifest

COLD_START_FEATURE_VERSION = "cold-start-v3-sidecar"
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class CandidateFeatureDefinition:
    """Auditable definition for one experimental feature."""

    feature_name: str
    definition: str
    source_columns: tuple[str, ...]
    available_at: str = "strictly_before_event_time"
    window: str = "all_strictly_prior_events"


DEGREE_COLUMNS = (
    "graph_sender_historical_out_degree",
    "graph_sender_historical_in_degree",
    "graph_receiver_historical_out_degree",
    "graph_receiver_historical_in_degree",
)

CANDIDATE_FEATURES = (
    CandidateFeatureDefinition(
        "graph_sender_seen_before_any_role",
        "1 when the sender appeared as sender or receiver before event time.",
        (CANONICAL.sender_account_id, CANONICAL.event_ts),
    ),
    CandidateFeatureDefinition(
        "graph_receiver_seen_before_any_role",
        "1 when the receiver appeared as sender or receiver before event time.",
        (CANONICAL.receiver_account_id, CANONICAL.event_ts),
    ),
    CandidateFeatureDefinition(
        "graph_either_endpoint_unseen_before",
        "1 when either endpoint has no transaction strictly before event time.",
        (
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.event_ts,
        ),
    ),
    CandidateFeatureDefinition(
        "graph_both_endpoints_unseen_before",
        "1 when neither endpoint has a transaction strictly before event time.",
        (
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.event_ts,
        ),
    ),
    CandidateFeatureDefinition(
        "graph_sender_observed_age_days",
        "Days since the sender was first observed in any role; 0 when unseen.",
        (CANONICAL.sender_account_id, CANONICAL.event_ts),
    ),
    CandidateFeatureDefinition(
        "graph_receiver_observed_age_days",
        "Days since the receiver was first observed in any role; 0 when unseen.",
        (CANONICAL.receiver_account_id, CANONICAL.event_ts),
    ),
    CandidateFeatureDefinition(
        "graph_sender_total_historical_degree",
        "Sender historical in-degree plus out-degree.",
        DEGREE_COLUMNS[:2],
    ),
    CandidateFeatureDefinition(
        "graph_receiver_total_historical_degree",
        "Receiver historical in-degree plus out-degree.",
        DEGREE_COLUMNS[2:],
    ),
    CandidateFeatureDefinition(
        "graph_endpoint_min_historical_degree",
        "Minimum total historical degree across the two endpoints.",
        DEGREE_COLUMNS,
    ),
    CandidateFeatureDefinition(
        "graph_endpoint_max_historical_degree",
        "Maximum total historical degree across the two endpoints.",
        DEGREE_COLUMNS,
    ),
    CandidateFeatureDefinition(
        "graph_endpoint_degree_imbalance_ratio",
        "Absolute endpoint total-degree difference divided by 1 plus their sum.",
        DEGREE_COLUMNS,
    ),
    CandidateFeatureDefinition(
        "graph_sender_receiver_only_transition",
        "1 when sender has prior incoming neighbors but no prior outgoing neighbor.",
        DEGREE_COLUMNS[:2],
    ),
    CandidateFeatureDefinition(
        "graph_receiver_sender_only_transition",
        "1 when receiver has prior outgoing neighbors but no prior incoming neighbor.",
        DEGREE_COLUMNS[2:],
    ),
)

CANDIDATE_FEATURE_NAMES = tuple(item.feature_name for item in CANDIDATE_FEATURES)
COLD_START_FEATURE_FAMILIES = {
    "event_time_novelty": CANDIDATE_FEATURE_NAMES[:6],
    "degree_interactions": CANDIDATE_FEATURE_NAMES[6:],
}


@dataclass(frozen=True, slots=True)
class ColdStartFeatureSummary:
    """Aggregate-only summary of a sidecar feature build."""

    version: str
    input_root: str
    output_root: str
    partition_count: int
    row_count: int
    feature_count: int
    event_date_min: str
    event_date_max: str
    run_id: str


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    parsed = pl.Series([value]).cast(pl.Datetime(time_zone="UTC"), strict=True)[0]
    assert isinstance(parsed, datetime)
    return parsed


class ColdStartGraphFeatureBuilder:
    """Append causal account-observation and endpoint-degree interactions."""

    def __init__(self) -> None:
        self._first_seen_at: dict[str, datetime] = {}
        self._last_processed_ts: datetime | None = None

    def transform_partition(self, transactions: pl.DataFrame | object) -> pl.DataFrame:
        """Transform one complete chronological partition without same-time leakage."""
        frame = to_polars(transactions)
        required = {
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.sender_account_id,
            CANONICAL.receiver_account_id,
            CANONICAL.source_row_number,
            *DEGREE_COLUMNS,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                "Cold-start sidecar input is missing required columns: " + ", ".join(missing)
            )
        if frame.is_empty():
            raise ValueError("Cold-start sidecar cannot transform an empty partition.")
        if frame[CANONICAL.transaction_id].is_duplicated().any():
            raise ValueError("transaction_id must be unique within a cold-start partition.")

        ordered = frame.sort(
            [CANONICAL.event_ts, CANONICAL.source_row_number],
            maintain_order=True,
        ).with_columns(pl.col(CANONICAL.event_ts).cast(pl.Datetime(time_zone="UTC"), strict=True))
        first_timestamp = _as_utc_datetime(ordered[CANONICAL.event_ts].min())
        if self._last_processed_ts is not None and first_timestamp <= self._last_processed_ts:
            raise ValueError("Cold-start partitions must not split or revisit an event timestamp.")

        invalid_degrees = ordered.select(
            pl.any_horizontal(
                *[
                    pl.col(column).is_null()
                    | pl.col(column).is_nan()
                    | pl.col(column).is_infinite()
                    | (pl.col(column) < 0)
                    for column in DEGREE_COLUMNS
                ]
            ).any()
        ).item()
        if invalid_degrees:
            raise ValueError("Historical graph degrees must be finite and non-negative.")

        account_column = "__cold_start_account_id"
        current_first_column = "__cold_start_current_first_seen_at"
        prior_first_column = "__cold_start_prior_first_seen_at"
        account_events = pl.concat(
            [
                ordered.select(
                    pl.col(CANONICAL.sender_account_id).cast(pl.String).alias(account_column),
                    pl.col(CANONICAL.event_ts),
                ),
                ordered.select(
                    pl.col(CANONICAL.receiver_account_id).cast(pl.String).alias(account_column),
                    pl.col(CANONICAL.event_ts),
                ),
            ]
        )
        current_first = account_events.group_by(account_column).agg(
            pl.col(CANONICAL.event_ts).min().alias(current_first_column)
        )
        relevant_accounts = current_first[account_column].to_list()
        prior_rows = [
            (account, self._first_seen_at[account])
            for account in relevant_accounts
            if account in self._first_seen_at
        ]
        if prior_rows:
            prior_first = pl.DataFrame(
                prior_rows,
                schema={
                    account_column: pl.String,
                    prior_first_column: pl.Datetime(time_zone="UTC"),
                },
                orient="row",
            )
            account_first = current_first.join(prior_first, on=account_column, how="left")
        else:
            account_first = current_first.with_columns(
                pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias(prior_first_column)
            )
        account_first = account_first.with_columns(
            pl.coalesce(prior_first_column, current_first_column).alias("__first_seen_at")
        ).select(account_column, "__first_seen_at")

        sender_first = account_first.rename(
            {
                account_column: CANONICAL.sender_account_id,
                "__first_seen_at": "__sender_first_seen_at",
            }
        )
        receiver_first = account_first.rename(
            {
                account_column: CANONICAL.receiver_account_id,
                "__first_seen_at": "__receiver_first_seen_at",
            }
        )
        enriched = ordered.with_columns(
            pl.col(CANONICAL.sender_account_id).cast(pl.String),
            pl.col(CANONICAL.receiver_account_id).cast(pl.String),
        ).join(sender_first, on=CANONICAL.sender_account_id, how="left", maintain_order="left")
        enriched = enriched.join(
            receiver_first,
            on=CANONICAL.receiver_account_id,
            how="left",
            maintain_order="left",
        ).with_columns(
            (pl.col("__sender_first_seen_at") < pl.col(CANONICAL.event_ts)).alias("__sender_seen"),
            (pl.col("__receiver_first_seen_at") < pl.col(CANONICAL.event_ts)).alias(
                "__receiver_seen"
            ),
            (
                pl.col("graph_sender_historical_out_degree")
                + pl.col("graph_sender_historical_in_degree")
            ).alias("__sender_total_degree"),
            (
                pl.col("graph_receiver_historical_out_degree")
                + pl.col("graph_receiver_historical_in_degree")
            ).alias("__receiver_total_degree"),
        )
        result = enriched.with_columns(
            pl.col("__sender_seen").cast(pl.Float64).alias("graph_sender_seen_before_any_role"),
            pl.col("__receiver_seen").cast(pl.Float64).alias("graph_receiver_seen_before_any_role"),
            (~pl.col("__sender_seen") | ~pl.col("__receiver_seen"))
            .cast(pl.Float64)
            .alias("graph_either_endpoint_unseen_before"),
            (~pl.col("__sender_seen") & ~pl.col("__receiver_seen"))
            .cast(pl.Float64)
            .alias("graph_both_endpoints_unseen_before"),
            pl.when("__sender_seen")
            .then(
                (pl.col(CANONICAL.event_ts) - pl.col("__sender_first_seen_at")).dt.total_seconds()
                / SECONDS_PER_DAY
            )
            .otherwise(0.0)
            .alias("graph_sender_observed_age_days"),
            pl.when("__receiver_seen")
            .then(
                (pl.col(CANONICAL.event_ts) - pl.col("__receiver_first_seen_at")).dt.total_seconds()
                / SECONDS_PER_DAY
            )
            .otherwise(0.0)
            .alias("graph_receiver_observed_age_days"),
            pl.col("__sender_total_degree").alias("graph_sender_total_historical_degree"),
            pl.col("__receiver_total_degree").alias("graph_receiver_total_historical_degree"),
            pl.min_horizontal("__sender_total_degree", "__receiver_total_degree").alias(
                "graph_endpoint_min_historical_degree"
            ),
            pl.max_horizontal("__sender_total_degree", "__receiver_total_degree").alias(
                "graph_endpoint_max_historical_degree"
            ),
            (
                (pl.col("__sender_total_degree") - pl.col("__receiver_total_degree")).abs()
                / (1.0 + pl.col("__sender_total_degree") + pl.col("__receiver_total_degree"))
            ).alias("graph_endpoint_degree_imbalance_ratio"),
            (
                (pl.col("graph_sender_historical_out_degree") == 0)
                & (pl.col("graph_sender_historical_in_degree") > 0)
            )
            .cast(pl.Float64)
            .alias("graph_sender_receiver_only_transition"),
            (
                (pl.col("graph_receiver_historical_in_degree") == 0)
                & (pl.col("graph_receiver_historical_out_degree") > 0)
            )
            .cast(pl.Float64)
            .alias("graph_receiver_sender_only_transition"),
        ).drop(
            "__sender_first_seen_at",
            "__receiver_first_seen_at",
            "__sender_seen",
            "__receiver_seen",
            "__sender_total_degree",
            "__receiver_total_degree",
        )

        for account, event_ts in current_first.iter_rows():
            self._first_seen_at.setdefault(str(account), _as_utc_datetime(event_ts))
        self._last_processed_ts = _as_utc_datetime(ordered[CANONICAL.event_ts].max())
        return result


def _discover_event_dates(input_root: Path) -> list[str]:
    event_dates: list[str] = []
    for path in sorted(input_root.glob("event_date=*")):
        if not path.is_dir():
            continue
        value = path.name.removeprefix("event_date=")
        date.fromisoformat(value)
        event_dates.append(value)
    if not event_dates:
        raise FileNotFoundError(
            f"No Hive event_date partitions found beneath feature dataset: {input_root}"
        )
    return event_dates


def _read_partition(input_root: Path, event_date: str) -> tuple[pl.DataFrame, str]:
    split_directories = sorted((input_root / f"event_date={event_date}").glob("split=*"))
    split_directories = [path for path in split_directories if path.is_dir()]
    if len(split_directories) != 1:
        raise ValueError(f"event_date={event_date} must contain exactly one split directory.")
    split = split_directories[0].name.removeprefix("split=")
    paths = sorted(split_directories[0].rglob("*.parquet"))
    if not paths:
        raise ValueError(f"event_date={event_date} contains no Parquet files.")
    frames = [pl.read_parquet(path) for path in paths]
    return pl.concat(frames, how="diagonal_relaxed"), split


def build_cold_start_feature_sidecar(
    input_root: Path,
    output_root: Path,
    *,
    max_dates: int | None = None,
) -> ColdStartFeatureSummary:
    """Copy one PIT dataset and append the candidate columns into a new root."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"PIT feature input does not exist: {input_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite cold-start sidecar output: {output_root}")
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Cold-start sidecar output must differ from its input root.")

    event_dates = _discover_event_dates(input_root)
    if max_dates is not None:
        if max_dates < 1:
            raise ValueError("max_dates must be positive when provided.")
        event_dates = event_dates[:max_dates]
    output_root.mkdir(parents=True, exist_ok=False)

    builder = ColdStartGraphFeatureBuilder()
    row_count = 0
    for event_date in event_dates:
        frame, split = _read_partition(input_root, event_date)
        transformed = builder.transform_partition(frame)
        target = output_root / f"event_date={event_date}" / f"split={split}"
        target.mkdir(parents=True, exist_ok=False)
        transformed.write_parquet(
            target / "part-00000.parquet",
            compression="zstd",
        )
        row_count += transformed.height

    contract = {
        "schema_version": "1.0",
        "feature_version": COLD_START_FEATURE_VERSION,
        "base_feature_root": str(input_root),
        "feature_count": len(CANDIDATE_FEATURES),
        "features": [asdict(item) for item in CANDIDATE_FEATURES],
        "feature_families": {
            family: list(columns) for family, columns in COLD_START_FEATURE_FAMILIES.items()
        },
        "semantic_warning": (
            "observed_age_days is time since first appearance in available data, "
            "not account-opening age"
        ),
    }
    (output_root / "_cold_start_feature_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = create_run_manifest(
        output_dir=output_root,
        command="aml-build-cold-start-features",
        random_seed=0,
        input_paths={"base_pit_feature_dataset": input_root},
        metadata={
            "feature_version": COLD_START_FEATURE_VERSION,
            "feature_columns": list(CANDIDATE_FEATURE_NAMES),
            "partition_count": len(event_dates),
            "row_count": row_count,
            "max_dates": max_dates,
            "labels_used": False,
            "same_timestamp_history_visible": False,
        },
        filename="_run_manifest.json",
    )
    summary = ColdStartFeatureSummary(
        version=COLD_START_FEATURE_VERSION,
        input_root=str(input_root),
        output_root=str(output_root),
        partition_count=len(event_dates),
        row_count=row_count,
        feature_count=len(CANDIDATE_FEATURES),
        event_date_min=event_dates[0],
        event_date_max=event_dates[-1],
        run_id=manifest.run_id,
    )
    (output_root / "_cold_start_feature_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_cold_start_feature_sidecar(
        args.input,
        args.output,
        max_dates=args.max_dates,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
