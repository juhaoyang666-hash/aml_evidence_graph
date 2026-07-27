"""Build a date-partitioned Point-in-Time feature dataset from prepared Parquet."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.engineering_config import (
    DEFAULT_FEATURE_ENGINEERING_CONFIG_PATH,
    FeatureEngineeringConfig,
    load_feature_engineering_config,
)
from aml_evidence_graph.features.graph_stats import CausalGraphStatisticsBuilder
from aml_evidence_graph.features.pit import PITFeatureBuilder
from aml_evidence_graph.features.registry import (
    DEFAULT_FEATURE_REGISTRY_PATH,
    load_static_feature_metadata,
    rule_feature_metadata,
    validate_feature_metadata,
    write_feature_registry,
)
from aml_evidence_graph.rules.engine import RuleHit, apply_rules, load_rules
from aml_evidence_graph.tracking.run import create_run_manifest


@dataclass(frozen=True)
class FeatureBuildSummary:
    """Aggregate-only output of a feature-build run."""

    input_root: str
    output_root: str
    partition_count: int
    row_count: int
    feature_column_count: int
    configured_rule_count: int
    rule_hit_count: int
    feature_registry_version: str
    event_date_min: str
    event_date_max: str
    created_at_utc: str
    run_id: str


def _discover_event_dates(input_root: Path) -> list[str]:
    event_dates: set[str] = set()
    for path in input_root.glob("event_date=*"):
        if not path.is_dir():
            continue
        value = path.name.removeprefix("event_date=")
        date.fromisoformat(value)
        event_dates.add(value)
    if not event_dates:
        raise FileNotFoundError(
            f"No Hive event_date partitions found beneath prepared dataset: {input_root}"
        )
    return sorted(event_dates)


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. "
            "Use overwrite=True only for a private, regenerable feature dataset."
        )
    if not output_root.is_dir():
        raise ValueError(f"Feature output path must be a directory: {output_root}")
    shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def _read_event_date_partition(input_root: Path, event_date: str) -> pl.DataFrame:
    """Read one Hive event_date partition with Polars, preserving partition columns."""
    paths = sorted(input_root.glob(f"event_date={event_date}/split=*/**/*.parquet"))
    if not paths:
        paths = sorted(input_root.glob(f"event_date={event_date}/**/*.parquet"))
    if not paths:
        raise ValueError(f"Discovered empty event-date partition: {event_date}")
    frames: list[pl.DataFrame] = []
    for path in paths:
        frame = pl.read_parquet(path)
        if "event_date" not in frame.columns:
            frame = frame.with_columns(pl.lit(event_date).alias("event_date"))
        if "split" not in frame.columns:
            split_value = next(
                (
                    part.removeprefix("split=")
                    for part in path.parts
                    if part.startswith("split=")
                ),
                None,
            )
            if split_value is not None:
                frame = frame.with_columns(pl.lit(split_value).alias("split"))
        frames.append(frame)
    return pl.concat(frames, how="diagonal_relaxed")


def _attach_rule_interaction_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive compact rule aggregates after individual ``rule_*_hit`` columns exist."""
    rule_hit_columns = [
        column
        for column in frame.columns
        if column.startswith("rule_") and column.endswith("_hit")
    ]
    if not rule_hit_columns:
        return frame.with_columns(
            pl.lit(0).cast(pl.Int8).alias("any_rule_hit"),
            pl.lit(0).cast(pl.Int16).alias("rule_hit_count"),
        )
    hit_exprs = [pl.col(column).cast(pl.Int16) for column in rule_hit_columns]
    hit_count = pl.sum_horizontal(hit_exprs)
    return frame.with_columns(
        hit_count.alias("rule_hit_count"),
        (hit_count > 0).cast(pl.Int8).alias("any_rule_hit"),
    )


def build_pit_feature_dataset(
    input_root: Path,
    output_root: Path,
    *,
    rules_path: Path | None = None,
    feature_registry_path: Path = DEFAULT_FEATURE_REGISTRY_PATH,
    feature_engineering_config_path: Path = DEFAULT_FEATURE_ENGINEERING_CONFIG_PATH,
    feature_engineering_config: FeatureEngineeringConfig | None = None,
    overwrite: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = None,
) -> FeatureBuildSummary:
    """Create causal transaction features one complete event-date at a time.

    Input must be the prepared dataset produced by ingestion. Each event-date is
    fully scored before its next date is read, and the history state persists
    across dates. Optional start/end/max_dates filters support smoke pipelines
    without changing the formal full-protocol path.

    For FE experiments, write to a sidecar root such as
    ``artifacts/pit_features_fe_v2`` rather than overwriting formal
    ``artifacts/pit_features``.
    """
    if not input_root.is_dir():
        raise FileNotFoundError(f"Prepared input dataset does not exist: {input_root}")
    event_dates = _discover_event_dates(input_root)
    if start_date is not None:
        date.fromisoformat(start_date)
        event_dates = [value for value in event_dates if value >= start_date]
    if end_date is not None:
        date.fromisoformat(end_date)
        event_dates = [value for value in event_dates if value <= end_date]
    if max_dates is not None:
        if max_dates < 1:
            raise ValueError("max_dates must be positive when provided.")
        event_dates = event_dates[:max_dates]
    if not event_dates:
        raise ValueError("No event dates remain after applying smoke/date filters.")
    _prepare_output_root(output_root, overwrite=overwrite)

    rules = load_rules(rules_path) if rules_path is not None else []
    engineering_config = feature_engineering_config or load_feature_engineering_config(
        feature_engineering_config_path
    )
    feature_metadata = load_static_feature_metadata(feature_registry_path)
    feature_metadata.extend(rule_feature_metadata(rules, path=feature_registry_path))
    builder = PITFeatureBuilder(engineering_config)
    graph_statistics = CausalGraphStatisticsBuilder()
    total_rows = 0
    feature_column_count = 0
    total_rule_hits = 0

    for event_date in event_dates:
        partition = _read_event_date_partition(input_root, event_date)
        if partition.is_empty():
            raise ValueError(f"Discovered empty event-date partition: {event_date}")
        transformed = builder.transform_partition(partition)
        graph_features = graph_statistics.transform_partition(partition)
        transformed = transformed.join(
            graph_features,
            on="transaction_id",
            how="left",
        )
        rule_features, rule_hits = apply_rules(
            transformed,
            rules,
            as_of_date=date.fromisoformat(event_date),
        )
        if rule_features.height > 0:
            transformed = pl.concat(
                [transformed, rule_features],
                how="horizontal_extend",
            )
        for rule in rules:
            feature_name = f"rule_{rule.rule_id}_hit"
            if rule.active and feature_name not in transformed.columns:
                transformed = transformed.with_columns(pl.lit(0).alias(feature_name))
        transformed = _attach_rule_interaction_features(transformed)
        generated_feature_columns = set(transformed.columns).difference(
            set(CANONICAL.required_columns) | {"event_date", "split"}
        )
        validate_feature_metadata(generated_feature_columns, feature_metadata)
        if rule_hits:
            _write_rule_hits(output_root, event_date, rule_hits)
            total_rule_hits += len(rule_hits)
        split_values = transformed["split"].drop_nulls().unique().to_list()
        if len(split_values) != 1:
            raise ValueError(
                f"An event-date must have exactly one chronological split: {event_date}"
            )
        split = str(split_values[0])
        target_dir = output_root / f"event_date={event_date}" / f"split={split}"
        target_dir.mkdir(parents=True, exist_ok=False)
        parquet_frame = transformed.drop(["event_date", "split"], strict=False)
        parquet_frame.write_parquet(
            target_dir / "part-00000.parquet",
            compression="zstd",
        )
        total_rows += transformed.height
        feature_column_count = len(transformed.columns)

    config_paths: dict[str, Path] = {
        "feature_registry": feature_registry_path,
    }
    if feature_engineering_config is None:
        config_paths["feature_engineering"] = feature_engineering_config_path
    if rules_path is not None:
        config_paths["rules"] = rules_path
    manifest = create_run_manifest(
        output_dir=output_root,
        command="aml-build-pit-features",
        random_seed=0,
        input_paths={"prepared_dataset": input_root},
        config_paths=config_paths,
        metadata={
            "partition_count": len(event_dates),
            "configured_rule_count": len(rules),
            "rule_hit_count": total_rule_hits,
            "start_date": start_date,
            "end_date": end_date,
            "max_dates": max_dates,
            "engine": "polars",
            "feature_engineering_version": engineering_config.version,
            "feature_registry_version": feature_metadata[0].version,
        },
        filename="_run_manifest.json",
    )
    summary = FeatureBuildSummary(
        input_root=str(input_root),
        output_root=str(output_root),
        partition_count=len(event_dates),
        row_count=total_rows,
        feature_column_count=feature_column_count,
        configured_rule_count=len(rules),
        rule_hit_count=total_rule_hits,
        feature_registry_version=feature_metadata[0].version,
        event_date_min=event_dates[0],
        event_date_max=event_dates[-1],
        created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        run_id=manifest.run_id,
    )
    (output_root / "_feature_build_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_feature_registry(output_root / "_feature_registry.json", feature_metadata)
    return summary


def _write_rule_hits(output_root: Path, event_date: str, hits: list[RuleHit]) -> None:
    """Persist structured private evidence separately from model feature rows."""
    evidence_dir = output_root / "_rule_evidence"
    evidence_dir.mkdir(exist_ok=True)
    evidence_path = evidence_dir / f"event_date={event_date}.json"
    evidence_path.write_text(
        json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared Parquet dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Feature dataset root. Prefer a sidecar path such as "
            "artifacts/pit_features_fe_v2 for FE experiments."
        ),
    )
    parser.add_argument("--rules", type=Path, help="Versioned rule YAML; optional.")
    parser.add_argument(
        "--feature-registry",
        type=Path,
        default=DEFAULT_FEATURE_REGISTRY_PATH,
        help="Versioned metadata contract for generated features.",
    )
    parser.add_argument(
        "--feature-engineering-config",
        type=Path,
        default=DEFAULT_FEATURE_ENGINEERING_CONFIG_PATH,
        help="Typology-proxy constants (high-risk locations, thresholds, payment types).",
    )
    parser.add_argument(
        "--start-date",
        help="Inclusive ISO date filter for smoke/subset builds.",
    )
    parser.add_argument(
        "--end-date",
        help="Inclusive ISO date filter for smoke/subset builds.",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        help="Keep only the first N filtered event dates (chronological).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_pit_feature_dataset(
        args.input,
        args.output,
        rules_path=args.rules,
        feature_registry_path=args.feature_registry,
        feature_engineering_config_path=args.feature_engineering_config,
        overwrite=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dates=args.max_dates,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
