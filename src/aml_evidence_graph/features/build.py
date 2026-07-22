"""Build a date-partitioned Point-in-Time feature dataset from tokenized Parquet."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from aml_evidence_graph.data.contract import CANONICAL
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
            f"No Hive event_date partitions found beneath tokenized dataset: {input_root}"
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


def build_pit_feature_dataset(
    input_root: Path,
    output_root: Path,
    *,
    rules_path: Path | None = None,
    feature_registry_path: Path = DEFAULT_FEATURE_REGISTRY_PATH,
    overwrite: bool = False,
) -> FeatureBuildSummary:
    """Create causal transaction features one complete event-date at a time.

    Input must be the tokenized dataset produced by the ingestion module. Each
    event-date is fully scored before its next date is read, and the history
    state persists across dates. The implementation never joins a transaction
    to a future event and does not use labels during feature calculation.
    """
    if not input_root.is_dir():
        raise FileNotFoundError(f"Tokenized input dataset does not exist: {input_root}")
    event_dates = _discover_event_dates(input_root)
    _prepare_output_root(output_root, overwrite=overwrite)

    rules = load_rules(rules_path) if rules_path is not None else []
    feature_metadata = load_static_feature_metadata(feature_registry_path)
    feature_metadata.extend(rule_feature_metadata(rules, path=feature_registry_path))
    source = ds.dataset(input_root, format="parquet", partitioning="hive")
    builder = PITFeatureBuilder()
    graph_statistics = CausalGraphStatisticsBuilder()
    total_rows = 0
    feature_column_count = 0
    total_rule_hits = 0

    for event_date in event_dates:
        partition = source.to_table(
            filter=ds.field("event_date") == event_date,
        ).to_pandas()
        if partition.empty:
            raise ValueError(f"Discovered empty event-date partition: {event_date}")
        transformed = builder.transform_partition(partition)
        graph_features = graph_statistics.transform_partition(partition)
        transformed = transformed.merge(
            graph_features,
            on="transaction_id",
            how="left",
            validate="one_to_one",
        )
        rule_features, rule_hits = apply_rules(
            transformed,
            rules,
            as_of_date=date.fromisoformat(event_date),
        )
        if not rule_features.empty:
            transformed = pd.concat([transformed, rule_features], axis=1)
        for rule in rules:
            feature_name = f"rule_{rule.rule_id}_hit"
            if rule.active and feature_name not in transformed:
                transformed[feature_name] = 0
        generated_feature_columns = set(transformed.columns).difference(
            set(CANONICAL.required_columns) | {"event_date", "split"}
        )
        validate_feature_metadata(generated_feature_columns, feature_metadata)
        if rule_hits:
            _write_rule_hits(output_root, event_date, rule_hits)
            total_rule_hits += len(rule_hits)
        split_values = transformed["split"].dropna().unique()
        if len(split_values) != 1:
            raise ValueError(
                f"An event-date must have exactly one chronological split: {event_date}"
            )
        split = str(split_values[0])
        target_dir = output_root / f"event_date={event_date}" / f"split={split}"
        target_dir.mkdir(parents=True, exist_ok=False)
        parquet_frame = transformed.drop(columns=["event_date", "split"])
        pq.write_table(
            pa.Table.from_pandas(parquet_frame, preserve_index=False),
            target_dir / "part-00000.parquet",
            compression="zstd",
        )
        total_rows += len(transformed)
        feature_column_count = len(transformed.columns)

    manifest = create_run_manifest(
        output_dir=output_root,
        command="aml-build-pit-features",
        random_seed=0,
        input_paths={"tokenized_dataset": input_root},
        config_paths={
            "feature_registry": feature_registry_path,
            **({"rules": rules_path} if rules_path is not None else {}),
        },
        metadata={
            "partition_count": len(event_dates),
            "configured_rule_count": len(rules),
            "rule_hit_count": total_rule_hits,
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
        created_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
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
    parser.add_argument("--input", type=Path, required=True, help="Tokenized Parquet dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Private feature dataset.")
    parser.add_argument("--rules", type=Path, help="Versioned rule YAML; optional.")
    parser.add_argument(
        "--feature-registry",
        type=Path,
        default=DEFAULT_FEATURE_REGISTRY_PATH,
        help="Versioned metadata contract for generated features.",
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
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
