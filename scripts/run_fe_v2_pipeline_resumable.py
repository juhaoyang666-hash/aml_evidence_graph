"""Resumable fe_v2 pipeline: PIT features -> table CatBoost only (no GraphSAGE).

Writes status to artifacts/logs/fe_v2_pipeline_status.json and logs under artifacts/logs/.
PIT resume: if output exists without a complete summary, replay all dates through the
causal builders (to rebuild history) and write only missing partitions.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.build import (
    FeatureBuildSummary,
    _attach_rule_interaction_features,
    _discover_event_dates,
    _read_event_date_partition,
    _write_rule_hits,
)
from aml_evidence_graph.features.engineering_config import load_feature_engineering_config
from aml_evidence_graph.features.graph_stats import CausalGraphStatisticsBuilder
from aml_evidence_graph.features.pit import PITFeatureBuilder
from aml_evidence_graph.features.registry import (
    DEFAULT_FEATURE_REGISTRY_PATH,
    load_static_feature_metadata,
    rule_feature_metadata,
    validate_feature_metadata,
    write_feature_registry,
)
from aml_evidence_graph.rules.engine import apply_rules, load_rules
from aml_evidence_graph.tracking.run import create_run_manifest

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = REPO / "artifacts" / "logs" / "fe_v2_pipeline_status.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": _utc_now()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _partition_complete(output_root: Path, event_date: str) -> bool:
    date_dir = output_root / f"event_date={event_date}"
    if not date_dir.is_dir():
        return False
    return any(date_dir.rglob("*.parquet"))


def build_pit_resumable(
    *,
    input_root: Path,
    output_root: Path,
    rules_path: Path,
    feature_registry_path: Path,
    feature_engineering_config_path: Path,
    force_rebuild: bool = False,
) -> FeatureBuildSummary:
    event_dates = _discover_event_dates(input_root)
    if not event_dates:
        raise ValueError(f"No event dates in {input_root}")

    summary_path = output_root / "_feature_build_summary.json"
    if (
        not force_rebuild
        and summary_path.is_file()
        and all(_partition_complete(output_root, d) for d in event_dates)
    ):
        return FeatureBuildSummary(**json.loads(summary_path.read_text(encoding="utf-8")))

    if force_rebuild and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Drop incomplete trailing partitions so we rewrite them cleanly.
    for event_date in event_dates:
        date_dir = output_root / f"event_date={event_date}"
        if date_dir.is_dir() and not _partition_complete(output_root, event_date):
            shutil.rmtree(date_dir)

    rules = load_rules(rules_path)
    engineering_config = load_feature_engineering_config(feature_engineering_config_path)
    feature_metadata = load_static_feature_metadata(feature_registry_path)
    feature_metadata.extend(rule_feature_metadata(rules, path=feature_registry_path))
    builder = PITFeatureBuilder(engineering_config)
    graph_statistics = CausalGraphStatisticsBuilder()
    total_rows = 0
    feature_column_count = 0
    total_rule_hits = 0
    written = 0
    replayed = 0

    for event_date in event_dates:
        partition = _read_event_date_partition(input_root, event_date)
        if partition.is_empty():
            raise ValueError(f"Empty partition: {event_date}")
        transformed = builder.transform_partition(partition)
        graph_features = graph_statistics.transform_partition(partition)
        transformed = transformed.join(graph_features, on="transaction_id", how="left")
        rule_features, rule_hits = apply_rules(
            transformed, rules, as_of_date=date.fromisoformat(event_date)
        )
        if rule_features.height > 0:
            transformed = pl.concat([transformed, rule_features], how="horizontal_extend")
        for rule in rules:
            feature_name = f"rule_{rule.rule_id}_hit"
            if rule.active and feature_name not in transformed.columns:
                transformed = transformed.with_columns(pl.lit(0).alias(feature_name))
        transformed = _attach_rule_interaction_features(transformed)
        generated = set(transformed.columns).difference(
            set(CANONICAL.required_columns) | {"event_date", "split"}
        )
        validate_feature_metadata(generated, feature_metadata)

        already = _partition_complete(output_root, event_date)
        if already:
            # Still count rows from existing write for summary consistency.
            existing = list((output_root / f"event_date={event_date}").rglob("*.parquet"))
            rows = sum(pl.scan_parquet(str(p)).select(pl.len()).collect().item() for p in existing)
            total_rows += int(rows)
            feature_column_count = len(transformed.columns)
            replayed += 1
            continue

        if rule_hits:
            _write_rule_hits(output_root, event_date, rule_hits)
            total_rule_hits += len(rule_hits)
        split_values = transformed["split"].drop_nulls().unique().to_list()
        if len(split_values) != 1:
            raise ValueError(f"Expected one split for {event_date}, got {split_values}")
        split = str(split_values[0])
        target_dir = output_root / f"event_date={event_date}" / f"split={split}"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=False)
        parquet_frame = transformed.drop(["event_date", "split"], strict=False)
        parquet_frame.write_parquet(target_dir / "part-00000.parquet", compression="zstd")
        total_rows += transformed.height
        feature_column_count = len(transformed.columns)
        written += 1

    manifest = create_run_manifest(
        output_dir=output_root,
        command="aml-build-pit-features-fe-v2-resumable",
        random_seed=0,
        input_paths={"prepared_dataset": input_root},
        config_paths={
            "feature_registry": feature_registry_path,
            "feature_engineering": feature_engineering_config_path,
            "rules": rules_path,
        },
        metadata={
            "partition_count": len(event_dates),
            "configured_rule_count": len(rules),
            "rule_hit_count": total_rule_hits,
            "engine": "polars",
            "feature_engineering_version": engineering_config.version,
            "feature_registry_version": feature_metadata[0].version,
            "resumed_replayed_partitions": replayed,
            "written_partitions": written,
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
        created_at_utc=_utc_now(),
        run_id=manifest.run_id,
    )
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_feature_registry(output_root / "_feature_registry.json", feature_metadata)
    return summary


def run_table(features: Path, output: Path, model_config: Path, overwrite: bool) -> int:
    from aml_evidence_graph.training import table_baseline as tb

    # Prefer CLI entry for consistent logging/artifacts.
    argv = [
        "--features",
        str(features),
        "--output",
        str(output),
        "--model-config",
        str(model_config),
    ]
    if overwrite:
        argv.append("--overwrite")
    old = sys.argv
    try:
        sys.argv = ["table_baseline", *argv]
        tb.main()
    finally:
        sys.argv = old
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable fe_v2 PIT + CatBoost pipeline")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--prepared", type=Path, default=REPO / "artifacts" / "prepared_transactions")
    parser.add_argument("--pit-output", type=Path, default=REPO / "artifacts" / "pit_features_fe_v2")
    parser.add_argument("--table-output", type=Path, default=REPO / "artifacts" / "table_baseline_fe_v2")
    parser.add_argument("--rules", type=Path, default=REPO / "configs" / "rules" / "default.yaml")
    parser.add_argument("--feature-registry", type=Path, default=REPO / "configs" / "features.yaml")
    parser.add_argument(
        "--feature-engineering-config",
        type=Path,
        default=REPO / "configs" / "feature_engineering.yaml",
    )
    parser.add_argument("--model-config", type=Path, default=REPO / "configs" / "models.yaml")
    parser.add_argument("--force-rebuild-pit", action="store_true")
    parser.add_argument("--skip-table", action="store_true")
    parser.add_argument("--overwrite-table", action="store_true", default=True)
    args = parser.parse_args()

    status: dict = {
        "pipeline": "fe_v2_pit_table",
        "baseline_reference_pr_auc": 0.8092,
        "baseline_note": "docs/实验结果.md CatBoost primary test PR-AUC",
        "steps": {},
    }
    if args.status.is_file():
        try:
            status = json.loads(args.status.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    status.setdefault("steps", {})
    status["baseline_reference_pr_auc"] = 0.8092

    # Step 1: PIT
    pit_step = status["steps"].setdefault("01_pit_fe_v2", {})
    metrics_ok = (args.pit_output / "_feature_build_summary.json").is_file()
    if pit_step.get("complete") and metrics_ok and not args.force_rebuild_pit:
        print(f"[skip] PIT already complete: {args.pit_output}", flush=True)
    else:
        pit_step.update({"status": "running", "started_at": _utc_now(), "complete": False})
        _write_status(args.status, status)
        try:
            summary = build_pit_resumable(
                input_root=args.prepared,
                output_root=args.pit_output,
                rules_path=args.rules,
                feature_registry_path=args.feature_registry,
                feature_engineering_config_path=args.feature_engineering_config,
                force_rebuild=args.force_rebuild_pit,
            )
            pit_step.update(
                {
                    "status": "complete",
                    "complete": True,
                    "finished_at": _utc_now(),
                    "summary": asdict(summary),
                    "error": None,
                }
            )
            _write_status(args.status, status)
            print(json.dumps(asdict(summary), ensure_ascii=False), flush=True)
        except Exception as exc:
            pit_step.update(
                {
                    "status": "failed",
                    "complete": False,
                    "finished_at": _utc_now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            _write_status(args.status, status)
            raise

    if args.skip_table:
        return 0

    # Step 2: table
    table_step = status["steps"].setdefault("02_table_fe_v2", {})
    table_metrics = args.table_output / "metrics.json"
    if table_step.get("complete") and table_metrics.is_file() and not args.overwrite_table:
        print(f"[skip] table already complete: {args.table_output}", flush=True)
        return 0

    table_step.update({"status": "running", "started_at": _utc_now(), "complete": False})
    _write_status(args.status, status)
    try:
        run_table(args.pit_output, args.table_output, args.model_config, overwrite=True)
        metrics = None
        if table_metrics.is_file():
            metrics = json.loads(table_metrics.read_text(encoding="utf-8"))
        test_prauc = None
        if isinstance(metrics, dict):
            tm = metrics.get("test_metrics") or {}
            if isinstance(tm, dict):
                cat = tm.get("catboost") if isinstance(tm.get("catboost"), dict) else tm
                if isinstance(cat, dict):
                    test_prauc = cat.get("pr_auc") or cat.get("average_precision") or cat.get("prauc")
                # nested model map
                for key, val in tm.items():
                    if isinstance(val, dict) and ("pr_auc" in val or "average_precision" in val):
                        if "catboost" in str(key).lower() or test_prauc is None:
                            test_prauc = val.get("pr_auc") or val.get("average_precision")
        table_step.update(
            {
                "status": "complete",
                "complete": True,
                "finished_at": _utc_now(),
                "output_dir": str(args.table_output),
                "test_pr_auc": test_prauc,
                "baseline_pr_auc": 0.8092,
                "delta_vs_baseline": None if test_prauc is None else float(test_prauc) - 0.8092,
                "error": None,
            }
        )
        _write_status(args.status, status)
        print(
            json.dumps(
                {
                    "test_pr_auc": test_prauc,
                    "baseline_pr_auc": 0.8092,
                    "delta_vs_baseline": table_step.get("delta_vs_baseline"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        table_step.update(
            {
                "status": "failed",
                "complete": False,
                "finished_at": _utc_now(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_status(args.status, status)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

