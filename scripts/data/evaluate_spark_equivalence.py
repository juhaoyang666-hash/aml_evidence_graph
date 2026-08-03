#!/usr/bin/env python3
"""Evaluate Spark representative PIT features against the official PIT output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import polars as pl

from aml_evidence_graph.features.spark_replay import REPRESENTATIVE_FEATURES


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _feature_comparison(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, object]:
    if left["transaction_id"].n_unique() != left.height:
        raise ValueError("Spark output contains duplicate transaction IDs.")
    if right["transaction_id"].n_unique() != right.height:
        raise ValueError("Reference output contains duplicate transaction IDs.")
    joined = left.join(right, on="transaction_id", how="inner", suffix="_reference")
    if joined.height != left.height or joined.height != right.height:
        raise ValueError("Spark and reference transaction populations do not align.")
    features: dict[str, object] = {}
    for feature in REPRESENTATIVE_FEATURES:
        difference = (
            pl.col(feature).cast(pl.Float64)
            - pl.col(f"{feature}_reference").cast(pl.Float64)
        ).abs()
        features[feature] = joined.select(
            (difference <= 1e-9).mean().alias("match_rate"),
            difference.max().alias("max_absolute_difference"),
            difference.mean().alias("mean_absolute_difference"),
        ).row(0, named=True)
    return {"rows": joined.height, "features": features}


def _read_features(path: Path, *, recursive: bool = False) -> pl.DataFrame:
    pattern = path / "**" / "*.parquet" if recursive else path / "*.parquet"
    return pl.read_parquet(
        pattern,
        columns=["transaction_id", *REPRESENTATIVE_FEATURES],
        hive_partitioning=recursive,
    )


def _render_markdown(payload: dict[str, object]) -> str:
    full = payload["full_replay"]
    incremental = payload["incremental_replay"]
    comparison = payload["full_equivalence"]
    assert isinstance(full, dict)
    assert isinstance(incremental, dict)
    assert isinstance(comparison, dict)
    features = comparison["features"]
    assert isinstance(features, dict)
    lines = [
        "# Spark 特征等价评估",
        "",
        f"> {payload['scope']}",
        "",
        f"- 结果：**{'通过' if payload['passed'] else '未通过'}**",
        f"- 全量重放：扫描/输出 {full['input_rows_scanned']:,}/{full['output_rows']:,} 行，"
        f"{full['duration_seconds']:.2f} 秒，峰值进程树内存 "
        f"{full['peak_process_tree_rss_mb']:.1f} MB",
        f"- 增量重放：扫描/输出 {incremental['input_rows_scanned']:,}/"
        f"{incremental['output_rows']:,} 行，{incremental['duration_seconds']:.2f} 秒，"
        f"峰值 {incremental['peak_process_tree_rss_mb']:.1f} MB",
        f"- Spark：`{full['master']}`，shuffle partitions={full['shuffle_partitions']}，"
        f"physical Exchange={full['exchange_count']}，JVM heap={full['jvm_max_heap_mb']:.0f} MB",
        f"- 写出：`{full['write_mode']}`，{full['output_bytes'] / 1024 / 1024:.1f} MB",
        "- 对齐：`transaction_id` 显式等值连接；没有普通笛卡尔积。",
        "",
        "| PIT 特征 | Match rate | 最大绝对误差 |",
        "|---|---:|---:|",
    ]
    for feature, raw_values in features.items():
        assert isinstance(raw_values, dict)
        lines.append(
            f"| {feature} | {raw_values['match_rate']:.6f} | "
            f"{raw_values['max_absolute_difference']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Windows 缺少 Hadoop native 写入组件，本机采用 Spark 计算后 Arrow driver 写出；",
            "窗口聚合与 shuffle 由 Spark 执行。该结果不代表生产集群 SLA 或原生 writer 性能。",
            "当前只验证 5 个代表性历史计数特征，不替代完整官方 PIT 流水线。",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_resource_curve(
    path: Path,
    full_summary: dict[str, object],
    incremental_summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_scope",
        "elapsed_seconds",
        "process_tree_rss_mb",
        "process_tree_cpu_time_seconds",
        "process_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for scope, summary in (
            ("full", full_summary),
            ("incremental", incremental_summary),
        ):
            samples = summary.get("resource_samples")
            if not isinstance(samples, list):
                raise ValueError(f"{scope} summary has no resource_samples list")
            for sample in samples:
                if not isinstance(sample, dict):
                    raise ValueError(f"{scope} resource sample must be an object")
                writer.writerow({"run_scope": scope, **sample})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--incremental-output", type=Path, required=True)
    parser.add_argument("--incremental-summary", type=Path, required=True)
    parser.add_argument("--target-event-date", required=True)
    parser.add_argument(
        "--scope",
        default="Windows 本地 Spark 重放；不是生产集群 SLA。",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/spark_replay_evaluation/metrics.json")
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/spark_replay_evaluation/equivalence_report.md"),
    )
    parser.add_argument(
        "--resource-csv",
        type=Path,
        default=Path("artifacts/spark_replay_evaluation/resource_curve.csv"),
    )
    args = parser.parse_args()

    spark = _read_features(args.spark_output)
    reference = _read_features(args.reference, recursive=True)
    full_comparison = _feature_comparison(spark, reference)
    incremental = _read_features(args.incremental_output)
    full_target = pl.read_parquet(args.spark_output / "*.parquet").filter(
        pl.col("event_date").cast(pl.String) == args.target_event_date
    ).select(["transaction_id", *REPRESENTATIVE_FEATURES])
    incremental_comparison = _feature_comparison(incremental, full_target)
    full_summary = _load_object(args.full_summary)
    incremental_summary = _load_object(args.incremental_summary)
    match_rates = [
        float(item["match_rate"])
        for item in full_comparison["features"].values()
    ]
    incremental_match_rates = [
        float(item["match_rate"])
        for item in incremental_comparison["features"].values()
    ]
    payload = {
        "schema_version": "1.1",
        "scope": args.scope,
        "feature_count": len(REPRESENTATIVE_FEATURES),
        "full_replay": full_summary,
        "incremental_replay": incremental_summary,
        "full_equivalence": full_comparison,
        "incremental_equivalence": incremental_comparison,
        "minimum_feature_match_rate": min(match_rates),
        "minimum_incremental_match_rate": min(incremental_match_rates),
        "passed": min(match_rates) == 1.0 and min(incremental_match_rates) == 1.0,
        "join_policy": "explicit equality join on transaction_id; no Cartesian join",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(payload), encoding="utf-8")
    _write_resource_curve(args.resource_csv, full_summary, incremental_summary)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
