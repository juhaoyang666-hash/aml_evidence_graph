#!/usr/bin/env python3
"""Evaluate Spark representative PIT features against the official PIT output."""

from __future__ import annotations

import argparse
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
        values = joined.select(
            (difference <= 1e-9).mean().alias("match_rate"),
            difference.max().alias("max_absolute_difference"),
            difference.mean().alias("mean_absolute_difference"),
        ).row(0, named=True)
        features[feature] = values
    return {"rows": joined.height, "features": features}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark-output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--incremental-output", type=Path, required=True)
    parser.add_argument("--incremental-summary", type=Path, required=True)
    parser.add_argument("--target-event-date", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/spark_replay_evaluation/metrics.json")
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/spark_replay_evaluation/equivalence_report.md"),
    )
    args = parser.parse_args()
    columns = ["transaction_id", *REPRESENTATIVE_FEATURES]
    spark = pl.read_parquet(args.spark_output / "*.parquet").select(columns)
    reference = pl.read_parquet(
        args.reference / "**" / "*.parquet", hive_partitioning=True
    ).select(columns)
    full_comparison = _feature_comparison(spark, reference)
    incremental = pl.read_parquet(args.incremental_output / "*.parquet").select(columns)
    full_target = pl.read_parquet(args.spark_output / "*.parquet").filter(
        pl.col("event_date").cast(pl.String) == args.target_event_date
    ).select(columns)
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
        "schema_version": "1.0",
        "scope": "Local Spark smoke replay; not a production Spark cluster benchmark.",
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
    lines = [
        "# Spark 特征等价评估",
        "",
        "> 本报告是 Windows 本地 Spark smoke 重放，不是生产集群 SLA。",
        "",
        f"- 结果：**{'通过' if payload['passed'] else '未通过'}**",
        f"- 全量重放：扫描/输出 {full_summary['input_rows_scanned']:,}/"
        f"{full_summary['output_rows']:,} 行，"
        f"{full_summary['duration_seconds']:.2f} 秒，峰值进程树内存 "
        f"{full_summary['peak_process_tree_rss_mb']:.1f} MB",
        f"- 增量重放：扫描/输出 {incremental_summary['input_rows_scanned']:,}/"
        f"{incremental_summary['output_rows']:,} 行，"
        f"{incremental_summary['duration_seconds']:.2f} 秒，峰值 "
        f"{incremental_summary['peak_process_tree_rss_mb']:.1f} MB",
        f"- Spark：`{full_summary['master']}`，shuffle partitions="
        f"{full_summary['shuffle_partitions']}，physical Exchange="
        f"{full_summary['exchange_count']}",
        "- 对齐：`transaction_id` 显式等值连接；没有普通笛卡尔积。",
        "",
        "| PIT 特征 | Match rate | 最大绝对误差 |",
        "|---|---:|---:|",
    ]
    for feature, values in full_comparison["features"].items():
        lines.append(
            f"| {feature} | {values['match_rate']:.6f} | "
            f"{values['max_absolute_difference']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Windows 缺少 Hadoop native 写入组件，本机采用 Spark 计算后 Arrow driver 写出；",
            "窗口聚合与 shuffle 由 Spark 执行。Linux/集群环境使用原生 Spark Parquet writer。",
            "当前记录物理 Exchange 节点数，不宣称生产集群 shuffle bytes。",
            "该结果只覆盖 5 个代表性历史计数特征和 smoke 数据，不替代全量官方 PIT 流水线。",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
