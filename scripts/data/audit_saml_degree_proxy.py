#!/usr/bin/env python3
"""Audit SAML-D minimum historical endpoint degree as a synthetic-label proxy.

The audit is deliberately aggregate-only and reads train/validation partitions.
It never emits transaction IDs or account identifiers and never reads test.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_evidence_graph.data.contract import CANONICAL

FEATURE = "graph_endpoint_min_historical_degree"
ALLOWED_SPLITS = ("train", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _distribution(values: pl.Series) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "q25": float(values.quantile(0.25, interpolation="linear")),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75, interpolation="linear")),
        "q90": float(values.quantile(0.90, interpolation="linear")),
        "zero_rate": float((values == 0).mean()),
    }


def _split_summary(frame: pl.DataFrame) -> dict[str, object]:
    labels = frame[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    values = frame[FEATURE].cast(pl.Float64).to_numpy()
    positive = frame.filter(pl.col(CANONICAL.is_laundering) == 1)[FEATURE]
    negative = frame.filter(pl.col(CANONICAL.is_laundering) == 0)[FEATURE]
    return {
        "row_count": frame.height,
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "univariate_pr_auc": float(average_precision_score(labels, values)),
        "univariate_roc_auc": float(roc_auc_score(labels, values)),
        "positive_distribution": _distribution(positive),
        "negative_distribution": _distribution(negative),
    }


def _grouped_rows(frame: pl.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    return (
        frame.group_by(columns)
        .agg(
            pl.len().alias("row_count"),
            pl.col(CANONICAL.is_laundering).sum().alias("positive_count"),
            pl.col(FEATURE).mean().alias("degree_mean"),
            pl.col(FEATURE).median().alias("degree_median"),
            pl.col(FEATURE).quantile(0.25).alias("degree_q25"),
            pl.col(FEATURE).quantile(0.75).alias("degree_q75"),
            (pl.col(FEATURE) == 0).mean().alias("degree_zero_rate"),
        )
        .with_columns((pl.col("positive_count") / pl.col("row_count")).alias("positive_rate"))
        .sort(columns)
        .to_dicts()
    )


def _monthly_univariate_metrics(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for monthly in frame.partition_by(["split", "event_month"], maintain_order=True):
        labels = monthly[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
        values = monthly[FEATURE].cast(pl.Float64).to_numpy()
        if len(np.unique(labels)) != 2:
            raise ValueError("Every audited month must contain both label classes.")
        rows.append(
            {
                "split": str(monthly["split"][0]),
                "event_month": str(monthly["event_month"][0]),
                "row_count": monthly.height,
                "positive_count": int(labels.sum()),
                "univariate_pr_auc": float(average_precision_score(labels, values)),
                "univariate_roc_auc": float(roc_auc_score(labels, values)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite degree-proxy audit: {args.output}")
    if not args.features.is_dir():
        raise FileNotFoundError(f"Feature root does not exist: {args.features}")

    frame = (
        pl.scan_parquet(
            str(args.features / "event_date=*" / "split=*" / "*.parquet"),
            hive_partitioning=True,
        )
        .filter(pl.col("split").is_in(ALLOWED_SPLITS))
        .select(
            CANONICAL.event_ts,
            CANONICAL.is_laundering,
            CANONICAL.laundering_type,
            FEATURE,
            "split",
        )
        .with_columns(
            pl.col(CANONICAL.event_ts)
            .cast(pl.Datetime(time_zone="UTC"), strict=True)
            .dt.strftime("%Y-%m")
            .alias("event_month")
        )
        .collect()
    )
    if set(frame["split"].unique()) != set(ALLOWED_SPLITS):
        raise ValueError("Degree-proxy audit requires both train and validation partitions.")
    if frame[FEATURE].null_count() or not np.isfinite(frame[FEATURE].to_numpy()).all():
        raise ValueError(f"{FEATURE} must be complete and finite.")

    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_root": str(args.features),
        "feature": FEATURE,
        "selection_scope": "train_validation_only",
        "test_split_read": False,
        "emits_identifiers": False,
        "split_summary": {
            split: _split_summary(frame.filter(pl.col("split") == split))
            for split in ALLOWED_SPLITS
        },
        "typology_summary": _grouped_rows(
            frame,
            ["split", CANONICAL.laundering_type],
        ),
        "monthly_label_summary": _grouped_rows(
            frame,
            ["split", "event_month", CANONICAL.is_laundering],
        ),
        "monthly_univariate_metrics": _monthly_univariate_metrics(frame),
        "interpretation_boundary": (
            "Association does not prove code-level leakage. SAML-D upstream publishes a fixed "
            "generated dataset and methodology, but its public GitHub repository does not "
            "contain the transaction-generator source code."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": frame.height,
                "test_split_read": False,
            }
        )
    )


if __name__ == "__main__":
    main()
