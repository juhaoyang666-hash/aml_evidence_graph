#!/usr/bin/env python3
"""Stress-test degree signals with month-matched negatives by positive typology.

Normal transactions do not have a suspicious typology. Each typology comparison
therefore keeps positives from that typology and samples normal negatives from
the same event months at a fixed ratio. Only aggregate validation metrics are
emitted; test data and identifiers are never written.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_evidence_graph.data.contract import CANONICAL

MIN_DEGREE = "graph_endpoint_min_historical_degree"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--score-artifact",
        action="append",
        required=True,
        metavar="NAME=PARQUET",
        help="Repeat for each frozen validation score artifact.",
    )
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_score_artifacts(values: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("Score artifacts must use NAME=PARQUET syntax.")
        if name in artifacts:
            raise ValueError(f"Duplicate score artifact name: {name}")
        artifacts[name] = Path(raw_path)
    return artifacts


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) != 2:
        raise ValueError("Matched comparisons require both label classes.")
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _sample_month_matched(
    frame: pl.DataFrame,
    *,
    typology: str | None,
    negative_ratio: int,
    random_seed: int,
) -> tuple[pl.DataFrame, list[dict[str, int | str]]]:
    positive = frame.filter(pl.col(CANONICAL.is_laundering) == 1)
    if typology is not None:
        positive = positive.filter(pl.col(CANONICAL.laundering_type) == typology)
    negative = frame.filter(pl.col(CANONICAL.is_laundering) == 0)
    rng = np.random.default_rng(random_seed)
    pieces = [positive]
    month_rows: list[dict[str, int | str]] = []
    for month_frame in positive.partition_by("event_month", maintain_order=True):
        month = str(month_frame["event_month"][0])
        pool = negative.filter(pl.col("event_month") == month)
        requested = month_frame.height * negative_ratio
        sampled_count = min(requested, pool.height)
        indices = rng.choice(pool.height, size=sampled_count, replace=False)
        pieces.append(pool[indices])
        month_rows.append(
            {
                "event_month": month,
                "positive_count": month_frame.height,
                "negative_count": sampled_count,
            }
        )
    return pl.concat(pieces, how="vertical"), month_rows


def _comparison(
    frame: pl.DataFrame,
    *,
    score_columns: tuple[str, ...],
    typology: str | None,
    negative_ratio: int,
    random_seed: int,
) -> dict[str, object]:
    matched, month_rows = _sample_month_matched(
        frame,
        typology=typology,
        negative_ratio=negative_ratio,
        random_seed=random_seed,
    )
    labels = matched[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    result: dict[str, object] = {
        "typology": typology or "__all_positive_typologies__",
        "row_count": matched.height,
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "realized_positive_rate": float(labels.mean()),
        "month_strata": month_rows,
        "minimum_degree_proxy": _metrics(labels, matched[MIN_DEGREE].to_numpy()),
        "models": {},
    }
    result["models"] = {
        column: _metrics(labels, matched[column].to_numpy()) for column in score_columns
    }
    return result


def main() -> None:
    args = parse_args()
    if args.negative_ratio <= 0:
        raise ValueError("--negative-ratio must be positive.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite stress-test output: {args.output}")
    score_artifacts = _parse_score_artifacts(args.score_artifact)
    dataset = ds.dataset(args.features, format="parquet", partitioning="hive")
    frame = pl.from_arrow(
        dataset.to_table(
            filter=ds.field("split") == "validation",
            columns=[
                CANONICAL.transaction_id,
                CANONICAL.event_ts,
                CANONICAL.is_laundering,
                CANONICAL.laundering_type,
                MIN_DEGREE,
            ],
        )
    ).with_columns(
        pl.col(CANONICAL.event_ts).dt.strftime("%Y-%m").alias("event_month")
    )
    for name, path in score_artifacts.items():
        scores = pl.read_parquet(path).select(
            CANONICAL.transaction_id,
            CANONICAL.is_laundering,
            pl.col("graphsage").alias(name),
        )
        before = frame.height
        frame = frame.join(
            scores,
            on=[CANONICAL.transaction_id, CANONICAL.is_laundering],
            how="inner",
            validate="1:1",
        )
        if frame.height != before or frame.height != scores.height:
            raise ValueError(f"Score artifact does not align one-to-one: {name}")
    score_columns = tuple(score_artifacts)
    typologies = sorted(
        frame.filter(pl.col(CANONICAL.is_laundering) == 1)[CANONICAL.laundering_type]
        .drop_nulls()
        .unique()
        .cast(pl.Utf8)
        .to_list()
    )
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_scope": "validation_only",
        "test_split_read": False,
        "emits_identifiers": False,
        "protocol": {
            "negative_ratio": args.negative_ratio,
            "random_seed": args.random_seed,
            "negative_label": "normal transactions",
            "matching": "event_month within each positive-typology comparison",
            "typology_boundary": (
                "Typology defines the positive stratum only; normal negatives do not have "
                "a suspicious typology. Negative samples may be reused across comparisons."
            ),
            "score_artifacts": {name: str(path) for name, path in score_artifacts.items()},
        },
        "aggregate": _comparison(
            frame,
            score_columns=score_columns,
            typology=None,
            negative_ratio=args.negative_ratio,
            random_seed=args.random_seed,
        ),
        "by_positive_typology": [
            _comparison(
                frame,
                score_columns=score_columns,
                typology=typology,
                negative_ratio=args.negative_ratio,
                random_seed=args.random_seed + index + 1,
            )
            for index, typology in enumerate(typologies)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "typology_count": len(typologies),
                "test_split_read": False,
            }
        )
    )


if __name__ == "__main__":
    main()
