"""Compare two aligned score artifacts with paired stratified bootstrap."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from aml_evidence_graph.evaluation.monitoring import paired_bootstrap_ranking_differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-column", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-column", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = pl.read_parquet(args.candidate).select(
        "transaction_id",
        "is_laundering",
        pl.col(args.candidate_column).alias("candidate_score"),
    )
    baseline = pl.read_parquet(args.baseline).select(
        "transaction_id",
        "is_laundering",
        pl.col(args.baseline_column).alias("baseline_score"),
    )
    if candidate["transaction_id"].is_duplicated().any():
        raise ValueError("Candidate artifact has duplicate transaction IDs.")
    if baseline["transaction_id"].is_duplicated().any():
        raise ValueError("Baseline artifact has duplicate transaction IDs.")
    aligned = candidate.join(
        baseline,
        on="transaction_id",
        how="inner",
        suffix="_baseline",
        validate="1:1",
    )
    if aligned.height != candidate.height or aligned.height != baseline.height:
        raise ValueError("Candidate and baseline transaction IDs are not identical.")
    label_mismatches = aligned.filter(
        pl.col("is_laundering") != pl.col("is_laundering_baseline")
    ).height
    if label_mismatches:
        raise ValueError(f"Candidate and baseline labels differ on {label_mismatches} rows.")
    differences = paired_bootstrap_ranking_differences(
        aligned["is_laundering"],
        aligned["candidate_score"],
        aligned["baseline_score"],
        iterations=args.iterations,
        random_seed=args.random_seed,
    )
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "paired candidate-minus-baseline stratified bootstrap",
        "candidate": str(args.candidate),
        "candidate_column": args.candidate_column,
        "baseline": str(args.baseline),
        "baseline_column": args.baseline_column,
        "row_count": aligned.height,
        "positive_count": int(aligned["is_laundering"].sum()),
        "random_seed": args.random_seed,
        "differences": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
