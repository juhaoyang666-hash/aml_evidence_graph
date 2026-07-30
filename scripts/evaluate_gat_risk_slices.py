#!/usr/bin/env python3
"""Evaluate a frozen GAT score artifact on pre-declared operational slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.dataset as ds

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evaluation.monitoring import (
    categorical_slice_report,
    new_account_slice_report,
    typology_slice_report,
)

DEGREE_COLUMNS = (
    "graph_sender_historical_out_degree",
    "graph_sender_historical_in_degree",
    "graph_receiver_historical_out_degree",
    "graph_receiver_historical_in_degree",
)
EVENT_TIME_NOVELTY_COLUMNS = (
    "graph_either_endpoint_unseen_before",
    "graph_both_endpoints_unseen_before",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    return parser.parse_args()


def _without_curves(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_curves(item) for key, item in value.items() if key != "curves"}
    if isinstance(value, list):
        return [_without_curves(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    dataset = ds.dataset(args.features, format="parquet", partitioning="hive")
    scored_columns = [
        CANONICAL.transaction_id,
        CANONICAL.is_laundering,
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        CANONICAL.laundering_type,
        "is_cross_border_current_transaction",
        "is_currency_conversion",
        *DEGREE_COLUMNS,
    ]
    scored_columns.extend(
        column for column in EVENT_TIME_NOVELTY_COLUMNS if column in dataset.schema.names
    )
    scored = pl.from_arrow(
        dataset.to_table(filter=ds.field("split") == args.split, columns=scored_columns)
    )
    account_columns = [
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        *DEGREE_COLUMNS,
    ]
    training_accounts_frame = pl.from_arrow(
        dataset.to_table(filter=ds.field("split") == "train", columns=account_columns)
    )
    training_accounts = set(
        pl.concat(
            [
                training_accounts_frame[CANONICAL.sender_account_id],
                training_accounts_frame[CANONICAL.receiver_account_id],
            ],
            how="vertical",
        )
        .drop_nulls()
        .cast(pl.Utf8)
        .unique()
        .to_list()
    )
    training_degree = training_accounts_frame.select(
        pl.sum_horizontal(DEGREE_COLUMNS).alias("endpoint_historical_degree")
    )["endpoint_historical_degree"]
    positive_training_degree = training_degree.filter(training_degree > 0)
    low_cutoff = float(positive_training_degree.quantile(0.25, interpolation="nearest"))
    high_cutoff = float(positive_training_degree.quantile(0.75, interpolation="nearest"))
    if high_cutoff <= low_cutoff:
        raise ValueError("Training degree quantiles do not define distinct slice cutoffs.")
    scores = pl.read_parquet(args.scores).select(
        [CANONICAL.transaction_id, CANONICAL.is_laundering, "graphsage"]
    )
    joined = scored.join(
        scores,
        on=[CANONICAL.transaction_id, CANONICAL.is_laundering],
        how="inner",
        validate="1:1",
    )
    if joined.height != scored.height or joined.height != scores.height:
        raise ValueError("Frozen scores and test features do not align one-to-one.")
    joined = joined.with_columns(
        pl.sum_horizontal(DEGREE_COLUMNS).alias("endpoint_historical_degree")
    ).with_columns(
        pl.when(pl.col("endpoint_historical_degree") == 0)
        .then(pl.lit("cold_zero"))
        .when(pl.col("endpoint_historical_degree") <= low_cutoff)
        .then(pl.lit("low_nonzero"))
        .when(pl.col("endpoint_historical_degree") <= high_cutoff)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
        .alias("degree_band")
    )
    probabilities = joined["graphsage"].to_numpy()
    payload = {
        "schema_version": "1.0",
        "protocol": {
            "score_artifact": str(args.scores),
            "split": args.split,
            "scored_rows": joined.height,
            "training_account_membership_only": True,
            "degree_bands": {
                "cold_zero": "sum of four endpoint historical degrees = 0",
                "low_nonzero": f"0 < degree <= training q25 ({low_cutoff:g})",
                "medium": (
                    f"training q25 ({low_cutoff:g}) < degree <= training q75 ({high_cutoff:g})"
                ),
                "high": f"degree > training q75 ({high_cutoff:g})",
            },
            "selection_note": (
                "Validation slices may inform candidate selection; test slices are descriptive "
                "and must only be generated after model selection is frozen."
            ),
        },
        "new_account": new_account_slice_report(
            joined, probabilities, training_accounts=training_accounts
        ),
        "degree_band": categorical_slice_report(joined, probabilities, column="degree_band"),
        "cross_border": categorical_slice_report(
            joined, probabilities, column="is_cross_border_current_transaction"
        ),
        "currency_conversion": categorical_slice_report(
            joined, probabilities, column="is_currency_conversion"
        ),
        "positive_typology": typology_slice_report(joined, probabilities),
    }
    if set(EVENT_TIME_NOVELTY_COLUMNS).issubset(joined.columns):
        payload["event_time_novelty"] = {
            "definition": (
                "Endpoint history is evaluated strictly before each transaction time; "
                "unlike training-membership slices, an account can become seen later."
            ),
            "either_endpoint_unseen_before": categorical_slice_report(
                joined,
                probabilities,
                column="graph_either_endpoint_unseen_before",
            ),
            "both_endpoints_unseen_before": categorical_slice_report(
                joined,
                probabilities,
                column="graph_both_endpoints_unseen_before",
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_without_curves(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps({"output": str(args.output), "split": args.split, "scored_rows": joined.height})
    )


if __name__ == "__main__":
    main()
