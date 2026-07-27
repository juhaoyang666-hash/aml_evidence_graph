"""CLI for immutable test-period evaluation of persisted OOF fusion artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.data.splits import TimeSplit
from aml_evidence_graph.training.fusion import (
    evaluate_persisted_fusion,
    load_persisted_fusion_artifacts,
    merge_component_scores,
)
from aml_evidence_graph.training.table_baseline import load_feature_split


def _read_scores(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError("Fusion score inputs must be Parquet or CSV.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True, action="append")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fusion, _ = load_persisted_fusion_artifacts(args.fusion_dir)
    frames = [_read_scores(path) for path in args.test]
    test_scores = (
        merge_component_scores(frames, component_columns=list(fusion.model_names))
        if len(frames) > 1
        else frames[0]
    )
    context = load_feature_split(args.features, TimeSplit.TEST) if args.features else None
    training_accounts = None
    if args.features:
        training = load_feature_split(args.features, TimeSplit.TRAIN)
        training_accounts = set(
            pl.concat(
                [
                    training[CANONICAL.sender_account_id],
                    training[CANONICAL.receiver_account_id],
                ],
                how="vertical",
            )
            .cast(pl.Utf8)
            .to_list()
        )
    summary = evaluate_persisted_fusion(
        args.fusion_dir,
        test_scores,
        args.output,
        test_context=context,
        training_accounts=training_accounts,
        bootstrap_iterations=args.bootstrap_iterations,
        input_paths={
            **{f"test_scores_{index}": path for index, path in enumerate(args.test)},
            **({"pit_feature_dataset": args.features} if args.features else {}),
        },
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "test_pr_auc": summary.test_metrics["pr_auc"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
