"""Run the representative Spark PIT feature replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aml_evidence_graph.features.spark_replay import replay_parquet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--target-event-date")
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Spark replay output must differ from input.")
    summary = replay_parquet(
        args.input,
        args.output,
        master=args.master,
        target_event_date=args.target_event_date,
        shuffle_partitions=args.shuffle_partitions,
    )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
