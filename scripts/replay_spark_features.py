"""Run the representative Spark PIT feature replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from aml_evidence_graph.features.spark_replay import replay_parquet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Spark replay output must differ from input.")
    replay_parquet(args.input, args.output, master=args.master)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
