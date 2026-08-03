#!/usr/bin/env python3
"""Run the versioned synthetic Golden set for the controlled investigation Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aml_evidence_graph.investigation.agent_evaluation import (
    evaluate_agent_cases,
    load_agent_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("golden/agent_cases_v2.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/agent_evaluation/metrics.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite.")
    metrics = evaluate_agent_cases(load_agent_cases(args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if key != "case_results"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
