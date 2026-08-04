#!/usr/bin/env python3
"""Publish a redacted LLM evaluation aggregate from frozen local artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aml_evidence_graph.investigation.llm_review import (  # noqa: E402
    build_public_llm_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-adjudication", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--development-adjudication", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation = build_public_llm_evaluation(
        baseline_summary_path=args.baseline_summary,
        baseline_adjudication_path=args.baseline_adjudication,
        development_summary_path=args.development_summary,
        development_adjudication_path=args.development_adjudication,
        evaluation_id=args.evaluation_id,
        evaluated_at=args.evaluated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
