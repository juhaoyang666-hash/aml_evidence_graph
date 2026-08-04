#!/usr/bin/env python3
"""Generate JSON and Markdown resume evidence from complete local artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aml_evidence_graph.reporting.resume_evidence import (  # noqa: E402
    ResumeEvidenceSpec,
    build_resume_evidence,
    render_resume_evidence_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--spec", type=Path, default=Path("configs/resume_evidence.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/resume_evidence"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/简历证据.md"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    spec = ResumeEvidenceSpec.model_validate(raw_spec)
    report = build_resume_evidence(
        spec,
        root=args.root.resolve(),
        allow_incomplete=args.allow_incomplete,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_resume_evidence_markdown(report), encoding="utf-8")
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
