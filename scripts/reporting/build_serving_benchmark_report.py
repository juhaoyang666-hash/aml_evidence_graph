"""Build the Chinese serving benchmark report from aggregate local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aml_evidence_graph.reporting.serving_benchmark import (
    ServingBenchmarkSpec,
    build_serving_benchmark_report,
    render_serving_benchmark_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/serving_benchmark.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/serving_benchmark_report"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/服务性能基准.md"),
    )
    args = parser.parse_args()
    spec = ServingBenchmarkSpec.model_validate(
        yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    )
    report = build_serving_benchmark_report(spec, root=args.root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_serving_benchmark_markdown(report), encoding="utf-8")
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
