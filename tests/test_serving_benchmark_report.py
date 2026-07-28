import json
from pathlib import Path

from aml_evidence_graph.reporting.serving_benchmark import (
    ServingBenchmarkSource,
    ServingBenchmarkSpec,
    build_serving_benchmark_report,
    render_serving_benchmark_markdown,
)


def test_serving_report_uses_only_aggregate_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/mock"
    artifact.mkdir(parents=True)
    (artifact / "metrics.json").write_text(
        json.dumps(
            {
                "method": "GET",
                "path": "/healthz",
                "concurrency": 2,
                "summary": {
                    "requests": 10,
                    "error_rate": 0.0,
                    "throughput_per_second": 50.0,
                    "latency_p50_ms": 4.0,
                    "latency_p95_ms": 8.0,
                    "latency_p99_ms": 9.0,
                },
            }
        ),
        encoding="utf-8",
    )
    spec = ServingBenchmarkSpec(
        hardware_disclosure="test host",
        sources=[
            ServingBenchmarkSource(
                source_id="mock",
                display_name="Mock",
                artifact_dir=Path("artifacts/mock"),
            )
        ],
    )

    report = build_serving_benchmark_report(spec, root=tmp_path)
    markdown = render_serving_benchmark_markdown(report)

    assert report.complete
    assert report.evidence[0].latency_p95_ms == 8.0
    assert "不是生产 SLA" in markdown
