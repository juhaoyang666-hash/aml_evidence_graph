import json
from pathlib import Path

import pytest

from aml_evidence_graph.evaluation.serving import summarize_serving_benchmark
from aml_evidence_graph.reporting.serving_benchmark import (
    ServingBenchmarkSource,
    ServingBenchmarkSpec,
    build_serving_benchmark_report,
    render_serving_benchmark_markdown,
)


def test_serving_summary_reports_percentiles_and_errors() -> None:
    summary = summarize_serving_benchmark(
        [10.0, 20.0, 30.0, 40.0],
        errors=1,
        wall_time_seconds=2.0,
    )

    assert summary.requests == 5
    assert summary.successes == 4
    assert summary.error_rate == 0.2
    assert summary.throughput_per_second == 2.5
    assert summary.latency_p50_ms == 25.0
    assert summary.latency_p99_ms > summary.latency_p95_ms


def test_serving_summary_rejects_empty_run() -> None:
    with pytest.raises(ValueError, match="At least one request"):
        summarize_serving_benchmark([], errors=0, wall_time_seconds=1.0)


def test_idempotency_benchmark_requires_consistent_execution_counts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "idempotency"
    artifact_dir.mkdir()
    payload = {
        "method": "POST+GET+POST",
        "path": "/v1/controlled-investigations",
        "concurrency": 5,
        "duplicate_reviews_per_thread": 8,
        "review_request_count": 80,
        "idempotent_replay_count": 70,
        "review_execution_count": 10,
        "summary": {
            "requests": 10,
            "error_rate": 0.0,
            "throughput_per_second": 5.0,
            "latency_p50_ms": 10.0,
            "latency_p95_ms": 20.0,
            "latency_p99_ms": 30.0,
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    spec = ServingBenchmarkSpec(
        hardware_disclosure="test",
        sources=[
            ServingBenchmarkSource(
                source_id="controlled_agent_idempotency",
                display_name="idempotency",
                artifact_dir=Path("idempotency"),
            )
        ],
    )

    complete = build_serving_benchmark_report(spec, root=tmp_path)
    payload["idempotent_replay_count"] = 69
    (artifact_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    inconsistent = build_serving_benchmark_report(spec, root=tmp_path)

    assert complete.complete
    assert complete.evidence[0].review_request_count == 80
    assert not inconsistent.complete
    assert inconsistent.incomplete_sources == [
        "controlled_agent_idempotency:invalid_metrics_ValueError"
    ]


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
    assert "所有配置的本机基准来源已齐全" in markdown
