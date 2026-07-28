import pytest

from aml_evidence_graph.evaluation.serving import summarize_serving_benchmark


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
