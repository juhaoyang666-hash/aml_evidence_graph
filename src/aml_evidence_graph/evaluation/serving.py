"""Serving benchmark aggregation without claiming a production SLA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ServingBenchmarkSummary:
    requests: int
    successes: int
    errors: int
    error_rate: float
    throughput_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


def summarize_serving_benchmark(
    latencies_ms: list[float],
    *,
    errors: int,
    wall_time_seconds: float,
) -> ServingBenchmarkSummary:
    """Summarize observed local requests; empty successful samples are allowed."""
    if errors < 0 or wall_time_seconds <= 0:
        raise ValueError("errors must be non-negative and wall time must be positive.")
    successes = len(latencies_ms)
    requests = successes + errors
    if requests == 0:
        raise ValueError("At least one request is required.")
    percentiles = (
        np.percentile(np.asarray(latencies_ms), [50, 95, 99]).tolist()
        if latencies_ms
        else [0.0, 0.0, 0.0]
    )
    return ServingBenchmarkSummary(
        requests=requests,
        successes=successes,
        errors=errors,
        error_rate=errors / requests,
        throughput_per_second=requests / wall_time_seconds,
        latency_p50_ms=float(percentiles[0]),
        latency_p95_ms=float(percentiles[1]),
        latency_p99_ms=float(percentiles[2]),
    )
