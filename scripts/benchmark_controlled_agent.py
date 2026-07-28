#!/usr/bin/env python3
"""Benchmark a complete local controlled-Agent start, query, and review cycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx
import psutil

from aml_evidence_graph.evaluation.serving import summarize_serving_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--alert-id", default="mock-alert-0001")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/serving_benchmark_controlled_agent")
    )
    return parser.parse_args()


async def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.requests < 1 or args.concurrency < 1:
        raise ValueError("requests and concurrency must be positive.")
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    errors = 0
    token = os.environ.get("AML_INTERNAL_API_TOKEN")
    headers = {"X-AML-Internal-Token": token} if token else {}
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=args.timeout, headers=headers
    ) as client:

        async def one_cycle() -> None:
            nonlocal errors
            thread_id = f"benchmark-{uuid.uuid4().hex}"
            async with semaphore:
                started = time.perf_counter()
                try:
                    started_response = await client.post(
                        f"/v1/controlled-investigations/{args.alert_id}",
                        json={"thread_id": thread_id},
                    )
                    started_response.raise_for_status()
                    queried = await client.get(
                        f"/v1/controlled-investigations/{thread_id}"
                    )
                    queried.raise_for_status()
                    reviewed = await client.post(
                        f"/v1/controlled-investigations/{thread_id}/review",
                        json={
                            "action": "approve",
                            "reviewer_reference": "local-benchmark",
                        },
                    )
                    reviewed.raise_for_status()
                    if reviewed.json().get("status") != "completed":
                        raise ValueError("Controlled investigation did not complete.")
                except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                    errors += 1
                else:
                    latencies.append((time.perf_counter() - started) * 1_000)

        wall_started = time.perf_counter()
        await asyncio.gather(*(one_cycle() for _ in range(args.requests)))
        wall_time = time.perf_counter() - wall_started
    summary = summarize_serving_benchmark(
        latencies, errors=errors, wall_time_seconds=wall_time
    )
    process = psutil.Process()
    return {
        "schema_version": "1.0",
        "scope": "local synthetic start/query/review cycle; not a production SLA",
        "base_url": args.base_url,
        "path": "/v1/controlled-investigations/{alert_id} -> status -> review",
        "method": "POST+GET+POST",
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "summary": asdict(summary),
        "client_environment": {
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "client_rss_mb": process.memory_info().rss / 1024 / 1024,
        },
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run_benchmark(args))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
