#!/usr/bin/env python3
"""Run a bounded async benchmark against the local Mock or controlled API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import psutil

from aml_evidence_graph.evaluation.serving import summarize_serving_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--path", default="/demo/cases/mock-alert-0001/draft")
    parser.add_argument("--method", choices=("GET", "POST"), default="POST")
    parser.add_argument(
        "--json-body",
        help="Optional JSON object sent with every request; use only non-sensitive references.",
    )
    parser.add_argument(
        "--json-body-file",
        type=Path,
        help="UTF-8 JSON object file; preferred on shells that rewrite nested quotes.",
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Honor HTTP(S)_PROXY and related environment settings (disabled by default).",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/serving_benchmark"))
    return parser.parse_args()


async def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.requests < 1 or args.concurrency < 1:
        raise ValueError("requests and concurrency must be positive.")
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    errors = 0
    token = os.environ.get("AML_INTERNAL_API_TOKEN")
    headers = {"X-AML-Internal-Token": token} if token else {}
    if args.json_body and args.json_body_file:
        raise ValueError("Use only one of json-body and json-body-file.")
    raw_json_body = (
        args.json_body_file.read_text(encoding="utf-8")
        if args.json_body_file
        else args.json_body
    )
    json_body = json.loads(raw_json_body) if raw_json_body else None
    if json_body is not None and not isinstance(json_body, dict):
        raise ValueError("json-body must decode to a JSON object.")

    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.timeout,
        headers=headers,
        trust_env=args.trust_env,
    ) as client:

        async def one_request() -> None:
            nonlocal errors
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.request(
                        args.method, args.path, json=json_body
                    )
                    response.raise_for_status()
                except (httpx.HTTPError, ValueError):
                    errors += 1
                else:
                    latencies.append((time.perf_counter() - started) * 1_000)

        wall_started = time.perf_counter()
        await asyncio.gather(*(one_request() for _ in range(args.requests)))
        wall_time = time.perf_counter() - wall_started

    summary = summarize_serving_benchmark(
        latencies,
        errors=errors,
        wall_time_seconds=wall_time,
    )
    process = psutil.Process()
    return {
        "schema_version": "1.0",
        "scope": "local benchmark; not a production SLA",
        "base_url": args.base_url,
        "path": args.path,
        "method": args.method,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "trust_env": args.trust_env,
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
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
