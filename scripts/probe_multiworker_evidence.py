#!/usr/bin/env python3
"""Probe whether one scored alert is readable across independent HTTP connections."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--partition-ref", required=True)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attempts < 1:
        raise ValueError("attempts must be positive.")
    token = os.environ.get("AML_INTERNAL_API_TOKEN")
    if not token:
        raise RuntimeError("AML_INTERNAL_API_TOKEN is required.")
    headers = {
        "X-AML-Internal-Token": token,
        "Connection": "close",
    }
    with httpx.Client(
        base_url=args.base_url,
        headers=headers,
        timeout=args.timeout,
        trust_env=False,
    ) as client:
        response = client.post(
            "/v1/score/batch", json={"partition_ref": args.partition_ref}
        )
        response.raise_for_status()
        alert_ids = response.json()["alert_ids"]
    if not alert_ids:
        raise RuntimeError("Scoring returned no alert reference to probe.")
    alert_id = alert_ids[0]

    statuses: list[int] = []
    for _ in range(args.attempts):
        # A fresh client and Connection: close prevent a keep-alive connection from
        # pinning the entire probe to the worker that served the scoring request.
        with httpx.Client(
            base_url=args.base_url,
            headers=headers,
            timeout=args.timeout,
            trust_env=False,
        ) as client:
            statuses.append(client.get(f"/v1/evidence/{alert_id}").status_code)
    counts = Counter(statuses)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "local multi-worker evidence visibility probe; no evidence content persisted",
        "partition_ref": args.partition_ref,
        "score_alert_count": len(alert_ids),
        "attempts": args.attempts,
        "status_counts": {str(status): count for status, count in sorted(counts.items())},
        "cross_worker_visibility_rate": counts.get(200, 0) / args.attempts,
        "interpretation": (
            "Any 404 proves process-local evidence storage is not safe behind multiple workers."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
