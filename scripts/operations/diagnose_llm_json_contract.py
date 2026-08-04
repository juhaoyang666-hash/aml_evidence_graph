#!/usr/bin/env python3
"""Diagnose ECNU JSON response failures on a non-evaluation synthetic matrix."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from aml_evidence_graph.investigation.llm import (
    diagnose_annotation_content,
    load_prompt_configuration,
)
from aml_evidence_graph.settings import Settings
from aml_evidence_graph.tracking.run import create_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/diagnostics/ecnu_json_contract_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/llm_diagnostics/json_contract_v1.json"),
    )
    parser.add_argument("--retain-raw-local", action="store_true")
    parser.add_argument(
        "--reclassify-existing",
        action="store_true",
        help="Recompute metadata from existing ignored raw files without provider calls.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _extract_content(payload: object) -> tuple[object, str | None]:
    if not isinstance(payload, dict):
        return None, None
    try:
        choice = payload["choices"][0]
        return choice["message"]["content"], choice.get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None, None


def _usage_metadata(payload: object) -> dict[str, int | None]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    return {
        key: value if isinstance(value, int) and not isinstance(value, bool) else None
        for key, value in {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }.items()
    }


def _summarize_records(
    records: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_id = variant["variant_id"]
        selected = [record for record in records if record["variant_id"] == variant_id]
        categories = Counter(str(record["category"]) for record in selected)
        by_variant[variant_id] = {
            "case_count": len(selected),
            "valid_contract_count": categories.get("valid_contract", 0),
            "production_parser_compatible_count": sum(
                record["production_parser_compatible"] for record in selected
            ),
            "category_counts": dict(sorted(categories.items())),
            "length_finish_count": sum(
                record["finish_reason"] == "length" for record in selected
            ),
        }
    return by_variant


def run_diagnostic(
    *,
    protocol_path: Path,
    output_path: Path,
    retain_raw_local: bool,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    settings = Settings()
    api_key = settings.require_llm_api_key()
    endpoint = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    records: list[dict[str, Any]] = []
    raw_dir = output_path.parent / f"{output_path.stem}_raw_local"
    if retain_raw_local:
        raw_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
        for variant in protocol["variants"]:
            configured = load_prompt_configuration(Path(variant["prompt_path"]))
            prompt = replace(configured, max_tokens=int(variant["max_tokens"]))
            for case in protocol["cases"]:
                request_payload = {
                    "model": settings.llm_model,
                    "temperature": prompt.temperature,
                    "max_tokens": prompt.max_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": prompt.system_instructions},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "prompt_version": prompt.version,
                                    "allowed_evidence_references": [
                                        "fusion_probability",
                                        "graph_evidence",
                                        "missing_evidence",
                                        "uncertainty_notes",
                                    ],
                                    "deidentified_evidence": case["deidentified_evidence"],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                }
                started = time.perf_counter()
                response = client.post(endpoint, headers=headers, json=request_payload)
                latency_ms = (time.perf_counter() - started) * 1_000
                response.raise_for_status()
                payload = response.json()
                content, finish_reason = _extract_content(payload)
                diagnostic = diagnose_annotation_content(content)
                record = {
                    "variant_id": variant["variant_id"],
                    "prompt_version": prompt.version,
                    "max_tokens": prompt.max_tokens,
                    "case_id": case["case_id"],
                    "latency_ms": latency_ms,
                    "finish_reason": finish_reason,
                    **_usage_metadata(payload),
                    **asdict(diagnostic),
                }
                records.append(record)
                if retain_raw_local:
                    raw_path = raw_dir / f"{variant['variant_id']}__{case['case_id']}.json"
                    raw_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

    by_variant = _summarize_records(records, protocol["variants"])
    result = {
        "schema_version": "1.0",
        "purpose": protocol["purpose"],
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_responses_retained_locally": retain_raw_local,
        "raw_responses_publishable": False,
        "summary_by_variant": by_variant,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    create_run_manifest(
        output_dir=output_path.parent,
        command="diagnose-llm-json-contract",
        random_seed=0,
        input_paths={"diagnostic_protocol": protocol_path},
        config_paths={
            variant["variant_id"]: Path(variant["prompt_path"])
            for variant in protocol["variants"]
        },
        metadata={
            "purpose": protocol["purpose"],
            "raw_responses_retained_locally": retain_raw_local,
            "summary_by_variant": by_variant,
        },
        filename=f"{output_path.stem}_run_manifest.json",
    )
    return result


def reclassify_existing(*, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    """Refresh safe metadata after classifier changes, without another external call."""
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(output_path.read_text(encoding="utf-8"))
    raw_dir = output_path.parent / f"{output_path.stem}_raw_local"
    for record in result["records"]:
        raw_path = raw_dir / f"{record['variant_id']}__{record['case_id']}.json"
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        content, finish_reason = _extract_content(payload)
        record.update(asdict(diagnose_annotation_content(content)))
        record["finish_reason"] = finish_reason
        record.update(_usage_metadata(payload))
    result["summary_by_variant"] = _summarize_records(
        result["records"], protocol["variants"]
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.reclassify_existing:
        result = reclassify_existing(
            protocol_path=args.protocol,
            output_path=args.output,
        )
        print(json.dumps(result["summary_by_variant"], ensure_ascii=False, indent=2))
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite.")
    result = run_diagnostic(
        protocol_path=args.protocol,
        output_path=args.output,
        retain_raw_local=args.retain_raw_local,
    )
    print(json.dumps(result["summary_by_variant"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
