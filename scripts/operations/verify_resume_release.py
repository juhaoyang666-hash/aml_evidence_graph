#!/usr/bin/env python3
"""Verify the public, mock-only resume release without private artifacts or a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from aml_evidence_graph import __version__  # noqa: E402
from aml_evidence_graph.api.app import create_app  # noqa: E402
from aml_evidence_graph.evidence.typology import (  # noqa: E402
    LocalBM25TypologyRetriever,
    load_typology_documents,
)

MOCK_ALERT_ID = "mock-alert-0001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic, mock-only resume release acceptance checks."
    )
    parser.add_argument(
        "--typologies",
        type=Path,
        default=Path("knowledge/typologies"),
        help="Versioned local Typology directory.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path.")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_release(typology_root: Path) -> dict[str, object]:
    require(typology_root.is_dir(), f"Typology directory does not exist: {typology_root}")
    retriever = LocalBM25TypologyRetriever(load_typology_documents(typology_root))
    client = TestClient(create_app(retriever))

    openapi = client.get("/openapi.json")
    require(openapi.status_code == 200, "OpenAPI document is unavailable.")
    require(
        openapi.json()["info"]["version"] == __version__,
        "API version does not match the resume release version.",
    )

    health = client.get("/healthz")
    require(health.status_code == 200, "Health endpoint failed.")
    require(health.json()["status"] == "ok", "Health endpoint is not healthy.")
    require(health.json()["llm_scoring"] == "disabled", "LLM scoring must stay disabled.")

    demo = client.get("/demo")
    require(demo.status_code == 200, "Demo page failed.")
    require("虚构 Evidence Package" in demo.text, "Demo boundary statement is missing.")

    evidence = client.get("/demo/evidence")
    require(evidence.status_code == 200, "Mock evidence endpoint failed.")
    evidence_payload = evidence.json()
    require(evidence_payload["alert_id"] == MOCK_ALERT_ID, "Unexpected mock alert ID.")

    draft = client.post(f"/demo/cases/{MOCK_ALERT_ID}/draft")
    require(draft.status_code == 200, "Deterministic investigation draft failed.")
    draft_payload = draft.json()
    require(
        draft_payload["status"] == "draft_requires_human_review",
        "Demo must stop at human review.",
    )
    require(draft_payload.get("sar_draft") is not None, "SAR draft skeleton is missing.")

    scoring = client.post("/v1/score/batch", json={"partition_ref": "mock-partition"})
    require(scoring.status_code == 200, "Mock batch scoring endpoint failed.")
    require(
        scoring.json()["model_version"] == "mock-model-v1",
        "Mock scoring must not claim a frozen production model.",
    )

    return {
        "release_version": __version__,
        "mode": "mock_only",
        "private_artifacts_required": False,
        "gpu_required": False,
        "external_llm_called": False,
        "checks": {
            "openapi_version": "passed",
            "health": "passed",
            "demo_boundary": "passed",
            "mock_evidence": "passed",
            "human_review_stop": "passed",
            "mock_scoring_boundary": "passed",
        },
    }


def main() -> None:
    args = parse_args()
    result = verify_release(args.typologies)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
