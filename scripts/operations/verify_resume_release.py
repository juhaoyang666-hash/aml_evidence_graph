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
from aml_evidence_graph.investigation.llm_review import (  # noqa: E402
    load_public_llm_evaluation,
    validate_public_llm_evaluation,
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
    parser.add_argument(
        "--llm-publication",
        type=Path,
        default=Path("reports/public/llm_ecnu_max_evaluation_20260804.json"),
    )
    parser.add_argument(
        "--llm-adjudication",
        type=Path,
        action="append",
        default=None,
        help="Tracked adjudication JSON; may be repeated.",
    )
    parser.add_argument(
        "--llm-holdout-protocol",
        type=Path,
        default=Path("golden/llm_holdout_protocol_v1.json"),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_release(
    typology_root: Path,
    llm_publication_path: Path,
    llm_adjudication_paths: tuple[Path, ...],
    llm_holdout_protocol_path: Path,
) -> dict[str, object]:
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

    require(llm_publication_path.is_file(), "Public LLM evaluation evidence is missing.")
    require(
        all(path.is_file() for path in llm_adjudication_paths),
        "One or more LLM adjudication files are missing.",
    )
    llm_evaluation = load_public_llm_evaluation(llm_publication_path)
    validate_public_llm_evaluation(
        llm_evaluation,
        llm_adjudication_paths,
        holdout_protocol_path=llm_holdout_protocol_path,
    )
    development = next(
        stage
        for stage in llm_evaluation.stages
        if stage.evaluation_role == "same_set_development_regression"
    )
    require(
        development.same_case_set_as_baseline
        and not development.prompt_isolated_blind_evaluation,
        "LLM development result must remain labelled as a same-set non-blind regression.",
    )
    require(
        llm_evaluation.cost_status != "unavailable"
        or all(
            stage.metrics.estimated_cost_usd is None for stage in llm_evaluation.stages
        ),
        "Unavailable LLM cost must not be represented as a numeric claim.",
    )
    holdout = next(
        stage
        for stage in llm_evaluation.stages
        if stage.evaluation_role == "prompt_isolated_project_internal_blind_holdout"
    )
    require(
        holdout.prompt_isolated_blind_evaluation
        and holdout.adjudication_independence == "project_internal",
        "Holdout must preserve the prompt-isolated/project-internal review boundary.",
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
            "llm_public_evidence": "passed",
        },
    }


def main() -> None:
    args = parse_args()
    adjudications = tuple(
        args.llm_adjudication
        or (
            Path("golden/llm_adjudication_ecnu_max_v1.json"),
            Path("golden/llm_adjudication_ecnu_max_v3.json"),
            Path("golden/llm_adjudication_ecnu_max_holdout_v1.json"),
        )
    )
    result = verify_release(
        args.typologies,
        args.llm_publication,
        adjudications,
        args.llm_holdout_protocol,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
