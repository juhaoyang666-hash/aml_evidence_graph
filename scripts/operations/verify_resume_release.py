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
from aml_evidence_graph.investigation.llm import (  # noqa: E402
    DEFAULT_PROMPT_CONFIGURATION,
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
        "--llm-diagnostic-publication",
        type=Path,
        default=Path("reports/public/llm_json_contract_diagnostic_20260804.json"),
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
        action="append",
        default=None,
        help="Tracked Holdout protocol JSON; may be repeated.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_release(
    typology_root: Path,
    llm_publication_path: Path,
    llm_diagnostic_publication_path: Path,
    llm_adjudication_paths: tuple[Path, ...],
    llm_holdout_protocol_paths: tuple[Path, ...],
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
        holdout_protocol_paths=llm_holdout_protocol_paths,
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
    candidate = next(
        stage
        for stage in llm_evaluation.stages
        if stage.evaluation_role
        == "prompt_v4_candidate_project_internal_blind_holdout"
    )
    require(
        candidate.success_criteria_met is False
        and "external_parse_success_rate_minimum"
        in candidate.failed_success_criteria,
        "Failed Prompt v4 availability gate must remain visible in public evidence.",
    )
    require(
        DEFAULT_PROMPT_CONFIGURATION.version == "ecnu-risk-evidence-v7",
        "The default prompt must match the qualified Prompt v7 Holdout v4 result.",
    )
    v6_promoted = next(
        stage
        for stage in llm_evaluation.stages
        if stage.evaluation_role
        == "prompt_v6_promoted_project_internal_blind_holdout"
    )
    require(
        v6_promoted.success_criteria_met is True
        and not v6_promoted.failed_success_criteria,
        "Prompt v6 promotion must remain tied to its successful preregistered Holdout.",
    )
    promoted = next(
        stage
        for stage in llm_evaluation.stages
        if stage.evaluation_role
        == "prompt_v7_promoted_project_internal_blind_holdout"
    )
    require(
        promoted.success_criteria_met is True
        and not promoted.failed_success_criteria
        and promoted.prompt_version == DEFAULT_PROMPT_CONFIGURATION.version,
        "The shipped default must be the prompt that passed the latest Holdout.",
    )
    require(
        promoted.metrics.truncation_retry_count == 0
        and promoted.metrics.retry_attributable_parse_gain == 0.0,
        "Holdout v4's zero-retry null result must stay visible, not be smoothed away.",
    )
    diagnostic = json.loads(llm_diagnostic_publication_path.read_text(encoding="utf-8"))
    require(
        diagnostic.get("diagnostic_not_model_evaluation") is True
        and diagnostic.get("holdout_cases_used") is False
        and diagnostic.get("raw_responses_included") is False,
        "Public LLM diagnostic must preserve its non-evaluation and no-raw boundary.",
    )
    v5_regression = diagnostic.get("v5_development_regression", {})
    require(
        v5_regression.get("development_set_reused") is True
        and v5_regression.get("candidate_promoted") is False,
        "Prompt v5 must remain labelled as a non-blind, unpromoted diagnostic candidate.",
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
            "llm_diagnostic_boundary": "passed",
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
            Path("golden/llm_adjudication_ecnu_max_holdout_v2.json"),
            Path("golden/llm_adjudication_ecnu_max_holdout_v3.json"),
            Path("golden/llm_adjudication_ecnu_max_holdout_v4.json"),
            Path("golden/llm_adjudication_ecnu_max_holdout_v5.json"),
        )
    )
    protocols = tuple(
        args.llm_holdout_protocol
        or (
            Path("golden/llm_holdout_protocol_v1.json"),
            Path("golden/llm_holdout_protocol_v2.json"),
            Path("golden/llm_holdout_protocol_v3.json"),
            Path("golden/llm_holdout_protocol_v4.json"),
            Path("golden/llm_holdout_protocol_v5.json"),
        )
    )
    result = verify_release(
        args.typologies,
        args.llm_publication,
        args.llm_diagnostic_publication,
        adjudications,
        protocols,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
