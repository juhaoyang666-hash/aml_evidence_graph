import hashlib
import json
from pathlib import Path

import pytest

from aml_evidence_graph.investigation.llm_review import (
    load_public_llm_evaluation,
    summarize_human_review,
    validate_public_llm_evaluation,
)


def test_human_review_separates_provider_and_content_quality(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "external_case_count": 2,
        "external_parse_success_rate": 0.5,
        "external_fact_validation_pass_rate": 1.0,
        "reported_prompt_tokens": 10,
        "reported_completion_tokens": 5,
        "estimated_cost_usd": None,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 200.0,
        "cases": [
            {
                "case_id": "accepted",
                "external_call_attempted": True,
                "annotation_used": True,
                "external_error_category": None,
            },
            {
                "case_id": "timeout",
                "external_call_attempted": True,
                "annotation_used": False,
                "external_error_category": "timeout",
            },
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    adjudication_path = tmp_path / "adjudication.json"
    adjudication_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "adjudication_id": "review-v1",
                "reviewed_at": "2026-08-04T00:00:00Z",
                "reviewer_role": "test-reviewer",
                "independence": "project_internal",
                "source_run_id": "run-1",
                "source_summary_sha256": summary_hash,
                "rubric": {"grounded": "test"},
                "reviews": [
                    {
                        "case_id": "accepted",
                        "evidence_grounded": False,
                        "conditional_non_decisive": True,
                        "questions_actionable": True,
                        "injection_resistant": None,
                        "notes": "Qualitative score claim lacked a supplied score value.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = summarize_human_review(summary_path, adjudication_path)

    assert result["external_error_counts"] == {"timeout": 1}
    assert result["human_review_coverage_rate"] == 1.0
    assert result["human_evidence_grounded_rate"] == 0.0
    assert result["human_questions_actionable_rate"] == 1.0
    assert result["human_overall_pass_rate"] == 0.0


def test_checked_in_public_llm_evidence_matches_adjudications() -> None:
    evaluation = load_public_llm_evaluation(
        Path("reports/public/llm_ecnu_max_evaluation_20260804.json")
    )
    adjudications = (
        Path("golden/llm_adjudication_ecnu_max_v1.json"),
        Path("golden/llm_adjudication_ecnu_max_v3.json"),
        Path("golden/llm_adjudication_ecnu_max_holdout_v1.json"),
        Path("golden/llm_adjudication_ecnu_max_holdout_v2.json"),
    )
    protocols = (
        Path("golden/llm_holdout_protocol_v1.json"),
        Path("golden/llm_holdout_protocol_v2.json"),
    )
    validate_public_llm_evaluation(
        evaluation,
        adjudications,
        holdout_protocol_paths=protocols,
    )

    baseline, development, holdout, candidate = evaluation.stages
    assert baseline.evaluation_role == "frozen_baseline"
    assert development.evaluation_role == "same_set_development_regression"
    assert development.same_case_set_as_baseline
    assert not development.prompt_isolated_blind_evaluation
    assert holdout.evaluation_role == "prompt_isolated_project_internal_blind_holdout"
    assert holdout.prompt_isolated_blind_evaluation
    assert holdout.adjudication_independence == "project_internal"
    assert holdout.metrics.human_overall_pass_rate == pytest.approx(11 / 15)
    assert holdout.success_criteria_met is False
    assert holdout.failed_success_criteria == [
        "human_evidence_grounded_rate_minimum",
        "human_overall_pass_rate_minimum",
    ]
    assert candidate.evaluation_role == (
        "prompt_v4_candidate_project_internal_blind_holdout"
    )
    assert candidate.metrics.external_parse_success_rate == 0.1
    assert candidate.success_criteria_met is False
    assert candidate.failed_success_criteria == [
        "external_parse_success_rate_minimum"
    ]
    assert evaluation.cost_status == "unavailable"
    assert all(stage.metrics.estimated_cost_usd is None for stage in evaluation.stages)

    mislabeled = evaluation.model_copy(deep=True)
    mislabeled.stages[1].same_case_set_as_baseline = False
    with pytest.raises(ValueError, match="same-set"):
        validate_public_llm_evaluation(
            mislabeled,
            adjudications,
            holdout_protocol_paths=protocols,
        )
