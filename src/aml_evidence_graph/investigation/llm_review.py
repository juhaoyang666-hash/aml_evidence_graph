"""Validate and summarize human review of frozen external-LLM Golden outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMHumanReview(BaseModel):
    """One strict, project-internal review of an annotation that passed automatic gates."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    evidence_grounded: bool
    conditional_non_decisive: bool
    questions_actionable: bool
    injection_resistant: bool | None = None
    notes: str = Field(min_length=1)

    @property
    def overall_pass(self) -> bool:
        injection_pass = self.injection_resistant is not False
        return (
            self.evidence_grounded
            and self.conditional_non_decisive
            and self.questions_actionable
            and injection_pass
        )


class LLMHumanAdjudication(BaseModel):
    """Versioned review protocol tied to one immutable LLM summary hash."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    adjudication_id: str
    reviewed_at: str
    reviewer_role: str
    independence: Literal["project_internal", "external_independent"]
    source_run_id: str
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric: dict[str, str]
    reviews: list[LLMHumanReview]


class LLMPublicStageMetrics(BaseModel):
    """Safe aggregate metrics; no prompt payload or generated text is publishable."""

    model_config = ConfigDict(extra="forbid")

    external_parse_success_rate: float = Field(ge=0, le=1)
    external_fact_validation_pass_rate: float = Field(ge=0, le=1)
    accepted_annotation_count: int = Field(ge=0)
    human_review_coverage_rate: float = Field(ge=0, le=1)
    human_evidence_grounded_rate: float = Field(ge=0, le=1)
    human_conditional_non_decisive_rate: float = Field(ge=0, le=1)
    human_questions_actionable_rate: float = Field(ge=0, le=1)
    human_injection_resistance_rate: float | None = Field(default=None, ge=0, le=1)
    human_overall_pass_rate: float = Field(ge=0, le=1)
    latency_p50_ms_all_cases: float = Field(ge=0)
    latency_p95_ms_all_cases: float = Field(ge=0)
    reported_prompt_tokens_for_accepted_annotations: int = Field(ge=0)
    reported_completion_tokens_for_accepted_annotations: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class LLMPublicStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    evaluation_role: Literal[
        "frozen_baseline",
        "same_set_development_regression",
        "prompt_isolated_project_internal_blind_holdout",
        "prompt_v4_candidate_project_internal_blind_holdout",
        "prompt_v6_promoted_project_internal_blind_holdout",
    ]
    prompt_version: str
    case_count: int = Field(ge=1)
    external_case_count: int = Field(ge=1)
    deterministic_probe_count: int = Field(ge=0)
    source_run_id: str
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjudication_id: str
    adjudication_independence: Literal["project_internal", "external_independent"]
    same_case_set_as_baseline: bool
    prompt_isolated_blind_evaluation: bool
    preregistered_protocol_id: str | None = None
    preregistered_protocol_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    preregistered_source_revision: str | None = None
    success_criteria_met: bool | None = None
    failed_success_criteria: list[str] = Field(default_factory=list)
    metrics: LLMPublicStageMetrics


class LLMPublicEvaluation(BaseModel):
    """Tracked, resume-safe representation of external-provider evaluation evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.3"] = "1.3"
    evaluation_id: str
    evaluated_at: str
    model_name: str
    cost_status: Literal["reported", "unavailable"]
    stages: list[LLMPublicStage] = Field(min_length=5)
    limitations: list[str] = Field(min_length=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_crlf_text(path: Path) -> str:
    """Reproduce preregistration hashes for JSON frozen on the Windows runner."""

    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.replace("\n", "\r\n").encode()).hexdigest()


def summarize_human_review(
    summary_path: Path,
    adjudication_path: Path,
) -> dict[str, object]:
    """Require complete review coverage and separate availability from content quality."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    adjudication = LLMHumanAdjudication.model_validate_json(
        adjudication_path.read_text(encoding="utf-8")
    )
    observed_hash = _sha256(summary_path)
    if observed_hash != adjudication.source_summary_sha256:
        raise ValueError("LLM summary hash does not match the frozen adjudication source.")

    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError("LLM summary must contain a case list.")
    reviewable_ids = {
        str(case["case_id"])
        for case in cases
        if case.get("external_call_attempted") and case.get("annotation_used")
    }
    review_ids = [review.case_id for review in adjudication.reviews]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Human adjudication contains duplicate case IDs.")
    if set(review_ids) != reviewable_ids:
        missing = sorted(reviewable_ids.difference(review_ids))
        unexpected = sorted(set(review_ids).difference(reviewable_ids))
        raise ValueError(
            f"Human review coverage mismatch; missing={missing}, unexpected={unexpected}."
        )

    reviews = adjudication.reviews
    adversarial_reviews = [
        review for review in reviews if review.injection_resistant is not None
    ]
    error_counts = Counter(
        str(case["external_error_category"])
        for case in cases
        if case.get("external_call_attempted") and case.get("external_error_category")
    )

    def rate(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "schema_version": "1.0",
        "adjudication_id": adjudication.adjudication_id,
        "source_run_id": adjudication.source_run_id,
        "source_summary_sha256": observed_hash,
        "reviewer_role": adjudication.reviewer_role,
        "independence": adjudication.independence,
        "external_case_count": int(summary["external_case_count"]),
        "external_parse_success_rate": summary["external_parse_success_rate"],
        "external_fact_validation_pass_rate": summary[
            "external_fact_validation_pass_rate"
        ],
        "external_error_counts": dict(sorted(error_counts.items())),
        "accepted_annotation_count": len(reviewable_ids),
        "human_review_coverage_rate": 1.0,
        "human_evidence_grounded_rate": rate(
            [review.evidence_grounded for review in reviews]
        ),
        "human_conditional_non_decisive_rate": rate(
            [review.conditional_non_decisive for review in reviews]
        ),
        "human_questions_actionable_rate": rate(
            [review.questions_actionable for review in reviews]
        ),
        "human_injection_resistance_rate": rate(
            [review.injection_resistant is True for review in adversarial_reviews]
        ),
        "human_overall_pass_rate": rate([review.overall_pass for review in reviews]),
        "reported_prompt_tokens_for_accepted_annotations": summary[
            "reported_prompt_tokens"
        ],
        "reported_completion_tokens_for_accepted_annotations": summary[
            "reported_completion_tokens"
        ],
        "estimated_cost_usd": summary["estimated_cost_usd"],
        "latency_p50_ms_all_cases": summary["latency_p50_ms"],
        "latency_p95_ms_all_cases": summary["latency_p95_ms"],
    }


def build_public_llm_evaluation(
    *,
    baseline_summary_path: Path,
    baseline_adjudication_path: Path,
    development_summary_path: Path,
    development_adjudication_path: Path,
    holdout_summary_path: Path,
    holdout_adjudication_path: Path,
    holdout_protocol_path: Path,
    holdout_run_manifest_path: Path,
    candidate_summary_path: Path,
    candidate_adjudication_path: Path,
    candidate_protocol_path: Path,
    candidate_run_manifest_path: Path,
    promoted_summary_path: Path,
    promoted_adjudication_path: Path,
    promoted_protocol_path: Path,
    promoted_run_manifest_path: Path,
    evaluation_id: str,
    evaluated_at: str,
) -> LLMPublicEvaluation:
    """Build a public aggregate while rechecking development and Holdout evidence."""

    definitions = (
        (
            "prompt_v1_frozen_baseline",
            "frozen_baseline",
            baseline_summary_path,
            baseline_adjudication_path,
            False,
            False,
        ),
        (
            "prompt_v3_same_set_regression",
            "same_set_development_regression",
            development_summary_path,
            development_adjudication_path,
            True,
            False,
        ),
        (
            "prompt_v3_preregistered_holdout",
            "prompt_isolated_project_internal_blind_holdout",
            holdout_summary_path,
            holdout_adjudication_path,
            False,
            True,
            holdout_protocol_path,
            holdout_run_manifest_path,
        ),
        (
            "prompt_v4_preregistered_holdout_v2",
            "prompt_v4_candidate_project_internal_blind_holdout",
            candidate_summary_path,
            candidate_adjudication_path,
            False,
            True,
            candidate_protocol_path,
            candidate_run_manifest_path,
        ),
        (
            "prompt_v6_preregistered_holdout_v3",
            "prompt_v6_promoted_project_internal_blind_holdout",
            promoted_summary_path,
            promoted_adjudication_path,
            False,
            True,
            promoted_protocol_path,
            promoted_run_manifest_path,
        ),
    )
    stages: list[LLMPublicStage] = []
    model_names: set[str] = set()
    costs: list[float | None] = []
    for definition in definitions:
        stage_id, role, summary_path, adjudication_path, same_set, blind, *prereg = (
            definition
        )
        raw_summary = _load_json_object(summary_path)
        aggregate = summarize_human_review(summary_path, adjudication_path)
        prompt_versions = raw_summary.get("prompt_versions")
        stage_models = raw_summary.get("model_names")
        if not isinstance(prompt_versions, list) or len(prompt_versions) != 1:
            raise ValueError(f"{stage_id} must contain exactly one prompt version.")
        if not isinstance(stage_models, list) or len(stage_models) != 1:
            raise ValueError(f"{stage_id} must contain exactly one model name.")
        case_count = int(raw_summary["case_count"])
        external_case_count = int(aggregate["external_case_count"])
        model_names.add(str(stage_models[0]))
        metrics = LLMPublicStageMetrics.model_validate(
            {
                name: aggregate[name]
                for name in LLMPublicStageMetrics.model_fields
            }
        )
        costs.append(metrics.estimated_cost_usd)
        stage = LLMPublicStage(
                stage_id=stage_id,
                evaluation_role=role,
                prompt_version=str(prompt_versions[0]),
                case_count=case_count,
                external_case_count=external_case_count,
                deterministic_probe_count=case_count - external_case_count,
                source_run_id=str(aggregate["source_run_id"]),
                source_summary_sha256=str(aggregate["source_summary_sha256"]),
                adjudication_id=str(aggregate["adjudication_id"]),
                adjudication_independence=str(aggregate["independence"]),
                same_case_set_as_baseline=same_set,
                prompt_isolated_blind_evaluation=blind,
                metrics=metrics,
            )
        if prereg:
            _attach_preregistration(stage, prereg[0], prereg[1])
        stages.append(stage)
    if len(model_names) != 1:
        raise ValueError("Published LLM stages must use one model.")

    result = LLMPublicEvaluation(
        evaluation_id=evaluation_id,
        evaluated_at=evaluated_at,
        model_name=model_names.pop(),
        cost_status="unavailable" if all(cost is None for cost in costs) else "reported",
        stages=stages,
        limitations=[
            "All adjudications are project-internal rather than external expert review.",
            "Prompt v3 is a same-Golden-set development regression, not an independent blind test.",
            "The Holdout is independent of prompt development and preregistered, but its human "
            "review is still project-internal.",
            "Prompt v4 failed its preregistered Holdout v2 availability gate and was not promoted.",
            "Prompt v6 passed its preregistered Holdout v3 gates and replaced v3 as the default; "
            "its human review remains project-internal.",
            "Provider parse success measures availability and format compliance, "
            "not content quality.",
            "The provider did not return price metadata, so monetary cost is unavailable.",
        ],
    )
    validate_public_llm_evaluation(
        result,
        (
            baseline_adjudication_path,
            development_adjudication_path,
            holdout_adjudication_path,
            candidate_adjudication_path,
            promoted_adjudication_path,
        ),
        holdout_protocol_paths=(
            holdout_protocol_path,
            candidate_protocol_path,
            promoted_protocol_path,
        ),
    )
    return result


def _attach_preregistration(
    stage: LLMPublicStage,
    protocol_path: Path,
    run_manifest_path: Path,
) -> None:
    protocol = _load_json_object(protocol_path)
    run_manifest = _load_json_object(run_manifest_path)
    if run_manifest.get("run_id") != stage.source_run_id:
        raise ValueError("Holdout run manifest does not match the reviewed summary.")
    inputs = run_manifest.get("inputs")
    configs = run_manifest.get("config_fingerprints")
    if not isinstance(inputs, dict) or not isinstance(configs, dict):
        raise ValueError("Holdout run manifest lacks input or prompt fingerprints.")
    case_input = inputs.get("golden_cases")
    prompt_input = configs.get("prompt_configuration")
    if (
        not isinstance(case_input, dict)
        or case_input.get("sha256") != protocol.get("cases_sha256")
        or not isinstance(prompt_input, dict)
        or prompt_input.get("sha256") != protocol.get("prompt_sha256")
    ):
        raise ValueError("Holdout run did not use the preregistered cases and prompt.")
    stage.preregistered_protocol_id = str(protocol["protocol_id"])
    stage.preregistered_protocol_sha256 = _sha256_crlf_text(protocol_path)
    stage.preregistered_source_revision = str(run_manifest["source_revision"])
    criteria = protocol.get("success_criteria")
    if not isinstance(criteria, dict):
        raise ValueError("Holdout protocol lacks success criteria.")
    failed_criteria: list[str] = []
    for criterion, threshold in criteria.items():
        metric_name = str(criterion).removesuffix("_minimum")
        observed = getattr(stage.metrics, metric_name, None)
        passed = (
            observed is not None
            and isinstance(threshold, int | float)
            and (
                observed >= float(threshold)
                if str(criterion).endswith("_minimum")
                else observed == float(threshold)
            )
        )
        if not passed:
            failed_criteria.append(str(criterion))
    stage.success_criteria_met = not failed_criteria
    stage.failed_success_criteria = failed_criteria


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_public_llm_evaluation(path: Path) -> LLMPublicEvaluation:
    return LLMPublicEvaluation.model_validate_json(path.read_text(encoding="utf-8"))


def validate_public_llm_evaluation(
    evaluation: LLMPublicEvaluation,
    adjudication_paths: tuple[Path, ...],
    *,
    holdout_protocol_paths: tuple[Path, ...] = (),
) -> None:
    """Validate checked-in aggregates without requiring ignored raw provider outputs."""

    adjudications = {
        item.adjudication_id: item
        for item in (
            LLMHumanAdjudication.model_validate_json(path.read_text(encoding="utf-8"))
            for path in adjudication_paths
        )
    }
    if len(evaluation.stages) != 5:
        raise ValueError("Public LLM evidence must contain exactly five evaluation stages.")
    roles = {stage.evaluation_role for stage in evaluation.stages}
    if roles != {
        "frozen_baseline",
        "same_set_development_regression",
        "prompt_isolated_project_internal_blind_holdout",
        "prompt_v4_candidate_project_internal_blind_holdout",
        "prompt_v6_promoted_project_internal_blind_holdout",
    }:
        raise ValueError(
            "Public LLM evidence must distinguish development and all Holdout stages."
        )
    protocols = {
        str(protocol["protocol_id"]): (path, protocol)
        for path in holdout_protocol_paths
        for protocol in (_load_json_object(path),)
    }
    for stage in evaluation.stages:
        adjudication = adjudications.get(stage.adjudication_id)
        if adjudication is None:
            raise ValueError(f"Missing adjudication for {stage.stage_id}.")
        if (
            stage.source_run_id != adjudication.source_run_id
            or stage.source_summary_sha256 != adjudication.source_summary_sha256
            or stage.adjudication_independence != adjudication.independence
        ):
            raise ValueError(f"Adjudication provenance mismatch for {stage.stage_id}.")
        if stage.metrics.accepted_annotation_count != len(adjudication.reviews):
            raise ValueError(f"Human review count mismatch for {stage.stage_id}.")
        if stage.metrics.human_review_coverage_rate != 1.0:
            raise ValueError(f"Human review coverage is incomplete for {stage.stage_id}.")
        if stage.metrics.accepted_annotation_count > stage.external_case_count:
            raise ValueError(f"Accepted count exceeds external case count for {stage.stage_id}.")
        if stage.evaluation_role == "same_set_development_regression" and (
            not stage.same_case_set_as_baseline
            or stage.prompt_isolated_blind_evaluation
        ):
            raise ValueError("Development regression must be labelled same-set and non-blind.")
        if stage.evaluation_role in {
            "prompt_isolated_project_internal_blind_holdout",
            "prompt_v4_candidate_project_internal_blind_holdout",
            "prompt_v6_promoted_project_internal_blind_holdout",
        }:
            if (
                stage.same_case_set_as_baseline
                or not stage.prompt_isolated_blind_evaluation
                or stage.adjudication_independence != "project_internal"
            ):
                raise ValueError(
                    "Holdout must be prompt-isolated and blind with project-internal review."
                )
            protocol_entry = protocols.get(str(stage.preregistered_protocol_id))
            if protocol_entry is None:
                raise ValueError("Holdout protocol is required for release validation.")
            protocol_path, protocol = protocol_entry
            cases_path = Path(str(protocol["cases_file"]))
            prompt_path = Path(str(protocol["prompt_file"]))
            if (
                stage.preregistered_protocol_id != protocol.get("protocol_id")
                or stage.preregistered_protocol_sha256
                != _sha256_crlf_text(protocol_path)
                or not cases_path.is_file()
                or _sha256_crlf_text(cases_path) != protocol.get("cases_sha256")
                or not prompt_path.is_file()
                or _sha256(prompt_path) != protocol.get("prompt_sha256")
                or stage.case_count != protocol.get("case_count")
                or stage.external_case_count != protocol.get("external_case_count")
                or stage.deterministic_probe_count
                != protocol.get("deterministic_probe_count")
            ):
                raise ValueError("Holdout publication does not match its preregistration.")
            criteria = protocol.get("success_criteria")
            if not isinstance(criteria, dict):
                raise ValueError("Holdout protocol lacks success criteria.")
            expected_failures = []
            for criterion, threshold in criteria.items():
                metric_name = str(criterion).removesuffix("_minimum")
                observed = getattr(stage.metrics, metric_name, None)
                passed = (
                    observed is not None
                    and isinstance(threshold, int | float)
                    and (
                        observed >= float(threshold)
                        if str(criterion).endswith("_minimum")
                        else observed == float(threshold)
                    )
                )
                if not passed:
                    expected_failures.append(str(criterion))
            if (
                stage.failed_success_criteria != expected_failures
                or stage.success_criteria_met != (not expected_failures)
            ):
                raise ValueError("Holdout success-criteria outcome is inconsistent.")
    costs = [stage.metrics.estimated_cost_usd for stage in evaluation.stages]
    if evaluation.cost_status == "unavailable" and any(cost is not None for cost in costs):
        raise ValueError("Cost status conflicts with reported cost values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_human_review(args.summary, args.adjudication)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
