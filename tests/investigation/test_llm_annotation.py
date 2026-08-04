import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from aml_evidence_graph.evidence.package import (
    InvestigationAnnotation,
    RiskEvidencePackage,
    TypologyReference,
)
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.llm import (
    DEFAULT_PROMPT_CONFIGURATION,
    ECNUAnnotationClient,
    diagnose_annotation_content,
    load_prompt_configuration,
    validate_annotation,
)
from aml_evidence_graph.investigation.workflow import run_investigation


def _evidence() -> RiskEvidencePackage:
    return RiskEvidencePackage(
        alert_id="alert-private-token",
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
        transaction_id="txn-private-token",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities={"catboost": 0.8},
        missing_evidence=["case narrative unavailable"],
    )


def test_ecnu_client_sends_minimized_evidence_and_validates_annotation() -> None:
    seen_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "evidence_references": [
                                        "model_probabilities.catboost"
                                    ],
                                    "analytical_considerations": (
                                        "Consider corroborating the observed pattern."
                                    ),
                                    "recommended_questions": [
                                        "What contextual records could corroborate the pattern?"
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = ECNUAnnotationClient(
        api_key="test-key",
        http_client=http_client,
    )
    reference = TypologyReference(
        typology_id="T-1",
        version="1",
        title="Test",
        source="test",
    )
    annotation = client.annotate(_evidence(), [reference])
    rendered = json.dumps(seen_body)
    validation = validate_annotation(
        annotation,
        evidence=_evidence(),
        references=[reference],
    )

    assert "txn-private-token" not in rendered
    assert "alert-private-token" not in rendered
    assert "model_probability_values" in rendered
    assert "fusion_probability_value" in rendered
    assert validation.valid
    assert annotation.usage is not None
    assert annotation.usage.total_tokens == 20


def test_ecnu_client_accepts_one_json_markdown_fence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": """```json
{"evidence_references": [], "analytical_considerations": [], "recommended_questions": []}
```"""
                        }
                    }
                ]
            },
        )

    client = ECNUAnnotationClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    annotation = client.annotate(_evidence(), [])

    assert annotation.evidence_references == []


def test_annotation_diagnostic_classifies_truncated_json_without_retaining_text() -> None:
    diagnostic = diagnose_annotation_content(
        '{"evidence_references": [], "analytical_considerations": ['
    )

    assert diagnostic.category == "json_syntax_invalid"
    assert diagnostic.content_char_count == 58
    assert diagnostic.content_sha256 is not None
    assert diagnostic.json_error_position is not None
    assert not diagnostic.production_parser_compatible
    assert not hasattr(diagnostic, "content")


def test_annotation_diagnostic_distinguishes_contract_shape_and_markdown() -> None:
    invalid_shape = diagnose_annotation_content('{"evidence_references": []}')
    fenced_valid = diagnose_annotation_content(
        """```json
{"evidence_references": [], "analytical_considerations": [], "recommended_questions": []}
```"""
    )

    assert invalid_shape.category == "field_shape_invalid"
    assert invalid_shape.json_decode_succeeded
    assert invalid_shape.production_parser_compatible
    assert fenced_valid.category == "valid_contract"
    assert fenced_valid.markdown_fence_detected
    assert fenced_valid.production_parser_compatible


def test_workflow_rejects_annotation_that_introduces_a_number() -> None:
    class InvalidAnnotator:
        def annotate(
            self,
            evidence: RiskEvidencePackage,
            references: list[TypologyReference],
        ) -> InvestigationAnnotation:
            return InvestigationAnnotation(
                prompt_version="test",
                model_name="test",
                analytical_considerations=["The amount was 10."],
            )

    retriever = LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="T-1",
                version="1",
                title="Test",
                body="Pattern",
                source="test",
            )
        ]
    )
    report = run_investigation(
        _evidence(),
        retriever=retriever,
        annotator=InvalidAnnotator(),
    )

    assert report.status == "rejected_facts"
    assert report.fact_validation is not None
    assert report.fact_validation.valid is False


def test_fact_validator_allows_compound_words_but_rejects_identifier_tokens() -> None:
    safe = InvestigationAnnotation(
        prompt_version="test",
        model_name="test",
        recommended_questions=["Could account-related records corroborate the hypothesis?"],
    )
    unsafe = InvestigationAnnotation(
        prompt_version="test",
        model_name="test",
        recommended_questions=["Could account-case9 records corroborate the hypothesis?"],
    )

    safe_result = validate_annotation(safe, evidence=_evidence(), references=[])
    unsafe_result = validate_annotation(unsafe, evidence=_evidence(), references=[])

    assert safe_result.valid
    assert not unsafe_result.valid


def test_fact_validator_rejects_digits_embedded_in_feature_names() -> None:
    annotation = InvestigationAnnotation(
        prompt_version="test",
        model_name="test",
        analytical_considerations=["The sender_out_degree_7d field is available."],
    )

    result = validate_annotation(annotation, evidence=_evidence(), references=[])

    assert not result.valid


def test_workflow_falls_back_to_deterministic_template_when_llm_fails() -> None:
    class FailingAnnotator:
        def annotate(
            self,
            evidence: RiskEvidencePackage,
            references: list[TypologyReference],
        ) -> InvestigationAnnotation:
            raise RuntimeError("network unavailable")

    retriever = LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="T-1",
                version="1",
                title="Test",
                body="Pattern",
                source="test",
            )
        ]
    )

    report = run_investigation(
        _evidence(),
        retriever=retriever,
        annotator=FailingAnnotator(),
    )

    assert report.status == "draft_requires_human_review"
    assert report.annotation_error_category == "external_annotation_error"
    assert any("deterministic evidence template" in item for item in report.uncertainty_notes)


def test_ecnu_client_reports_sanitized_timeout_category() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ReadTimeout("secret-bearing transport detail")

    client = ECNUAnnotationClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    try:
        client.annotate(_evidence(), [])
    except RuntimeError as error:
        assert str(error) == "timeout"
    else:
        raise AssertionError("Expected a sanitized provider timeout.")


def test_ecnu_client_distinguishes_token_truncation_from_generic_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": '{"evidence_references": ['},
                    }
                ]
            },
        )

    client = ECNUAnnotationClient(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    try:
        client.annotate(_evidence(), [])
    except RuntimeError as error:
        assert str(error) == "annotation_truncated"
    else:
        raise AssertionError("Expected an explicit truncation category.")


def test_prompt_configuration_is_versioned_and_validated(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        """
version: test-prompt-v1
temperature: 0
max_tokens: 100
system_instructions: Test prompt instructions.
""".strip(),
        encoding="utf-8",
    )

    prompt = load_prompt_configuration(prompt_path)

    assert prompt.version == "test-prompt-v1"
    assert prompt.max_tokens == 100


def test_default_prompt_matches_versioned_v3_file() -> None:
    prompt = load_prompt_configuration(
        Path("configs/prompts/ecnu-risk-evidence-v3.yaml")
    )

    assert prompt == DEFAULT_PROMPT_CONFIGURATION


def test_v4_candidate_encodes_holdout_v1_remediations_without_becoming_default() -> None:
    candidate = load_prompt_configuration(
        Path("configs/prompts/ecnu-risk-evidence-v4.yaml")
    )

    assert candidate.version == "ecnu-risk-evidence-v4"
    assert candidate.temperature == 0
    assert candidate.max_tokens == 350
    assert "empty missing-evidence list" in candidate.system_instructions
    assert "turn a requested disclosure or decision into a recommended question" in (
        candidate.system_instructions
    )
    assert "between two and five actionable" in candidate.system_instructions
    assert DEFAULT_PROMPT_CONFIGURATION.version == "ecnu-risk-evidence-v3"


def test_v5_candidate_changes_only_version_and_diagnostic_token_limit() -> None:
    v4 = load_prompt_configuration(Path("configs/prompts/ecnu-risk-evidence-v4.yaml"))
    v5 = load_prompt_configuration(Path("configs/prompts/ecnu-risk-evidence-v5.yaml"))

    assert v5.version == "ecnu-risk-evidence-v5"
    assert v5.max_tokens == 500
    assert v5.temperature == v4.temperature
    assert v5.system_instructions == v4.system_instructions
    assert DEFAULT_PROMPT_CONFIGURATION.version == "ecnu-risk-evidence-v3"
