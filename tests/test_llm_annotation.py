import json
from datetime import UTC, datetime

import httpx

from aml_evidence_graph.evidence.package import (
    InvestigationAnnotation,
    RiskEvidencePackage,
    TypologyReference,
)
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.llm import (
    ECNUAnnotationClient,
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
    assert validation.valid
    assert annotation.usage is not None
    assert annotation.usage.total_tokens == 20


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
    assert any("deterministic evidence template" in item for item in report.uncertainty_notes)


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
