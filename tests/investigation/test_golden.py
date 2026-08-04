import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from aml_evidence_graph.evidence.package import (
    AnnotationUsage,
    InvestigationAnnotation,
    RiskEvidencePackage,
)
from aml_evidence_graph.evidence.typology import LocalBM25TypologyRetriever, TypologyDocument
from aml_evidence_graph.investigation.golden import (
    GoldenCase,
    evaluate_golden_set,
    load_golden_cases,
)


def _crlf_sha256(path: Path) -> str:
    normalized = (
        path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\r\n")
        .encode()
    )
    return hashlib.sha256(normalized).hexdigest()


def test_llm_holdout_v1_is_frozen_and_disjoint_from_prompt_development_set() -> None:
    protocol = json.loads(
        Path("golden/llm_holdout_protocol_v1.json").read_text(encoding="utf-8")
    )
    holdout_path = Path(protocol["cases_file"])
    holdout_cases = load_golden_cases(holdout_path)
    development_cases = load_golden_cases(Path("golden/cases_v1.json"))

    assert _crlf_sha256(holdout_path) == protocol["cases_sha256"]
    assert (
        hashlib.sha256(
            Path(protocol["prompt_file"]).read_bytes()
        ).hexdigest()
        == protocol["prompt_sha256"]
    )
    assert len(holdout_cases) == protocol["case_count"] == 24
    assert sum(case.injected_annotation is None for case in holdout_cases) == 20
    assert sum(case.injected_annotation is not None for case in holdout_cases) == 4
    assert {case.case_id for case in holdout_cases}.isdisjoint(
        case.case_id for case in development_cases
    )


def test_llm_holdout_v2_is_frozen_and_disjoint_from_all_earlier_sets() -> None:
    protocol = json.loads(
        Path("golden/llm_holdout_protocol_v2.json").read_text(encoding="utf-8")
    )
    holdout_path = Path(protocol["cases_file"])
    holdout_cases = load_golden_cases(holdout_path)
    prior_cases = [
        *load_golden_cases(Path("golden/cases_v1.json")),
        *load_golden_cases(Path("golden/llm_holdout_cases_v1.json")),
    ]

    assert _crlf_sha256(holdout_path) == protocol["cases_sha256"]
    assert hashlib.sha256(Path(protocol["prompt_file"]).read_bytes()).hexdigest() == (
        protocol["prompt_sha256"]
    )
    assert len(holdout_cases) == protocol["case_count"] == 24
    assert sum(case.injected_annotation is None for case in holdout_cases) == 20
    assert sum(case.injected_annotation is not None for case in holdout_cases) == 4
    assert {case.case_id for case in holdout_cases}.isdisjoint(
        case.case_id for case in prior_cases
    )


def test_golden_set_tracks_hallucination_intercept_and_latency() -> None:
    evidence = RiskEvidencePackage(
        alert_id="golden-adv-1",
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
        transaction_id="mock-txn-adv-1",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities={"catboost": 0.5},
        missing_evidence=["narrative unavailable"],
    )
    retriever = LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="TYPOLOGY-TEST",
                version="1",
                title="Test",
                body="transaction risk",
                source="test",
            )
        ]
    )
    summary = evaluate_golden_set(
        [
            GoldenCase(
                case_id="golden-adv-1",
                evidence=evidence,
                case_category="adversarial",
                expect_rejected_facts=True,
                injected_annotation=InvestigationAnnotation(
                    prompt_version="probe",
                    model_name="injected",
                    analytical_considerations=["Amount was 99."],
                ),
            ),
            GoldenCase(
                case_id="golden-low-1",
                evidence=evidence,
                case_category="low_evidence",
            ),
        ],
        retriever=retriever,
    )
    assert summary.hallucination_intercept_rate == 1.0
    assert summary.no_evidence_refusal_rate == 1.0
    assert summary.latency_p50_ms >= 0
    assert summary.latency_p95_ms >= summary.latency_p50_ms


def test_golden_set_tracks_fact_and_tool_limits() -> None:
    evidence = RiskEvidencePackage(
        alert_id="golden-1",
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
        transaction_id="mock-txn-1",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities={"catboost": 0.5},
    )
    retriever = LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="TYPOLOGY-TEST",
                version="1",
                title="Test",
                body="transaction risk",
                source="test",
            )
        ]
    )

    summary = evaluate_golden_set(
        [
            GoldenCase(
                case_id="golden-1",
                evidence=evidence,
                expected_typology_ids=["TYPOLOGY-TEST"],
            )
        ],
        retriever=retriever,
    )

    assert summary.schema_compliance_rate == 1
    assert summary.fact_snapshot_match_rate == 1
    assert summary.tool_limit_pass_rate == 1
    assert summary.llm_annotation_rate == 0


def test_golden_set_tracks_llm_prompt_version_and_usage() -> None:
    evidence = RiskEvidencePackage(
        alert_id="golden-llm-1",
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
        transaction_id="mock-txn-llm-1",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities={"catboost": 0.5},
    )
    retriever = LocalBM25TypologyRetriever(
        [
            TypologyDocument(
                typology_id="TYPOLOGY-TEST",
                version="1",
                title="Test",
                body="transaction risk",
                source="test",
            )
        ]
    )

    class UsageAnnotator:
        def annotate(
            self,
            evidence: RiskEvidencePackage,
            references: list[object],
        ) -> InvestigationAnnotation:
            return InvestigationAnnotation(
                prompt_version="golden-prompt-v1",
                model_name="mock-model",
                analytical_considerations=["Consider corroborating the available evidence."],
                usage=AnnotationUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    estimated_cost_usd=0.01,
                ),
            )

    summary = evaluate_golden_set(
        [GoldenCase(case_id="golden-llm-1", evidence=evidence)],
        retriever=retriever,
        annotator=UsageAnnotator(),
    )

    assert summary.llm_annotation_rate == 1
    assert summary.external_case_count == 1
    assert summary.external_parse_success_rate == 1
    assert summary.external_fact_validation_pass_rate == 1
    assert summary.token_usage_coverage_rate == 1
    assert summary.reported_prompt_tokens == 10
    assert summary.estimated_cost_usd == 0.01
    assert summary.prompt_versions == ["golden-prompt-v1"]
    assert summary.cases[0].analytical_considerations == [
        "Consider corroborating the available evidence."
    ]
