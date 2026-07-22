"""Golden-set checks for evidence-bound investigation reports."""

from __future__ import annotations

from dataclasses import dataclass

from aml_evidence_graph.evidence.package import InvestigationReport, RiskEvidencePackage


@dataclass(frozen=True)
class GoldenSetResult:
    """Minimal objective checks before a draft can be presented to reviewers."""

    schema_valid: bool
    alert_id_matches: bool
    fact_snapshot_matches: bool
    evidence_coverage: float
    no_evidence_refusal: bool


def evaluate_investigation_report(
    evidence: RiskEvidencePackage,
    report: InvestigationReport,
) -> GoldenSetResult:
    """Check schema, factual identity, evidence coverage, and no-evidence behavior."""
    expected_snapshot = evidence.model_dump(mode="json")
    evidence_items = len(evidence.model_probabilities) + len(evidence.rule_hits) + len(
        evidence.key_features
    )
    covered_items = len(report.factual_summary)
    coverage = min(1.0, covered_items / evidence_items) if evidence_items else 0.0
    no_evidence_refusal = bool(evidence.missing_evidence) <= bool(report.missing_evidence)
    return GoldenSetResult(
        schema_valid=report.report_schema_version == "1.0",
        alert_id_matches=report.alert_id == evidence.alert_id,
        fact_snapshot_matches=report.fact_snapshot == expected_snapshot,
        evidence_coverage=coverage,
        no_evidence_refusal=no_evidence_refusal,
    )
