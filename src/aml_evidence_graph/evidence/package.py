"""Pydantic contracts for evidence-bound AML investigation drafts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuleEvidence(BaseModel):
    """A versioned approved-rule hit and its observed feature value."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_version: str
    feature: str
    observed_value: float
    threshold: float
    operator: Literal["gt", "gte", "lt", "lte"]
    explanation: str


class FeatureEvidence(BaseModel):
    """A model input or derived statistic selected for an alert explanation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | str | bool | None
    window: str | None = None
    source: str


class GraphEvidence(BaseModel):
    """Bounded graph evidence from a history-only snapshot."""

    model_config = ConfigDict(extra="forbid")

    source_node_index: int
    destination_node_index: int
    historical_source_out_degree: int
    historical_destination_in_degree: int
    prior_directed_edge_count: int
    prior_reverse_edge_count: int
    two_hop_intermediary_count: int
    two_hop_intermediary_node_indices: list[int] = Field(default_factory=list)
    snapshot_as_of: datetime
    interpretation_limit: str


class TypologyReference(BaseModel):
    """A retrieved Typology document, not an assertion that a case matches it."""

    model_config = ConfigDict(extra="forbid")

    typology_id: str
    version: str
    title: str
    source: str


class RiskEvidencePackage(BaseModel):
    """The only permissible factual input to an investigation drafting workflow."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    alert_id: str
    generated_at: datetime
    transaction_id: str = Field(
        description="Internal de-identified transaction/row ID; never a source-system identifier."
    )
    event_timestamp: datetime
    model_probabilities: dict[str, float]
    fusion_probability: float | None = None
    rule_hits: list[RuleEvidence] = Field(default_factory=list)
    key_features: list[FeatureEvidence] = Field(default_factory=list)
    graph_evidence: GraphEvidence | None = None
    typology_references: list[TypologyReference] = Field(default_factory=list)
    source_versions: dict[str, str] = Field(default_factory=dict)
    missing_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)

    @field_validator("model_probabilities")
    @classmethod
    def validate_probabilities(cls, values: dict[str, float]) -> dict[str, float]:
        if not values:
            raise ValueError("At least one model probability is required.")
        if any(not 0 <= value <= 1 for value in values.values()):
            raise ValueError("Model probabilities must be in [0, 1].")
        return values

    @field_validator("fusion_probability")
    @classmethod
    def validate_fusion_probability(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("fusion_probability must be in [0, 1].")
        return value


class InvestigationReport(BaseModel):
    """A draft for human review; all facts are copied from the evidence package."""

    model_config = ConfigDict(extra="forbid")

    report_schema_version: Literal["1.0"] = "1.0"
    alert_id: str
    status: Literal["draft_requires_human_review", "rejected_facts"] = (
        "draft_requires_human_review"
    )
    factual_summary: list[str]
    typology_considerations: list[str]
    missing_evidence: list[str]
    uncertainty_notes: list[str]
    fact_snapshot: dict[str, object]
    review_instruction: str
    llm_annotation: InvestigationAnnotation | None = None
    fact_validation: FactValidationResult | None = None
    tool_call_count: int = Field(default=0, ge=0, le=4)


class InvestigationAnnotation(BaseModel):
    """Non-factual LLM analysis that must cite only known evidence field paths."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: str
    model_name: str
    evidence_references: list[str] = Field(default_factory=list)
    analytical_considerations: list[str] = Field(default_factory=list)
    recommended_questions: list[str] = Field(default_factory=list)
    usage: AnnotationUsage | None = None


class AnnotationUsage(BaseModel):
    """Provider-reported token usage and an optional configured cost estimate."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class FactValidationResult(BaseModel):
    """Outcome of validating LLM references and prohibited factual claims."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[str] = Field(default_factory=list)
