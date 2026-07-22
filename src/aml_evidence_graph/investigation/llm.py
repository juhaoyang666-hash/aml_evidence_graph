"""OpenAI-compatible ECNU annotation client with evidence-bound fact validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from aml_evidence_graph.evidence.package import (
    AnnotationUsage,
    FactValidationResult,
    InvestigationAnnotation,
    RiskEvidencePackage,
    TypologyReference,
)
from aml_evidence_graph.settings import Settings

PROMPT_VERSION = "ecnu-risk-evidence-v1"
_FORBIDDEN_FACT_PATTERN = re.compile(
    r"(?:\b\d+(?:\.\d+)?\b|\b(?:transaction|account|alert)[_-][A-Za-z0-9]+\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptConfiguration:
    """Auditable static prompt configuration for the constrained ECNU annotation call."""

    version: str
    system_instructions: str
    temperature: float
    max_tokens: int


DEFAULT_PROMPT_CONFIGURATION = PromptConfiguration(
    version=PROMPT_VERSION,
    system_instructions=(
        "You are an AML investigation annotation assistant. "
        "You do not score risk and must not state numbers, dates, "
        "account IDs, transaction IDs, rankings, or unsupported facts. "
        "Return JSON only with evidence_references, "
        "analytical_considerations, and recommended_questions. "
        "Use only supplied evidence reference paths. "
        "Analytical text must be non-factual and conditional."
    ),
    temperature=0,
    max_tokens=500,
)


def load_prompt_configuration(path: Path) -> PromptConfiguration:
    """Load and validate a versioned prompt file before an external call is enabled."""
    if not path.is_file():
        raise FileNotFoundError(f"LLM prompt configuration does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LLM prompt configuration must be a YAML object.")
    try:
        configuration = PromptConfiguration(
            version=str(payload["version"]),
            system_instructions=str(payload["system_instructions"]),
            temperature=float(payload["temperature"]),
            max_tokens=int(payload["max_tokens"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("LLM prompt configuration is incomplete or invalid.") from error
    if not configuration.version or not configuration.system_instructions:
        raise ValueError("LLM prompt configuration requires version and instructions.")
    if not 0 <= configuration.temperature <= 2 or configuration.max_tokens < 1:
        raise ValueError("LLM prompt configuration has invalid generation limits.")
    return configuration


class EvidenceAnnotationClient(Protocol):
    """A client allowed to produce only referenced analytical annotations."""

    def annotate(
        self,
        evidence: RiskEvidencePackage,
        references: list[TypologyReference],
    ) -> InvestigationAnnotation: ...


def minimize_evidence_for_llm(
    evidence: RiskEvidencePackage,
    references: list[TypologyReference],
) -> dict[str, Any]:
    """Remove IDs/timestamps/source metadata before an external annotation request."""
    return {
        "model_probability_names": sorted(evidence.model_probabilities),
        "has_fusion_probability": evidence.fusion_probability is not None,
        "rule_ids": [rule.rule_id for rule in evidence.rule_hits],
        "feature_names": [feature.name for feature in evidence.key_features],
        "has_graph_evidence": evidence.graph_evidence is not None,
        "typology_references": [
            {
                "typology_id": reference.typology_id,
                "version": reference.version,
                "title": reference.title,
            }
            for reference in references
        ],
        "missing_evidence_categories": evidence.missing_evidence,
        "uncertainty_categories": evidence.uncertainty_notes,
    }


def allowed_evidence_references(
    evidence: RiskEvidencePackage,
    references: list[TypologyReference],
) -> set[str]:
    """Enumerate references that an annotation may cite without inventing facts."""
    allowed = {
        *[f"model_probabilities.{name}" for name in evidence.model_probabilities],
        *[f"rule_hits[{index}]" for index in range(len(evidence.rule_hits))],
        *[f"key_features[{index}]" for index in range(len(evidence.key_features))],
        *[
            f"typology_references[{index}]"
            for index in range(len(references))
        ],
        "fusion_probability",
        "graph_evidence",
        "missing_evidence",
        "uncertainty_notes",
    }
    if evidence.fusion_probability is None:
        allowed.remove("fusion_probability")
    if evidence.graph_evidence is None:
        allowed.remove("graph_evidence")
    return allowed


def validate_annotation(
    annotation: InvestigationAnnotation,
    *,
    evidence: RiskEvidencePackage,
    references: list[TypologyReference],
) -> FactValidationResult:
    """Reject unsupported field citations and annotations containing factual values/tokens."""
    errors: list[str] = []
    allowed = allowed_evidence_references(evidence, references)
    unsupported = sorted(set(annotation.evidence_references).difference(allowed))
    if unsupported:
        errors.append(f"Unsupported evidence references: {', '.join(unsupported)}")
    for section_name, values in (
        ("analytical_considerations", annotation.analytical_considerations),
        ("recommended_questions", annotation.recommended_questions),
    ):
        for value in values:
            if _FORBIDDEN_FACT_PATTERN.search(value):
                errors.append(
                    f"{section_name} must not introduce numeric values or entity tokens."
                )
                break
    return FactValidationResult(valid=not errors, errors=errors)


class ECNUAnnotationClient:
    """Minimal OpenAI-compatible client for ECNU, limited to anonymous annotations."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://chat.ecnu.edu.cn/open/api/v1",
        model_name: str = "ecnu-max",
        timeout_seconds: float = 30,
        input_cost_per_million_tokens_usd: float | None = None,
        output_cost_per_million_tokens_usd: float | None = None,
        prompt_configuration: PromptConfiguration | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.input_cost_per_million_tokens_usd = input_cost_per_million_tokens_usd
        self.output_cost_per_million_tokens_usd = output_cost_per_million_tokens_usd
        self.prompt_configuration = prompt_configuration or DEFAULT_PROMPT_CONFIGURATION
        self._http_client = http_client

    @classmethod
    def from_settings(cls, settings: Settings) -> ECNUAnnotationClient:
        if not settings.llm_enabled:
            raise RuntimeError("AML_LLM_ENABLED must be true before creating an LLM client.")
        return cls(
            api_key=settings.require_llm_api_key(),
            base_url=settings.llm_base_url,
            model_name=settings.llm_model,
            input_cost_per_million_tokens_usd=settings.llm_input_cost_per_million_tokens_usd,
            output_cost_per_million_tokens_usd=settings.llm_output_cost_per_million_tokens_usd,
            prompt_configuration=load_prompt_configuration(settings.llm_prompt_config_path),
        )

    def annotate(
        self,
        evidence: RiskEvidencePackage,
        references: list[TypologyReference],
    ) -> InvestigationAnnotation:
        """Request constrained JSON; only minimized, deidentified evidence is sent."""
        request_payload = {
            "model": self.model_name,
            "temperature": self.prompt_configuration.temperature,
            "max_tokens": self.prompt_configuration.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": self.prompt_configuration.system_instructions,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "prompt_version": self.prompt_configuration.version,
                            "allowed_evidence_references": sorted(
                                allowed_evidence_references(evidence, references)
                            ),
                            "deidentified_evidence": minimize_evidence_for_llm(
                                evidence,
                                references,
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self._http_client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self._http_client is None
        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_payload,
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            if close_client:
                client.close()
        try:
            content = payload["choices"][0]["message"]["content"]
            decoded = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("ECNU response did not contain a valid JSON annotation.") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("ECNU response annotation must be a JSON object.")

        def normalize_text_list(field: str) -> list[str]:
            value = decoded.get(field, [])
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return value
            raise RuntimeError(f"ECNU response field {field} must be text or a text list.")

        usage = _parse_usage(
            payload.get("usage"),
            input_cost_per_million_tokens_usd=self.input_cost_per_million_tokens_usd,
            output_cost_per_million_tokens_usd=self.output_cost_per_million_tokens_usd,
        )

        return InvestigationAnnotation(
            prompt_version=self.prompt_configuration.version,
            model_name=self.model_name,
            evidence_references=normalize_text_list("evidence_references"),
            analytical_considerations=normalize_text_list("analytical_considerations"),
            recommended_questions=normalize_text_list("recommended_questions"),
            usage=usage,
        )


def _parse_usage(
    raw_usage: object,
    *,
    input_cost_per_million_tokens_usd: float | None,
    output_cost_per_million_tokens_usd: float | None,
) -> AnnotationUsage | None:
    """Normalize optional provider usage without inventing unavailable token counts or prices."""
    if not isinstance(raw_usage, dict):
        return None

    def token_value(name: str) -> int | None:
        value = raw_usage.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    prompt_tokens = token_value("prompt_tokens")
    completion_tokens = token_value("completion_tokens")
    total_tokens = token_value("total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    estimated_cost_usd = None
    if (
        prompt_tokens is not None
        and completion_tokens is not None
        and input_cost_per_million_tokens_usd is not None
        and output_cost_per_million_tokens_usd is not None
    ):
        estimated_cost_usd = (
            prompt_tokens * input_cost_per_million_tokens_usd
            + completion_tokens * output_cost_per_million_tokens_usd
        ) / 1_000_000
    return AnnotationUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )
