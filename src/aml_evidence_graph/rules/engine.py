"""Safe, versioned numeric-threshold rules without dynamic expression evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from aml_evidence_graph.data.contract import CANONICAL

RuleOperator = Literal["gt", "gte", "lt", "lte"]


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    version: str
    status: str
    effective_from: date
    feature: str
    operator: RuleOperator
    threshold: float | None
    explanation_template: str
    owner: str = "unassigned"
    required_features: tuple[str, ...] = ()
    effective_to: date | None = None
    approval_reference: str | None = None
    backtest_summary: str | None = None

    @property
    def active(self) -> bool:
        return (
            self.status == "approved"
            and self.threshold is not None
            and self.approval_reference is not None
            and self.backtest_summary is not None
        )

    def is_active_on(self, as_of_date: date) -> bool:
        """Check approval and effective date range for a historical transaction."""
        return (
            self.active
            and self.effective_from <= as_of_date
            and (self.effective_to is None or as_of_date <= self.effective_to)
        )


@dataclass(frozen=True)
class RuleHit:
    transaction_id: str
    rule_id: str
    rule_version: str
    feature: str
    observed_value: float
    threshold: float
    operator: RuleOperator
    explanation: str


def _parse_rule(raw: dict[str, Any], *, version: str) -> RuleDefinition:
    if raw.get("kind") != "numeric_threshold":
        raise ValueError(f"Rule {raw.get('rule_id')} has unsupported kind.")
    operator = raw.get("operator")
    if operator not in {"gt", "gte", "lt", "lte"}:
        raise ValueError(f"Rule {raw.get('rule_id')} has invalid operator.")
    threshold = raw.get("parameters", {}).get("threshold")
    status = str(raw["status"])
    if status not in {"draft", "approved", "retired"}:
        raise ValueError(f"Rule {raw.get('rule_id')} has invalid status.")
    required_features = tuple(str(value) for value in raw.get("required_features", []))
    feature = str(raw["feature"])
    if feature not in required_features:
        raise ValueError(
            f"Rule {raw.get('rule_id')} must list its feature in required_features."
        )
    effective_to = raw.get("effective_to")
    rule = RuleDefinition(
        rule_id=str(raw["rule_id"]),
        version=version,
        status=status,
        effective_from=date.fromisoformat(str(raw["effective_from"])),
        feature=feature,
        operator=operator,
        threshold=float(threshold) if threshold is not None else None,
        explanation_template=str(raw["explanation_template"]),
        owner=str(raw["owner"]),
        required_features=required_features,
        effective_to=(
            date.fromisoformat(str(effective_to)) if effective_to is not None else None
        ),
        approval_reference=(
            str(raw["approval_reference"])
            if raw.get("approval_reference") is not None
            else None
        ),
        backtest_summary=(
            str(raw["backtest_summary"])
            if raw.get("backtest_summary") is not None
            else None
        ),
    )
    if rule.effective_to is not None and rule.effective_to < rule.effective_from:
        raise ValueError(f"Rule {rule.rule_id} has effective_to before effective_from.")
    if status == "approved" and not rule.active:
        raise ValueError(
            f"Approved rule {rule.rule_id} requires threshold, approval_reference, "
            "and backtest_summary."
        )
    return rule


def load_rules(path: Path) -> list[RuleDefinition]:
    """Load validated rule definitions from a YAML file."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "version" not in document or "rules" not in document:
        raise ValueError("Rule file must contain version and rules.")
    rules = [_parse_rule(raw, version=str(document["version"])) for raw in document["rules"]]
    rule_ids = [rule.rule_id for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Rule IDs must be unique within a rule file.")
    return rules


def _hit_mask(values: pd.Series, rule: RuleDefinition) -> pd.Series:
    assert rule.threshold is not None
    if rule.operator == "gt":
        return values > rule.threshold
    if rule.operator == "gte":
        return values >= rule.threshold
    if rule.operator == "lt":
        return values < rule.threshold
    return values <= rule.threshold


def apply_rules(
    frame: pd.DataFrame,
    rules: list[RuleDefinition],
    *,
    as_of_date: date,
) -> tuple[pd.DataFrame, list[RuleHit]]:
    """Produce boolean rule features and structured hits for active approved rules."""
    if CANONICAL.transaction_id not in frame:
        raise ValueError("Rule evaluation requires canonical transaction_id.")
    output = pd.DataFrame(index=frame.index)
    hits: list[RuleHit] = []

    for rule in rules:
        if not rule.is_active_on(as_of_date):
            continue
        if rule.feature not in frame:
            raise ValueError(f"Rule {rule.rule_id} requires missing feature {rule.feature}.")
        values = pd.to_numeric(frame[rule.feature], errors="raise")
        mask = _hit_mask(values, rule)
        output[f"rule_{rule.rule_id}_hit"] = mask.astype("int8")
        for index in frame.index[mask]:
            observed = float(values.loc[index])
            assert rule.threshold is not None
            hits.append(
                RuleHit(
                    transaction_id=str(frame.at[index, CANONICAL.transaction_id]),
                    rule_id=rule.rule_id,
                    rule_version=rule.version,
                    feature=rule.feature,
                    observed_value=observed,
                    threshold=rule.threshold,
                    operator=rule.operator,
                    explanation=rule.explanation_template,
                )
            )
    return output, hits
