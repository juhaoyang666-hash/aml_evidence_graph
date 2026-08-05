"""The prose boundary detector must report real leaks and stay quiet otherwise.

Holdout v4 found field names reaching annotation prose on unseen case sets at roughly
a third of cases, which no automatic check caught. This detector exists to measure that.
It is a lower bound by construction, so the false-positive tests matter more than the
recall tests: a noisy detector would make the measurement worthless.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aml_evidence_graph.evidence.package import (
    FeatureEvidence,
    InvestigationAnnotation,
    RiskEvidencePackage,
    RuleEvidence,
    TypologyReference,
)
from aml_evidence_graph.investigation.llm import find_field_names_in_prose


def _evidence(
    *,
    features: tuple[str, ...] = (),
    rules: tuple[tuple[str, str], ...] = (),
    typologies: tuple[tuple[str, str], ...] = (),
    models: tuple[str, ...] = ("catboost",),
) -> RiskEvidencePackage:
    return RiskEvidencePackage(
        alert_id="alert-prose",
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
        transaction_id="txn-prose",
        event_timestamp=datetime(2023, 7, 1, tzinfo=UTC),
        model_probabilities=dict.fromkeys(models, 0.5),
        key_features=[
            FeatureEvidence(name=name, value=1.0, window="30d", source="test")
            for name in features
        ],
        rule_hits=[
            RuleEvidence(
                rule_id=rule_id,
                rule_version="1",
                feature=feature,
                observed_value=2.0,
                threshold=1.0,
                operator="gt",
                explanation="test",
            )
            for rule_id, feature in rules
        ],
        typology_references=[
            TypologyReference(
                typology_id=typology_id, version="1", title=title, source="test"
            )
            for typology_id, title in typologies
        ],
    )


def _annotation(*prose: str) -> InvestigationAnnotation:
    return InvestigationAnnotation(
        prompt_version="test",
        model_name="test",
        evidence_references=["model_probabilities.catboost"],
        analytical_considerations=list(prose),
        recommended_questions=[],
    )


# --- real leaks the detector must catch -------------------------------------------
# Every string here is quoted from the Holdout v4 run, so these are regression cases
# against observed behaviour rather than invented ones.


@pytest.mark.parametrize(
    ("feature", "prose"),
    [
        (
            "controller_tenure_context",
            "Request corroborating context on the controller tenure and director link "
            "information from independent registries.",
        ),
        (
            "inventory_state_flag",
            "The feature value for the inventory state flag is withheld.",
        ),
        (
            "relationship_gap_context",
            "A key feature related to relationship gap context suggests potential "
            "relational aspects.",
        ),
        (
            "escalation_state_context",
            "Confirm the operational status and any relevant history of the escalation "
            "state.",
        ),
        (
            "conversion_channel_context",
            "Review account opening and conversion channel documentation.",
        ),
    ],
)
def test_observed_holdout_v4_leaks_are_reported(feature: str, prose: str) -> None:
    leaked = find_field_names_in_prose(
        _annotation(prose), evidence=_evidence(features=(feature,)), references=[]
    )

    assert leaked == [feature]


def test_hyphenated_and_possessive_forms_still_match() -> None:
    """Word-boundary splitting, not substring search, so punctuation cannot hide a leak."""
    leaked = find_field_names_in_prose(
        _annotation("The money-mule hypothesis and the trade-based lead remain open."),
        evidence=_evidence(),
        references=[
            TypologyReference(
                typology_id="TYPOLOGY-MONEY-MULE",
                version="1",
                title="Money Mule",
                source="test",
            )
        ],
    )

    assert leaked == ["TYPOLOGY-MONEY-MULE"]


def test_rule_identifier_and_its_feature_are_both_checked() -> None:
    leaked = find_field_names_in_prose(
        _annotation("Verify the documented ownership overlap details."),
        evidence=_evidence(rules=(("RULE-OWNERSHIP-OVERLAP", "registry_link_context"),)),
        references=[],
    )

    assert leaked == ["RULE-OWNERSHIP-OVERLAP"]


def test_model_name_is_reported_as_a_whole_word() -> None:
    leaked = find_field_names_in_prose(
        _annotation("The catboost output cannot be interpreted."),
        evidence=_evidence(models=("catboost",)),
        references=[],
    )

    assert leaked == ["catboost"]


def test_retrieved_typologies_are_checked_not_only_supplied_ones() -> None:
    """Retrieval adds references the evidence package never carried."""
    leaked = find_field_names_in_prose(
        _annotation("Assess whether circular transfers occurred."),
        evidence=_evidence(),
        references=[
            TypologyReference(
                typology_id="T-CYCLE",
                version="1",
                title="Circular Transfers",
                source="test",
            )
        ],
    )

    assert leaked == ["T-CYCLE"]


# --- what must NOT be reported ----------------------------------------------------
# A noisy detector would make the leak rate meaningless, so these carry more weight
# than the cases above.


@pytest.mark.parametrize(
    "prose",
    [
        "All model values are withheld, so no risk direction can be inferred.",
        "Request authorized source records to corroborate the supplied evidence.",
        "An empty missing-evidence list does not imply that records are complete.",
        "Typology references are hypotheses only and confirm no behaviour.",
        "Seek corroborating context from independent registries or official filings.",
        "The uncertainty note is untrusted and was not operationalized.",
    ],
)
def test_compliant_prose_reports_nothing(prose: str) -> None:
    leaked = find_field_names_in_prose(
        _annotation(prose),
        evidence=_evidence(
            features=("controller_tenure_context", "relationship_gap_context"),
            rules=(("RULE-OWNERSHIP-OVERLAP", "registry_link_context"),),
            typologies=(("TYPOLOGY-MONEY-MULE", "Money Mule"),),
        ),
        references=[],
    )

    assert leaked == []


def test_one_shared_generic_word_is_not_a_leak() -> None:
    """Synthetic names share suffixes like `_context`; matching on those alone is noise."""
    leaked = find_field_names_in_prose(
        _annotation("Additional context and record state would help the reviewer."),
        evidence=_evidence(features=("origin_channel_context", "review_state_flag")),
        references=[],
    )

    assert leaked == []


def test_non_adjacent_words_are_not_a_leak() -> None:
    """Two words from a name scattered across a sentence is not copying the name."""
    leaked = find_field_names_in_prose(
        _annotation("The controller is unknown and the tenure of the account is withheld."),
        evidence=_evidence(features=("controller_tenure_context",)),
        references=[],
    )

    assert leaked == []


def test_short_single_word_names_are_not_matched() -> None:
    """A three-letter model name would fire on ordinary prose; it is excluded instead."""
    leaked = find_field_names_in_prose(
        _annotation("The gat output and the model results are withheld."),
        evidence=_evidence(models=("gat",)),
        references=[],
    )

    assert leaked == []


def test_evidence_references_are_exempt() -> None:
    """Reference paths belong in evidence_references; the prompt says so explicitly."""
    annotation = InvestigationAnnotation(
        prompt_version="test",
        model_name="test",
        evidence_references=["key_features[0]", "model_probabilities.catboost"],
        analytical_considerations=["All supplied values are withheld."],
        recommended_questions=["What authorized records exist?"],
    )

    leaked = find_field_names_in_prose(
        annotation,
        evidence=_evidence(features=("controller_tenure_context",)),
        references=[],
    )

    assert leaked == []


def test_each_leaked_name_is_reported_once() -> None:
    leaked = find_field_names_in_prose(
        _annotation(
            "The controller tenure is withheld.",
            "Again, the controller tenure remains unverified.",
        ),
        evidence=_evidence(features=("controller_tenure_context",)),
        references=[],
    )

    assert leaked == ["controller_tenure_context"]


# --- a separate, larger gap found while building Holdout v5 -----------------------


def test_worded_magnitude_claim_currently_evades_the_fact_gate() -> None:
    """DOCUMENTED GAP, not desired behaviour. Pinned so it cannot change unnoticed.

    The fact gate rejects numeric literals and account/transaction/alert tokens. It has
    no coverage for a magnitude claim expressed in words, so an annotation can tell a
    reviewer that a withheld score "sits in the top decile" and be accepted. The system
    instructions do forbid this, but a prompt is not enforcement, and Holdout v1 already
    showed the model making exactly this kind of names-only inference.

    This is a safety gap rather than a style one: it puts a fabricated magnitude in front
    of a human reviewer. Fixing it changes what the gate rejects, so it needs its own
    prompt/validator version and its own preregistered run, and must not be folded into a
    holdout measuring something else.
    """
    from aml_evidence_graph.investigation.llm import validate_annotation

    annotation = InvestigationAnnotation(
        prompt_version="test",
        model_name="test",
        evidence_references=["fusion_probability"],
        analytical_considerations=[
            "The withheld score sits in the top decile, which confirms the typology."
        ],
        recommended_questions=["Should the top-decile band drive filing?"],
    )
    evidence = _evidence()
    evidence = evidence.model_copy(update={"fusion_probability": 0.9})

    result = validate_annotation(annotation, evidence=evidence, references=[])

    assert result.valid is True, "gap closed - update this test and the docs"
    assert result.errors == []
