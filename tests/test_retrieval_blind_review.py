from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aml_evidence_graph.retrieval.blind_review import (
    BlindJudgment,
    BlindQuery,
    assert_no_prior_overlap,
    load_blind_queries,
    normalized_query,
    sha256_directory,
)


def test_blind_queries_reject_prior_case_and_normalized_query_overlap(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps([{"case_id": "old-1", "query": "Rapid cash out"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps prior"):
        assert_no_prior_overlap(
            [BlindQuery(case_id="new-1", query="  RAPID   cash out ")], [prior]
        )


def test_blind_query_loader_rejects_duplicates_after_normalization(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {"case_id": "a", "query": "unusual transfer pattern"},
                {"case_id": "b", "query": "Unusual   transfer pattern"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique after normalization"):
        load_blind_queries(path)


def test_blind_judgment_enforces_decision_label_consistency() -> None:
    answerable = BlindJudgment(
        case_id="blind-1",
        decision="answerable",
        relevant_typology_ids=["TYPOLOGY-CYCLE"],
        confidence="high",
        rationale="The described closed-loop flow matches the catalog entry.",
    )
    assert answerable.relevant_typology_ids == ["TYPOLOGY-CYCLE"]

    with pytest.raises(ValidationError, match="require at least one typology"):
        BlindJudgment(
            case_id="blind-2",
            decision="answerable",
            confidence="medium",
            rationale="Missing label should fail.",
        )


def test_directory_hash_includes_file_names_and_bytes(tmp_path: Path) -> None:
    (tmp_path / "b.yaml").write_text("value: 2\n", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("value: 1\n", encoding="utf-8")
    first = sha256_directory(tmp_path)
    (tmp_path / "a.yaml").write_text("value: 3\n", encoding="utf-8")
    second = sha256_directory(tmp_path)

    assert first != second
    assert normalized_query(" A   B ") == "a b"


def test_project_blind_v4_has_complete_disclosed_adjudication() -> None:
    root = Path(__file__).resolve().parents[1]
    queries = load_blind_queries(root / "golden" / "retrieval_queries_v4_project_blind.json")
    assert_no_prior_overlap(
        queries,
        [
            root / "golden" / "retrieval_queries_v1.json",
            root / "golden" / "retrieval_queries_v2.json",
            root / "golden" / "retrieval_queries_v3_additions.json",
        ],
    )
    payload = json.loads(
        (root / "golden" / "retrieval_adjudication_v4_project_blind.json").read_text(
            encoding="utf-8"
        )
    )
    judgments = [BlindJudgment.model_validate(item) for item in payload["judgments"]]
    assert len(queries) == len(judgments) == 50
    assert {item.case_id for item in queries} == {item.case_id for item in judgments}
    assert sum(item.decision == "answerable" for item in judgments) == 35
    assert sum(item.decision == "no_answer" for item in judgments) == 15
    assert payload["blind_to_model_outputs"] is True
    assert payload["independent_from_system_development"] is False

    protocol = yaml.safe_load(
        (root / "configs" / "retrieval" / "project_blind_review_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["frozen_system"]["dense_threshold"] == 0.35
    assert protocol["adjudication"]["independent_from_system_development"] is False
