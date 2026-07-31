"""Contracts and integrity checks for a model-blind retrieval review set."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BlindQuery(BaseModel):
    """A query supplied to an adjudicator without model predictions or labels."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=3)
    tags: list[str] = Field(default_factory=list)


class BlindJudgment(BaseModel):
    """A completed independent judgment."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    decision: str
    relevant_typology_ids: list[str] = Field(default_factory=list)
    confidence: str
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> BlindJudgment:
        if self.decision not in {"answerable", "no_answer", "exclude"}:
            raise ValueError("decision must be answerable, no_answer, or exclude")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")
        if self.decision == "answerable" and not self.relevant_typology_ids:
            raise ValueError("answerable judgments require at least one typology id")
        if self.decision != "answerable" and self.relevant_typology_ids:
            raise ValueError("no_answer/exclude judgments cannot contain typology ids")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path, pattern: str = "*.yaml") -> str:
    """Hash names and bytes in stable order so corpus changes invalidate a review."""
    digest = hashlib.sha256()
    files = sorted(path.glob(pattern), key=lambda item: item.name)
    if not files:
        raise ValueError(f"No {pattern} files found in {path}")
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def normalized_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_blind_queries(path: Path) -> list[BlindQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Blind query input must be a JSON list")
    queries = [BlindQuery.model_validate(item) for item in payload]
    ids = [item.case_id for item in queries]
    normalized = [normalized_query(item.query) for item in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Blind query case_id values must be unique")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Blind query texts must be unique after normalization")
    return queries


def assert_no_prior_overlap(queries: list[BlindQuery], prior_paths: list[Path]) -> None:
    prior_ids: set[str] = set()
    prior_queries: set[str] = set()
    for path in prior_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Prior case file must be a JSON list: {path}")
        for item in payload:
            prior_ids.add(str(item["case_id"]))
            prior_queries.add(normalized_query(str(item["query"])))
    duplicate_ids = sorted({item.case_id for item in queries} & prior_ids)
    duplicate_queries = sorted(
        {normalized_query(item.query) for item in queries} & prior_queries
    )
    if duplicate_ids or duplicate_queries:
        raise ValueError(
            "Blind set overlaps prior evaluation data: "
            f"case_ids={duplicate_ids}, normalized_queries={duplicate_queries}"
        )

