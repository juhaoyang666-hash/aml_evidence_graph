"""Versioned local BM25 retrieval for AML Typology references."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class TypologyDocument:
    """A local curated Typology document."""

    typology_id: str
    version: str
    title: str
    body: str
    source: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower())


def load_typology_documents(root: Path) -> list[TypologyDocument]:
    """Load YAML Typology documents; source content remains version-controlled."""
    if not root.is_dir():
        raise FileNotFoundError(f"Typology directory does not exist: {root}")
    documents: list[TypologyDocument] = []
    for path in sorted(root.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Typology file must contain a mapping: {path}")
        required = {"typology_id", "version", "title", "body", "source"}
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(f"Typology file {path} is missing: {', '.join(missing)}")
        documents.append(
            TypologyDocument(
                typology_id=str(raw["typology_id"]),
                version=str(raw["version"]),
                title=str(raw["title"]),
                body=str(raw["body"]),
                source=str(raw["source"]),
            )
        )
    if not documents:
        raise ValueError(f"No YAML Typology documents found in {root}")
    return documents


class LocalBM25TypologyRetriever:
    """Small, deterministic retrieval layer with no service or vector database."""

    def __init__(self, documents: list[TypologyDocument]) -> None:
        if not documents:
            raise ValueError("At least one Typology document is required.")
        self.documents = documents
        self._bm25 = BM25Okapi(
            [_tokenize(f"{document.title} {document.body}") for document in documents]
        )

    def retrieve(self, query: str, *, limit: int = 3) -> list[TypologyDocument]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        scores = self._bm25.get_scores(_tokenize(query))
        ordered = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), self.documents[item[0]].typology_id),
        )
        return [self.documents[index] for index, _ in ordered[:limit]]
