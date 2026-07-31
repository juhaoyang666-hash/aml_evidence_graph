from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from aml_evidence_graph.tracking.candidate_comparison import (
    assert_same_population,
    load_score_frame,
    population_summary,
    validate_expected_population,
)


def _frame(scores: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transaction_id": ["b", "a"],
            "event_ts": [
                datetime(2023, 5, 2, tzinfo=UTC),
                datetime(2023, 5, 1, tzinfo=UTC),
            ],
            "is_laundering": [1, 0],
            "score": scores,
        }
    )


def test_population_fingerprint_ignores_score_and_row_order() -> None:
    incumbent = _frame([0.9, 0.1])
    candidate = _frame([0.8, 0.2]).reverse()
    digest = assert_same_population(incumbent, candidate, split="validation")
    assert digest == population_summary(incumbent)["sha256"]


def test_population_mismatch_is_rejected() -> None:
    candidate = _frame([0.8, 0.2]).with_columns(pl.lit(0).alias("is_laundering"))
    with pytest.raises(ValueError, match="not identical"):
        assert_same_population(_frame([0.9, 0.1]), candidate, split="validation")


def test_expected_population_checks_dates_and_counts() -> None:
    summary = population_summary(_frame([0.9, 0.1]))
    validate_expected_population(
        summary,
        {
            "sample_count": 2,
            "positive_count": 1,
            "start_date": "2023-05-01",
            "end_date": "2023-05-02",
        },
        split="validation",
    )


def test_score_loader_rejects_out_of_range_probability(tmp_path) -> None:
    path = tmp_path / "scores.parquet"
    _frame([1.1, 0.2]).write_parquet(path)
    with pytest.raises(ValueError, match="finite probabilities"):
        load_score_frame(path, "score")
