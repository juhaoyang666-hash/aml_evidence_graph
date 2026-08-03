import polars as pl
import pytest

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.training.table_baseline import (
    deterministic_hard_negative_downsample,
    deterministic_negative_downsample,
)


def _training_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            CANONICAL.transaction_id: [f"t{index}" for index in range(8)],
            CANONICAL.event_ts: [f"2022-10-{7 + index:02d}T00:00:00Z" for index in range(8)],
            CANONICAL.source_row_number: range(8),
            CANONICAL.is_laundering: [1, 1, 0, 0, 0, 0, 0, 0],
        }
    )


def test_stable_negative_sampling_keeps_all_positive_rows_deterministically() -> None:
    sampled = deterministic_negative_downsample(
        _training_frame(),
        maximum_negative_rows=2,
    )

    assert sampled.height == 4
    assert sampled[CANONICAL.is_laundering].sum() == 2
    repeated = deterministic_negative_downsample(
        _training_frame(),
        maximum_negative_rows=2,
    )
    assert (
        sampled[CANONICAL.transaction_id].to_list()
        == repeated[CANONICAL.transaction_id].to_list()
    )


def test_hard_negative_sampling_keeps_all_positives_and_top_temporal_oof_negatives() -> None:
    training = _training_frame()
    oof = pl.DataFrame(
        {
            CANONICAL.transaction_id: ["t2", "t3", "t4", "t5"],
            "catboost": [0.2, 0.9, 0.8, 0.1],
        }
    )

    sampled = deterministic_hard_negative_downsample(
        training,
        oof,
        maximum_negative_rows=2,
    )

    positive_ids = set(
        sampled.filter(pl.col(CANONICAL.is_laundering) == 1)[CANONICAL.transaction_id].to_list()
    )
    negative_ids = set(
        sampled.filter(pl.col(CANONICAL.is_laundering) == 0)[CANONICAL.transaction_id].to_list()
    )
    assert positive_ids == {"t0", "t1"}
    assert negative_ids == {"t3", "t4"}


def test_hard_negative_sampling_rejects_scores_outside_training_period() -> None:
    oof = pl.DataFrame(
        {
            CANONICAL.transaction_id: ["not-a-training-row"],
            "catboost": [0.9],
        }
    )

    with pytest.raises(ValueError, match="outside the training period"):
        deterministic_hard_negative_downsample(
            _training_frame(),
            oof,
            maximum_negative_rows=2,
        )
