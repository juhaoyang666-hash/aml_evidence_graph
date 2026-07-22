import pandas as pd
import pytest

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.training.table_baseline import (
    deterministic_hard_negative_downsample,
    deterministic_negative_downsample,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            CANONICAL.transaction_id: [f"t{index}" for index in range(8)],
            CANONICAL.event_ts: pd.date_range("2022-10-07", periods=8, tz="UTC"),
            CANONICAL.source_row_number: range(8),
            CANONICAL.is_laundering: [1, 1, 0, 0, 0, 0, 0, 0],
        }
    )


def test_stable_negative_sampling_keeps_all_positive_rows_deterministically() -> None:
    sampled = deterministic_negative_downsample(
        _training_frame(),
        maximum_negative_rows=2,
    )

    assert len(sampled) == 4
    assert sampled[CANONICAL.is_laundering].sum() == 2
    repeated = deterministic_negative_downsample(
        _training_frame(),
        maximum_negative_rows=2,
    )
    assert sampled[CANONICAL.transaction_id].tolist() == repeated[CANONICAL.transaction_id].tolist()


def test_hard_negative_sampling_keeps_all_positives_and_top_temporal_oof_negatives() -> None:
    training = _training_frame()
    oof = pd.DataFrame(
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

    assert set(sampled.loc[sampled[CANONICAL.is_laundering].eq(1), CANONICAL.transaction_id]) == {
        "t0",
        "t1",
    }
    assert set(sampled.loc[sampled[CANONICAL.is_laundering].eq(0), CANONICAL.transaction_id]) == {
        "t3",
        "t4",
    }


def test_hard_negative_sampling_rejects_scores_outside_training_period() -> None:
    oof = pd.DataFrame(
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
