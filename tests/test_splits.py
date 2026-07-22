import pandas as pd
import pytest

from aml_evidence_graph.data.splits import TimeSplit, assign_time_split


def test_assign_time_split_is_chronological_and_exhaustive() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            ["2022-10-07T00:00:00Z", "2023-05-01T00:00:00Z", "2023-08-23T23:59:59Z"],
            utc=True,
        )
    )

    assert assign_time_split(timestamps).tolist() == [
        TimeSplit.TRAIN.value,
        TimeSplit.VALIDATION.value,
        TimeSplit.TEST.value,
    ]


def test_assign_time_split_rejects_data_outside_protocol() -> None:
    timestamps = pd.Series(pd.to_datetime(["2023-08-24T00:00:00Z"], utc=True))

    with pytest.raises(ValueError, match="outside"):
        assign_time_split(timestamps)

