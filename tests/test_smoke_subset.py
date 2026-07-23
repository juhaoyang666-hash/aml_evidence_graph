from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aml_evidence_graph.ingestion.smoke_subset import prepare_smoke_subset


def test_prepare_smoke_subset_copies_requested_dates(tmp_path: Path) -> None:
    source = tmp_path / "prepared"
    for event_date in ("2023-04-30", "2023-05-01", "2023-07-01"):
        target = source / f"event_date={event_date}" / "split=train"
        target.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame({"transaction_id": [event_date]})),
            target / "part.parquet",
        )

    summary = prepare_smoke_subset(
        source,
        tmp_path / "smoke",
        event_dates=("2023-04-30", "2023-05-01", "2023-07-02"),
    )

    assert summary.copied_dates == ["2023-04-30", "2023-05-01"]
    assert summary.missing_dates == ["2023-07-02"]
    assert (tmp_path / "smoke" / "event_date=2023-04-30").is_dir()
