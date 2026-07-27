from pathlib import Path

import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.training.run_graphsage import _write_graph_scores


def test_graph_score_artifact_uses_snapshot_chronological_order(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            CANONICAL.transaction_id: ["later", "earlier"],
            CANONICAL.event_ts: pl.Series(
                ["2023-07-02T00:00:00Z", "2023-07-01T00:00:00Z"]
            ).str.to_datetime(time_zone="UTC"),
            CANONICAL.source_row_number: [2, 1],
            CANONICAL.is_laundering: [0, 1],
        }
    )

    output = _write_graph_scores(
        tmp_path,
        split_name="test",
        frame=frame,
        scores=[0.9, 0.1],
    )

    assert output[CANONICAL.transaction_id].to_list() == ["earlier", "later"]
    assert output["graphsage"].to_list() == [0.9, 0.1]
    persisted = pl.read_parquet(tmp_path / "scores" / "graphsage_test_scores.parquet")
    assert persisted.equals(output)
