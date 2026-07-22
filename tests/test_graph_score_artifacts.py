from pathlib import Path

import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.training.run_graphsage import _write_graph_scores


def test_graph_score_artifact_uses_snapshot_chronological_order(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            CANONICAL.transaction_id: ["later", "earlier"],
            CANONICAL.event_ts: ["2023-07-02T00:00:00Z", "2023-07-01T00:00:00Z"],
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

    assert output[CANONICAL.transaction_id].tolist() == ["earlier", "later"]
    assert output["graphsage"].tolist() == [0.9, 0.1]
    persisted = pd.read_parquet(tmp_path / "scores" / "graphsage_test_scores.parquet")
    assert persisted.equals(output)
