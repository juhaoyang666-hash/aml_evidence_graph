import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.features.graph_stats import CausalGraphStatisticsBuilder


def test_graph_statistics_exclude_same_timestamp_edges() -> None:
    frame = pl.DataFrame(
        {
            CANONICAL.transaction_id: ["one", "two", "three"],
            CANONICAL.event_ts: pl.Series(
                [
                    "2023-01-01T10:00:00Z",
                    "2023-01-01T10:00:00Z",
                    "2023-01-01T10:01:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            CANONICAL.sender_account_id: ["a", "b", "b"],
            CANONICAL.receiver_account_id: ["b", "a", "a"],
            CANONICAL.source_row_number: [1, 2, 3],
        }
    )

    features = CausalGraphStatisticsBuilder().transform_partition(frame)

    assert features.row(0, named=True)["graph_sender_historical_out_degree"] == 0.0
    assert features.row(1, named=True)["graph_sender_historical_out_degree"] == 0.0
    assert features.row(2, named=True)["graph_directed_edge_prior_count"] == 1.0
    assert features.row(2, named=True)["graph_prior_reciprocal_relationship"] == 1.0
