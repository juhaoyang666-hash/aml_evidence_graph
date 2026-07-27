from datetime import UTC, datetime

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evidence.builder import build_risk_evidence_package
from aml_evidence_graph.graph.explain import GraphEdgeEvidence


def test_evidence_builder_uses_only_selected_transaction_facts() -> None:
    transaction = {
        CANONICAL.transaction_id: "txn-row-0001",
        CANONICAL.event_ts: "2023-07-01T12:00:00Z",
        "sender_outgoing_count_7d": 4.0,
        "non_evidence_value": "must-not-appear",
    }

    package = build_risk_evidence_package(
        transaction,
        alert_id="alert-001",
        model_probabilities={"catboost": 0.6},
        source_versions={"model_run_id": "run-1"},
        selected_feature_names=["sender_outgoing_count_7d"],
        graph_snapshot_as_of=datetime(2023, 7, 1, tzinfo=UTC),
    )

    assert package.transaction_id == "txn-row-0001"
    assert [feature.name for feature in package.key_features] == [
        "sender_outgoing_count_7d"
    ]
    assert "non_evidence_value" not in package.model_dump_json()


def test_evidence_builder_keeps_bounded_graph_node_and_path_evidence() -> None:
    transaction = {
        CANONICAL.transaction_id: "txn-row-0002",
        CANONICAL.event_ts: "2023-07-01T12:00:00Z",
    }
    graph_evidence = GraphEdgeEvidence(
        event_date="2023-07-01",
        source_node=3,
        destination_node=4,
        historical_source_out_degree=2,
        historical_destination_in_degree=1,
        prior_directed_edge_count=1,
        prior_reverse_edge_count=0,
        two_hop_intermediary_nodes=[5],
    )

    package = build_risk_evidence_package(
        transaction,
        alert_id="alert-002",
        model_probabilities={"graphsage": 0.6},
        source_versions={"graph_model": "run-1"},
        selected_feature_names=[],
        graph_edge_evidence=graph_evidence,
        graph_snapshot_as_of=datetime(2023, 7, 1, tzinfo=UTC),
    )

    assert package.graph_evidence is not None
    assert package.graph_evidence.source_node_index == 3
    assert package.graph_evidence.destination_node_index == 4
    assert package.graph_evidence.two_hop_intermediary_node_indices == [5]
