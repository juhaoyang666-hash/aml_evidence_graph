"""Deterministic structural evidence for graph alerts; not causal attribution."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from aml_evidence_graph.graph.snapshots import TemporalGraphSnapshot


@dataclass(frozen=True)
class GraphEdgeEvidence:
    """Historical topology surrounding one scoring edge."""

    event_date: str
    source_node: int
    destination_node: int
    historical_source_out_degree: int
    historical_destination_in_degree: int
    prior_directed_edge_count: int
    prior_reverse_edge_count: int
    two_hop_intermediary_nodes: list[int]
    interpretation_limit: str = (
        "These are historical structural facts, not causal explanations or attention weights."
    )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalGraphEvidenceIndex:
    """Reusable historical adjacency index for bounded per-alert explanations."""

    outgoing_neighbors: dict[int, frozenset[int]]
    incoming_predecessors: dict[int, frozenset[int]]
    directed_edge_counts: dict[tuple[int, int], int]


def build_historical_evidence_index(
    snapshot: TemporalGraphSnapshot,
) -> HistoricalGraphEvidenceIndex:
    """Index one snapshot once instead of scanning every history edge per alert."""
    outgoing: defaultdict[int, set[int]] = defaultdict(set)
    incoming: defaultdict[int, set[int]] = defaultdict(set)
    counts: Counter[tuple[int, int]] = Counter()
    for raw_source, raw_destination in snapshot.history_edge_index.T:
        source = int(raw_source)
        destination = int(raw_destination)
        outgoing[source].add(destination)
        incoming[destination].add(source)
        counts[(source, destination)] += 1
    return HistoricalGraphEvidenceIndex(
        outgoing_neighbors={node: frozenset(values) for node, values in outgoing.items()},
        incoming_predecessors={node: frozenset(values) for node, values in incoming.items()},
        directed_edge_counts=dict(counts),
    )


def extract_graph_edge_evidence(
    snapshot: TemporalGraphSnapshot,
    *,
    scoring_edge_position: int,
    max_intermediaries: int = 10,
    history_index: HistoricalGraphEvidenceIndex | None = None,
) -> GraphEdgeEvidence:
    """Extract bounded historical path evidence for a transaction edge."""
    if not 0 <= scoring_edge_position < snapshot.scoring_edge_index.shape[1]:
        raise IndexError("scoring_edge_position is outside the snapshot scoring edge array.")
    if max_intermediaries < 1:
        raise ValueError("max_intermediaries must be positive.")
    source = int(snapshot.scoring_edge_index[0, scoring_edge_position])
    destination = int(snapshot.scoring_edge_index[1, scoring_edge_position])
    if history_index is None:
        history = snapshot.history_edge_index
        outgoing = history[0] == source
        incoming = history[1] == destination
        source_neighbors = set(history[1, outgoing].astype(int).tolist())
        destination_predecessors = set(history[0, incoming].astype(int).tolist())
        prior_directed_edge_count = int(
            ((history[0] == source) & (history[1] == destination)).sum()
        )
        prior_reverse_edge_count = int(
            ((history[0] == destination) & (history[1] == source)).sum()
        )
    else:
        source_neighbors = set(history_index.outgoing_neighbors.get(source, ()))
        destination_predecessors = set(
            history_index.incoming_predecessors.get(destination, ())
        )
        prior_directed_edge_count = history_index.directed_edge_counts.get(
            (source, destination), 0
        )
        prior_reverse_edge_count = history_index.directed_edge_counts.get(
            (destination, source), 0
        )
    historical_source_out_degree = len(source_neighbors)
    historical_destination_in_degree = len(destination_predecessors)
    intermediaries = sorted(source_neighbors.intersection(destination_predecessors))
    return GraphEdgeEvidence(
        event_date=snapshot.event_date,
        source_node=source,
        destination_node=destination,
        historical_source_out_degree=historical_source_out_degree,
        historical_destination_in_degree=historical_destination_in_degree,
        prior_directed_edge_count=prior_directed_edge_count,
        prior_reverse_edge_count=prior_reverse_edge_count,
        two_hop_intermediary_nodes=intermediaries[:max_intermediaries],
    )
