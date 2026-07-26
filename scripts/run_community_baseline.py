#!/usr/bin/env python3
"""Community / connected-component baseline on a high-score account subgraph.

Not a heterogeneous GNN. Builds an undirected projection from the investigation
window (seed high-score accounts + 1-hop neighbors), runs connected components
and Louvain, and reports label concentration vs a size-matched random partition
plus overlap with GAT investigation case_views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from aml_evidence_graph.data.contract import CANONICAL


def _load_window_edges(
    prepared_root: Path,
    *,
    window_start: str,
    window_end: str,
) -> pd.DataFrame:
    start = pd.Timestamp(window_start, tz="UTC")
    end = pd.Timestamp(window_end, tz="UTC")
    frames: list[pd.DataFrame] = []
    columns = [
        CANONICAL.transaction_id,
        CANONICAL.event_ts,
        CANONICAL.sender_account_id,
        CANONICAL.receiver_account_id,
        CANONICAL.is_laundering,
        CANONICAL.amount,
    ]
    for part in sorted(prepared_root.glob("event_date=*/split=*/*.parquet")):
        event_date = part.parts[-3].removeprefix("event_date=")
        day = pd.Timestamp(event_date, tz="UTC")
        if day < start.normalize() or day > end.normalize():
            continue
        frames.append(pd.read_parquet(part, columns=columns))
    if not frames:
        raise FileNotFoundError(f"No prepared partitions in [{window_start}, {window_end}]")
    out = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(out[CANONICAL.event_ts], utc=True)
    return out.loc[(ts >= start) & (ts <= end)].copy()


def _mean_community_purity(y: np.ndarray, sizes: list[int], order: np.ndarray) -> float:
    offset = 0
    purities: list[float] = []
    for size in sizes:
        idx = order[offset : offset + size]
        offset += size
        if len(idx) == 0:
            continue
        purities.append(float(y[idx].mean()))
    return float(np.mean(purities)) if purities else 0.0


def _random_mean_purity(
    labels: dict[str, int],
    community_sizes: list[int],
    *,
    rng: np.random.Generator,
    repeats: int = 100,
) -> float:
    """Unweighted mean community purity under random reassignment (size-matched)."""
    accounts = np.array(list(labels.keys()))
    y = np.array([labels[a] for a in accounts], dtype=int)
    if len(accounts) == 0:
        return 0.0
    sizes = [s for s in community_sizes if s > 0]
    return float(
        np.mean(
            [
                _mean_community_purity(y, sizes, rng.permutation(len(accounts)))
                for _ in range(repeats)
            ]
        )
    )


def _edge_positive_stats(
    graph: nx.Graph,
    communities: list[set[str]],
) -> dict[str, float]:
    membership: dict[str, int] = {}
    for idx, community in enumerate(communities):
        for account in community:
            membership[account] = idx
    intra_pos = intra_tot = inter_pos = inter_tot = 0
    for u, v, data in graph.edges(data=True):
        positive = int(data.get("positive_tx", 0) > 0)
        cu, cv = membership.get(u), membership.get(v)
        if cu is not None and cu == cv:
            intra_tot += 1
            intra_pos += positive
        else:
            inter_tot += 1
            inter_pos += positive
    intra_rate = intra_pos / intra_tot if intra_tot else 0.0
    inter_rate = inter_pos / inter_tot if inter_tot else 0.0
    return {
        "intra_community_edge_count": float(intra_tot),
        "inter_community_edge_count": float(inter_tot),
        "intra_edge_positive_rate": float(intra_rate),
        "inter_edge_positive_rate": float(inter_rate),
        "edge_positive_lift_intra_vs_inter": float(intra_rate / inter_rate)
        if inter_rate > 0
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/community_baseline"))
    parser.add_argument(
        "--account-risk",
        type=Path,
        default=Path("artifacts/test_investigation_views_gat/account_risk.parquet"),
    )
    parser.add_argument(
        "--case-views",
        type=Path,
        default=Path("artifacts/test_investigation_views_gat/case_views.json"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("artifacts/prepared_transactions"),
    )
    parser.add_argument("--window-start", default="2023-07-25T00:00:00Z")
    parser.add_argument("--window-end", default="2023-08-23T23:59:59Z")
    parser.add_argument("--top-accounts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accounts = pd.read_parquet(args.account_risk).sort_values(
        ["max_risk_score", "alert_transaction_count", "account_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    seed_accounts = set(accounts.head(args.top_accounts)["account_id"].astype(str))

    edges = _load_window_edges(
        args.prepared_root,
        window_start=args.window_start,
        window_end=args.window_end,
    )
    edges[CANONICAL.sender_account_id] = edges[CANONICAL.sender_account_id].astype(str)
    edges[CANONICAL.receiver_account_id] = edges[CANONICAL.receiver_account_id].astype(str)

    # 1-hop expansion: keep edges touching a seed account, then induce on touched nodes.
    touch = edges.loc[
        edges[CANONICAL.sender_account_id].isin(seed_accounts)
        | edges[CANONICAL.receiver_account_id].isin(seed_accounts)
    ]
    expanded_nodes = set(touch[CANONICAL.sender_account_id]).union(
        touch[CANONICAL.receiver_account_id]
    )
    induced = edges.loc[
        edges[CANONICAL.sender_account_id].isin(expanded_nodes)
        & edges[CANONICAL.receiver_account_id].isin(expanded_nodes)
    ].copy()

    positive_accounts: set[str] = set()
    pos = edges.loc[edges[CANONICAL.is_laundering].astype(int).eq(1)]
    positive_accounts.update(pos[CANONICAL.sender_account_id].astype(str))
    positive_accounts.update(pos[CANONICAL.receiver_account_id].astype(str))

    graph = nx.Graph()
    for row in induced.itertuples(index=False):
        u = getattr(row, CANONICAL.sender_account_id)
        v = getattr(row, CANONICAL.receiver_account_id)
        if u == v:
            continue
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1.0
            graph[u][v]["amount_sum"] += float(getattr(row, CANONICAL.amount))
            graph[u][v]["positive_tx"] += int(getattr(row, CANONICAL.is_laundering))
        else:
            graph.add_edge(
                u,
                v,
                weight=1.0,
                amount_sum=float(getattr(row, CANONICAL.amount)),
                positive_tx=int(getattr(row, CANONICAL.is_laundering)),
            )
    graph.add_nodes_from(expanded_nodes)

    components = [set(c) for c in nx.connected_components(graph) if len(c) >= 2]
    working = graph.subgraph([n for n, d in graph.degree() if d > 0]).copy()
    if working.number_of_edges() == 0:
        louvain_communities: list[set[str]] = []
    else:
        louvain_communities = [
            set(c)
            for c in nx.community.louvain_communities(working, weight="weight", seed=args.seed)
            if len(c) >= 2
        ]

    def community_rows(communities: list[set[str]], method: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, members in enumerate(sorted(communities, key=lambda c: (-len(c), min(c)))):
            member_list = sorted(members)
            n_pos = sum(1 for a in member_list if a in positive_accounts)
            rows.append(
                {
                    "method": method,
                    "community_id": f"{method}-{idx:04d}",
                    "size": len(member_list),
                    "positive_account_count": n_pos,
                    "positive_purity": n_pos / len(member_list),
                    "account_ids": member_list,
                }
            )
        return rows

    cc_rows = community_rows(components, "connected_components")
    lv_rows = community_rows(louvain_communities, "louvain")

    def summarize(rows: list[dict[str, Any]], communities: list[set[str]]) -> dict[str, Any]:
        if not rows:
            return {
                "community_count": 0,
                "accounts_in_communities": 0,
                "mean_purity": 0.0,
                "global_positive_rate_in_communities": 0.0,
                "random_baseline_mean_purity": 0.0,
                "lift_mean_purity_vs_random": None,
            }
        sizes = [r["size"] for r in rows]
        mean_purity = float(np.mean([r["positive_purity"] for r in rows]))
        labels = {a: int(a in positive_accounts) for community in communities for a in community}
        global_rate = float(np.mean(list(labels.values()))) if labels else 0.0
        rng = np.random.default_rng(args.seed)
        random_mean = _random_mean_purity(labels, sizes, rng=rng)
        lift = (mean_purity / random_mean) if random_mean > 0 else None
        # Concentration: share of positive accounts in the largest 10% of communities by size.
        n_top = max(1, int(np.ceil(0.1 * len(rows))))
        top = rows[:n_top]
        top_pos = sum(r["positive_account_count"] for r in top)
        all_pos = sum(r["positive_account_count"] for r in rows)
        return {
            "community_count": len(rows),
            "accounts_in_communities": int(sum(sizes)),
            "mean_purity": mean_purity,
            "global_positive_rate_in_communities": global_rate,
            "random_baseline_mean_purity": float(random_mean),
            "lift_mean_purity_vs_random": float(lift) if lift is not None else None,
            "top10pct_communities_positive_share": float(top_pos / all_pos) if all_pos else 0.0,
            "size_p50": float(np.median(sizes)),
            "size_max": int(max(sizes)),
            **_edge_positive_stats(graph, communities),
        }

    case_views = json.loads(args.case_views.read_text())
    case_account_sets = [set(map(str, c["account_ids"])) for c in case_views]
    case_accounts = set().union(*case_account_sets) if case_account_sets else set()

    def overlap_with_cases(communities: list[set[str]]) -> dict[str, Any]:
        if not communities or not case_account_sets:
            return {"communities_with_any_case_account": 0, "mean_best_jaccard_to_case": 0.0}
        hits = 0
        jaccards: list[float] = []
        covered = set()
        for community in communities:
            if community & case_accounts:
                hits += 1
                covered |= community & case_accounts
            best = 0.0
            for case_set in case_account_sets:
                union = community | case_set
                if union:
                    best = max(best, len(community & case_set) / len(union))
            jaccards.append(best)
        return {
            "communities_with_any_case_account": hits,
            "mean_best_jaccard_to_case": float(np.mean(jaccards)) if jaccards else 0.0,
            "case_account_coverage": float(len(covered) / len(case_accounts)) if case_accounts else 0.0,
        }

    summary = {
        "protocol": {
            "window_start": args.window_start,
            "window_end": args.window_end,
            "top_seed_accounts": args.top_accounts,
            "seed_account_count": len(seed_accounts),
            "expanded_node_count": len(expanded_nodes),
            "induced_edge_transactions": int(len(induced)),
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_edges": int(graph.number_of_edges()),
            "positive_accounts_in_window": len(positive_accounts),
            "investigation_case_count": len(case_views),
            "expansion": "seed_high_score_plus_1hop",
            "honest_boundary": (
                "Homogeneous undirected projection on high-score seeds + 1-hop neighbors; "
                "not a full heterogeneous GNN or production gang detector. "
                "Labels used only for offline purity / lift on synthetic SAML-D."
            ),
        },
        "connected_components": {
            **summarize(cc_rows, components),
            **overlap_with_cases(components),
        },
        "louvain": {
            **summarize(lv_rows, louvain_communities),
            **overlap_with_cases(louvain_communities),
        },
    }

    compact_cols = [
        "method",
        "community_id",
        "size",
        "positive_account_count",
        "positive_purity",
    ]
    pd.DataFrame([{k: r[k] for k in compact_cols} for r in cc_rows + lv_rows]).to_csv(
        args.output_dir / "community_metrics.csv", index=False
    )
    pd.DataFrame([{k: r[k] for k in compact_cols} for r in cc_rows + lv_rows]).to_parquet(
        args.output_dir / "community_metrics.parquet", index=False
    )
    (args.output_dir / "community_members.json").write_text(
        json.dumps(
            {
                "connected_components": [
                    {"community_id": r["community_id"], "account_ids": r["account_ids"]}
                    for r in cc_rows
                ],
                "louvain": [
                    {"community_id": r["community_id"], "account_ids": r["account_ids"]}
                    for r in lv_rows
                ],
            },
            indent=2,
        )
        + "\n"
    )
    (args.output_dir / "community_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
