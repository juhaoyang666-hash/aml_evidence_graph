# Community / connected-component baseline

**Status**: offline graph-association baseline on the GAT investigation window.  
**Artifacts**: `artifacts/community_baseline/`  
**Command**:

```bash
export PYTHONPATH=src
python scripts/run_community_baseline.py
```

## What this is / is not

| Is | Is not |
|---|---|
| Homogeneous undirected projection | Full heterogeneous GNN |
| Louvain + connected components | Production gang detector |
| High-score seed accounts + 1-hop | Full-universe community mining |
| Offline purity / case overlap on synthetic labels | Third-party typology adjudication |

## Protocol

- Seed: top **2,000** accounts by `max_risk_score` from `artifacts/test_investigation_views_gat/account_risk.parquet`.
- Window: 2023-07-25 → 2023-08-23 (same as investigation views).
- Expansion: keep transactions touching a seed account, then induce on the expanded node set (**12,764** nodes, **12,089** undirected edges, **65,344** txs).
- Algorithms: NetworkX connected components + `louvain_communities` (weight = multi-edge count).
- Contrast set: 212 GAT `case_views`.

## Results

| Method | Communities (size≥2) | Mean purity | Random mean purity | Lift (mean) | Top-10% communities’ share of positives | Case-account coverage | Mean best Jaccard→case |
|---|---:|---:|---:|---:|---:|---:|---:|
| Connected components | 721 | 0.053 | 0.075 | 0.70 | **52.5%** | **1.00** | 0.055 |
| Louvain | 724 | 0.053 | 0.075 | 0.70 | **52.7%** | **1.00** | 0.055 |

Global positive-account rate inside the subgraph ≈ **7.5%**.  
CC and Louvain are nearly identical here: the projection is already component-dominated (Louvain cuts only **3** inter-community edges).

### How to read the “lift &lt; 1”

Unweighted **mean** community purity is pulled down by many small all-negative communities; a size-matched random partition spreads positives more evenly, so mean purity rises toward the global rate. That is an honest property of this seed subgraph, not a claim that community structure is useless.

More useful signals for interviews / ops:

1. **Concentration** — largest ~10% of communities hold ~**53%** of positive accounts in the subgraph.
2. **Case coverage** — every account appearing in GAT investigation cases is covered by at least one community (`case_account_coverage = 1.0`).
3. **Intra-edge positive rate** ≈ **6.7%** of undirected edges touch ≥1 laundering transaction in-window (labels offline only).

## Artifacts

- `community_summary.json` — metrics + protocol
- `community_metrics.csv` / `.parquet` — per-community size / purity
- `community_members.json` — account membership lists

## Honest boundary

Synthetic SAML-D · labels used only for offline reporting · **not** a substitute for the supervised GAT edge model (test PR-AUC 0.948) · explicitly **not** a heterogeneous multi-relational GNN. Neural relation ablation: multi-rel RGCN R=4 test PR-AUC **0.887** (below single-rel RGCN 0.903) — see [RELATION_ABLATION.md](RELATION_ABLATION.md).
