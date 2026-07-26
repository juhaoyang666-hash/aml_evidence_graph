# Full-run results (SAML-D)

**Protocol**: public synthetic SAML-D · PIT features · fixed time-out split ·
test = 1,558,821 rows / 1,813 positives (positive rate ≈ 0.116%).  
**Primary metric**: PR-AUC. ROC-AUC is secondary at this rarity.  
**Primary table source**: `artifacts/table_baseline_rules` (includes rules KPI).

## Test ranking

| Model | Role | PR-AUC | ROC-AUC | 0.1% Precision | 0.1% Recall | 0.5% Precision | 0.5% Recall | run_id |
|---|---|---:|---:|---:|---:|---:|---:|---|
| rules (v2026.2) | baseline | 0.0015 | 0.5634 | 0.0019 | 0.0017 | 0.0026 | 0.0110 | `20260725T103959Z-27c7d65545` |
| logistic | table | 0.1966 | 0.9829 | 0.2809 | 0.2416 | 0.0916 | 0.3938 | same |
| **catboost** | **primary** | **0.8092** | 0.9963 | **0.8576** | **0.7375** | 0.2021 | 0.8687 | same |
| **graphsage** | **primary** | **0.8777** | 0.9996 | **0.8736** | **0.7512** | 0.2271 | 0.9763 | `20260725T090511Z-70e9a15e7b` |
| graph_stats_catboost | ablation / contrast | 0.9991 | ≈1.000 | 1.000 | 0.8599 | 0.2326 | 1.000 | `20260725T103959Z-27c7d65545` |
| fusion (catboost + graph_stats_catboost + graphsage) | fusion (3-way, contrast) | 0.9963 | ≈1.000 | 0.9987 | 0.8588 | 0.2326 | 1.000 | `20260725T111050Z-87c95c225c` |
| **fusion (catboost + graphsage)** | **fusion (2-way)** | **0.8973** | 0.9996 | **0.9076** | **0.7805** | 0.2281 | 0.9807 | `20260725T145745Z-a7d61d7c8d` |

Notes:

- Prefer **CatBoost**, **GraphSAGE**, and the **2-way fusion** for main-line reporting.
- The 3-way fusion (includes `graph_stats_catboost`) is retained as a contrast run only.
- Budget columns are precision/recall at top-k alert fractions of the test set.

## Honest fusion comparison (catboost + graphsage only)

Artifacts: `artifacts/fusion_cb_gs` (fit `run_id=20260725T145556Z-1fafa8b428`) →
`artifacts/fusion_test_cb_gs` (test `run_id=20260725T145745Z-a7d61d7c8d`).
Components: **`catboost,graphsage`** only (excludes `graph_stats_catboost`).

| Model | PR-AUC | 0.1% Precision | 0.1% Recall |
|---|---:|---:|---:|
| CatBoost alone | 0.8092 | 0.8576 | 0.7375 |
| GraphSAGE alone | 0.8777 | 0.8736 | 0.7512 |
| Fusion 2-way (cb + gs) | **0.8973** | **0.9076** | **0.7805** |
| Fusion 3-way (+ graph_stats) | 0.9963 | 0.9987 | 0.8588 |

**Does fusion help?** Yes for the honest 2-way stack: PR-AUC rises from GraphSAGE
0.8777 → **0.8973**, and the 0.1% alert budget improves on both precision and recall
versus either single model. The 3-way number is not comparable as a “fusion benefit”
claim because it includes the graph_stats contrast component.

## Alert reduction vs rules (same recall)

Source: `artifacts/table_baseline_rules` → `alert_reduction_vs_rules`.

| Model | @50% recall · reduction | model alerts | rules alerts | @70% | @85% |
|---|---:|---:|---:|---:|---:|
| **catboost** | **99.86%** | 908 | 649,036 | 99.88% | 99.56% |
| logistic | 98.03% | 12,780 | 649,036 | 97.37% | 95.99% |
| graph_stats_catboost | 99.86% | 907 | 649,036 | 99.89% | 99.89% |

Rules: `configs/rules/default.yaml` v2026.2 (train-period quantiles only).  
Full PIT rule hits: 836,487 across 9,504,852 rows.

## Investigation views

Frozen fusion scores → `artifacts/test_investigation_views`  
(`run_id=20260725T111133Z-e73ad70672`, as_of `2023-08-23`).

| Artifact | Count |
|---|---:|
| accounts | 307,103 |
| funds paths | 266 |
| investigation cases | 228 |

## Suggested resume wording

> On public synthetic SAML-D (~9.5M txs, ~0.1% positives), PIT features + CatBoost
> reach test PR-AUC 0.809; at a 0.1% alert budget, Precision 0.858 / Recall 0.737.
> History-only GraphSAGE reaches test PR-AUC 0.878. CatBoost + GraphSAGE OOF fusion
> reaches test PR-AUC 0.897 (0.1% budget P 0.908 / R 0.780). Versus train-period
> quantile rules, CatBoost cuts alert volume by ~99.9% at matched 50% recall.

Always attach: **dataset + run_id + time-split protocol**.

## Golden 30 (project-adjudicated investigation regression)

Source: `golden/cases_v1.json` + adjudication record `golden/adjudication_v1.json`
(18 typology + 6 low-evidence + 6 adversarial).  
**Adjudicator**: `agent-authorized-by-user` (user-authorized agent pass on 2026-07-26).  
This **is** the project’s adjudicated Golden v1 for regression — **not** an independent
third-party human panel, and not production compliance labels.

### Template path (no LLM)

`artifacts/golden_summary.json` · `run_id=20260726T020456Z-e698864ca4`

| Metric | Value |
|---|---:|
| Cases | 30 |
| Schema / fact-snapshot match | 1.0 / 1.0 |
| Typology match rate | 1.0 |
| Correct rejection rate | 1.0 |
| Hallucination intercept rate (injected probes) | **1.0** (4/4) |
| No-evidence refusal rate | **1.0** |
| Latency p50 / p95 (ms) | ~7.3 / ~7.8 |

### LLM path (ECNU)

`artifacts/golden_summary_llm.json` · `run_id=20260726T020614Z-61765ffec9` · model `ecnu-max` ·
prompt `ecnu-risk-evidence-v1`

| Metric | Value |
|---|---:|
| Cases | 30 |
| Schema / fact-snapshot match | 1.0 / 1.0 |
| Hallucination intercept rate (injected probes) | **1.0** (4/4) |
| No-evidence refusal rate | **1.0** |
| Correct rejection rate | 0.90 |
| LLM annotation rate | 0.733 |
| Latency p50 / p95 (ms) | **2684 / 3721** |
| Reported tokens (prompt / completion) | 7618 / 6161 |

Notes: three cases (`agent-typo-10-in`, `agent-typo-16-cycle`, `agent-low-evidence-03`)
received LLM annotations that failed fact validation (`rejected_facts`) — the safety
gate worked; this lowers `correct_rejection_rate` because those cases expect
`draft_requires_human_review`. Injected adversarial probes still intercept at 100%.
Prompt-injection cases (`agent-adv-05/06`) completed as human-review drafts.

## Edge GNN architecture comparison (complete)

Same PIT features + time split + hyperparams as GraphSAGE (`configs/models.yaml` protocol;
architecture overlays in `configs/models.{gat,rgcn,pna}.yaml`).  
Ran via `scripts/run_arch_comparison.sh` (sequential GAT → RGCN → PNA; outputs
`artifacts/{gat,rgcn,pna}`; done flag `logs/arch_comparison_done.flag`).

| Model | Test PR-AUC | ROC-AUC | 0.1% Precision | 0.1% Recall | run_id | artifact |
|---|---:|---:|---:|---:|---|---|
| **GAT** | **0.9483** | 0.9996 | **0.9743** | **0.8378** | `20260726T023031Z-5ccb81baec` | `artifacts/gat` |
| RGCN | 0.9031 | 0.9997 | 0.9051 | 0.7783 | `20260726T062636Z-ef6be55956` | `artifacts/rgcn` |
| GraphSAGE | 0.8777 | 0.9996 | 0.8736 | 0.7512 | `20260725T090511Z-70e9a15e7b` | `artifacts/graphsage` |
| PNA | 0.7049 | 0.9992 | 0.6972 | 0.5996 | `20260726T064804Z-28028363ce` | `artifacts/pna` |

Ranking by test PR-AUC: **GAT > RGCN > GraphSAGE > PNA**.

**Primary resume metrics remain CatBoost / GraphSAGE / 2-way fusion**; this table is an
additional same-protocol architecture comparison, not a change to the main-line stack.
