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
| graphsage | graph (superseded) | 0.8777 | 0.9996 | 0.8736 | 0.7512 | 0.2271 | 0.9763 | `20260725T090511Z-70e9a15e7b` |
| **GAT** | **primary graph** | **0.9483** | 0.9996 | **0.9743** | **0.8378** | 0.2285 | 0.9823 | `20260726T023031Z-5ccb81baec` |
| graph_stats_catboost | ablation / contrast | 0.9991 | ≈1.000 | 1.000 | 0.8599 | 0.2326 | 1.000 | `20260725T103959Z-27c7d65545` |
| fusion (catboost + graph_stats_catboost + graphsage) | fusion (3-way, contrast) | 0.9963 | ≈1.000 | 0.9987 | 0.8588 | 0.2326 | 1.000 | `20260725T111050Z-87c95c225c` |
| fusion (catboost + graphsage) | fusion (2-way, superseded) | 0.8973 | 0.9996 | 0.9076 | 0.7805 | 0.2281 | 0.9807 | `20260725T145745Z-a7d61d7c8d` |
| **fusion (catboost + GAT)** | **fusion (2-way)** | **0.9175** | 0.9994 | **0.9359** | **0.8047** | 0.2298 | 0.9879 | `20260726T090744Z-4a97fdf32f` |

Notes:

- Main-line reporting: **CatBoost**, **GAT**, and the **2-way `catboost + GAT` fusion**.
  GraphSAGE and the `catboost + graphsage` fusion are retained as the earlier main line
  for comparability, not as the recommended stack.
- The 3-way fusion (includes `graph_stats_catboost`) is retained as a contrast run only.
  `graph_stats` is **excluded** from both 2-way fusions.
- Budget columns are precision/recall at top-k alert fractions of the test set.

## Main-line model choice: GAT vs GraphSAGE (2-way fusion)

Both fusions use identical protocol and identical frozen CatBoost score files
(`artifacts/table_baseline/scores`), components **`catboost,graphsage`** only, no
`graph_stats`. Only the graph component differs. The OOF score column is literally
named `graphsage` in both runs because the OOF writer labels the graph column by CLI
`--model`, not by architecture; in the GAT run that column holds **GAT** scores
(`artifacts/graph_oof_gat`, `run_id=20260726T090533Z-9e665d8d94`,
`model_config=configs/models.gat.yaml`, 4,524,912 OOF rows, 3 expanding folds,
minimum 2 training months).

| Fusion | Test PR-AUC | ROC-AUC | 0.1% P | 0.1% R | 0.5% P | 0.5% R | fit run_id | test run_id |
|---|---:|---:|---:|---:|---:|---:|---|---|
| catboost + graphsage | 0.8973 | 0.9996 | 0.9076 | 0.7805 | 0.2281 | 0.9807 | `20260725T145556Z-1fafa8b428` | `20260725T145745Z-a7d61d7c8d` |
| **catboost + GAT** | **0.9175** | 0.9994 | **0.9359** | **0.8047** | **0.2298** | **0.9879** | `20260726T090619Z-0ff8a9c7ff` | `20260726T090744Z-4a97fdf32f` |

Artifacts: `artifacts/graph_oof_gat` → `artifacts/fusion_cb_gat` →
`artifacts/fusion_test_cb_gat`.

**Verdict.** Swapping GraphSAGE for GAT improves the 2-way fusion on every reported
figure except ROC-AUC (0.9996 → 0.9994, immaterial at this rarity): PR-AUC
0.8973 → **0.9175** (+0.0202), 0.1% budget precision 0.9076 → **0.9359**, recall
0.7805 → **0.8047**. **`catboost + GAT` replaces `catboost + graphsage` as the
recommended main-line stack.**

**Caveat, stated plainly:** the fusion does **not** beat GAT alone. On the same test
split GAT alone scores PR-AUC **0.9483** with 0.1% budget P 0.9743 / R 0.8378, which is
better than the 0.9175 fusion. So fusion helps relative to CatBoost (0.8092) and
relative to the old GraphSAGE fusion, but for this dataset the single GAT edge model is
the strongest scorer. Two defensible reporting choices follow, and this project keeps
both visible rather than picking the flattering one:

- If the goal is peak ranking quality: report **GAT alone** (0.9483).
- If the goal is the fusion architecture the system is built around (tabular + graph,
  leakage-safe OOF stacking, calibrated threshold policy): report
  **`catboost + GAT`** (0.9175) and disclose that GAT alone ranks higher.

No single-model/fusion comparison in the original training manifests used confidence
intervals (`bootstrap_iterations=0`). A post-hoc stratified bootstrap (200 iterations)
is reported below so gaps can be discussed with intervals.

## Bootstrap CI (test set, 200 stratified iterations)

Source: `artifacts/diagnostics/bootstrap_ci.json` · seed `20260722` · 95% intervals.  
Scores: CatBoost from `table_baseline_rules`, GAT from `artifacts/gat`, fusion from
`fusion_test_cb_gat` (`fusion_calibrated_probability`), aligned on `transaction_id`.

| Model | PR-AUC point | 95% CI | ROC-AUC point | 95% CI |
|---|---:|---|---:|---|
| CatBoost | 0.8092 | **[0.7914, 0.8260]** | 0.9963 | [0.9958, 0.9969] |
| **GAT** | 0.9483 | **[0.9396, 0.9549]** | 0.9996 | [0.9994, 0.9998] |
| **catboost + GAT** | 0.9175 | **[0.9073, 0.9276]** | 0.9994 | [0.9987, 0.9998] |

**Reading the gaps**

- Fusion vs CatBoost: fusion CI sits entirely above CatBoost CI → improvement is stable.
- Fusion vs GAT alone: fusion CI sits entirely **below** GAT CI → “fusion &lt; GAT alone”
  is also stable on this split — keep disclosing both numbers.
- Point CatBoost+GraphSAGE fusion (0.8973) lies below the CatBoost+GAT fusion lower bound
  (0.9073) → replacing GraphSAGE with GAT in the 2-way stack is a clear gain, even without
  re-bootstrapping the older fusion file.

## CatBoost vs GAT gap diagnosis

See [CATBOOST_GAP_DIAGNOSIS.md](CATBOOST_GAP_DIAGNOSIS.md). At 0.1% budget, GAT recovers
**355** true positives that CatBoost misses (overlap 1,164); CatBoost-only TP = 173.
CatBoost already uses `relationship_*` window features in its top importances — the residual
gap is consistent with neighborhood message-passing beyond pairwise aggregates.

Hard-negative retrain (`artifacts/table_baseline_hardneg`) **regressed** CatBoost test
PR-AUC from 0.8092 → **0.4548**; keep hash-sampled CatBoost. Details:
[CATBOOST_GAP_DIAGNOSIS.md](CATBOOST_GAP_DIAGNOSIS.md).

Source: `artifacts/table_baseline_rules` → `alert_reduction_vs_rules`.

| Model | @50% recall · reduction | model alerts | rules alerts | @70% | @85% |
|---|---:|---:|---:|---:|---:|
| **catboost** | **99.86%** | 908 | 649,036 | 99.88% | 99.56% |
| logistic | 98.03% | 12,780 | 649,036 | 97.37% | 95.99% |
| graph_stats_catboost | 99.86% | 907 | 649,036 | 99.89% | 99.89% |

Rules: `configs/rules/default.yaml` v2026.2 (train-period quantiles only).  
Full PIT rule hits: 836,487 across 9,504,852 rows.

## Investigation views

Built from frozen fusion test scores (`fusion_calibrated_probability`, as_of
`2023-08-23T23:59:59Z`, 30-day window, risk threshold 0.5).

| Artifact | Source fusion | run_id | accounts | funds paths | cases |
|---|---|---|---:|---:|---:|
| `artifacts/test_investigation_views` | 3-way fusion (`artifacts/fusion_test`) | `20260725T111133Z-e73ad70672` | 307,103 | 266 | 228 |
| **`artifacts/test_investigation_views_gat`** | **catboost + GAT** (`artifacts/fusion_test_cb_gat`) | `20260726T090826Z-b9ef382e9f` | 307,103 | 135 | 212 |

The GAT-based views carry fewer funds paths and cases because the calibrated GAT fusion
puts fewer accounts above the 0.5 risk threshold — a sharper, more concentrated alert
set, not a data change. Account universe is identical (307,103).


## Suggested resume wording

> On public synthetic SAML-D (~9.5M txs, ~0.1% positives), PIT features + CatBoost
> reach test PR-AUC 0.809; at a 0.1% alert budget, Precision 0.858 / Recall 0.737.
> A history-only GAT edge classifier reaches test PR-AUC 0.948 (0.1% budget
> P 0.974 / R 0.838). Leakage-safe CatBoost + GAT OOF fusion reaches test PR-AUC 0.917
> (0.1% budget P 0.936 / R 0.805), improving on the earlier CatBoost + GraphSAGE
> fusion (0.897). Versus train-period quantile rules, CatBoost cuts alert volume by
> ~99.9% at matched 50% recall.

Always attach: **dataset + run_id + time-split protocol**. If you quote the fusion
number, also quote GAT alone (0.948) — the fusion is not the top scorer here.


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

GAT was subsequently promoted through the full frozen-score chain (OOF → fusion →
test → investigation views); see “Main-line model choice” above.
