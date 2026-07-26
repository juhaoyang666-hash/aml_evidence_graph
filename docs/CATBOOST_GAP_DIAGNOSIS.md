# CatBoost vs GAT ranking gap（诊断）

Source: `artifacts/diagnostics/catboost_gat_gap.json`  
Test set: 1,558,821 rows / 1,813 positives · aligned CatBoost
(`table_baseline_rules`) vs GAT (`artifacts/gat`, score column `graphsage`).

## Headline

| Model | PR-AUC | 0.1% budget Precision | 0.1% Recall | TP @0.1% |
|---|---:|---:|---:|---:|
| CatBoost | 0.8092 | 0.8576 | 0.7375 | 1,337 |
| GAT | 0.9483 | 0.9743 | 0.8378 | 1,519 |

At a 0.1% alert budget (k=1,559):

- Overlap TP: **1,164**
- GAT-only TP (CatBoost misses): **355**
- CatBoost-only TP (GAT misses): **173**

Positive mean rank (0 = best): CatBoost **6,672** vs GAT **1,505** — CatBoost leaves a
long tail of positives far down the list; medians are similar (~907) because both
models rank many positives near the top, but CatBoost’s failures are much worse in
the tail.

## What CatBoost already uses

Global importance (top features) already includes **pairwise relationship window
stats**, not only ego account counts:

1. `payment_type`
2. `relationship_same_currency_amount_sum_30d`
3. `amount`
4. `relationship_unique_counterparties_30d`
5. `relationship_count_30d`
6. `sender_outgoing_count_30d`
…

So the gap is **not** “CatBoost has zero relationship signal.” It has aggregated
sender↔receiver history, but still underperforms GAT.

## Interpretation (interview-safe)

- **Capability split**: tabular windows summarize local behavior; GAT message-passing
  over historical neighborhoods captures multi-hop / structural patterns those
  aggregates do not fully encode.
- **System implication**: keep CatBoost for behavior/PIT coverage and rules
  interaction; keep GAT as the primary ranker / graph component. Do not expect
  CatBoost alone to match GAT PR-AUC on this synthetic graph-heavy label process.
- **Not claimed**: causal proof; real-world transfer; that adding more hand features
  would close the full 0.14 PR-AUC gap.

## Hard-negative retrain（已完成）

Experiment: retrain with
`--hard-negative-oof artifacts/table_oof/table_oof_scores.parquet`
→ `artifacts/table_baseline_hardneg`
(`run_id=20260726T113335Z-a63ad9a6e2`, sampling=`temporal_oof_hard_negative`).

| CatBoost variant | Test PR-AUC |
|---|---:|
| Stable-hash downsample (`table_baseline_rules`) | **0.8092** |
| Temporal OOF hard-negative | **0.4548** |

**Conclusion:** on this setup, OOF hard-negative sampling **hurt** ranking quality versus
the default stable-hash negative downsample. Keep the hash-sampled CatBoost as the
tabular main line; treat hard-negative as a negative ablation, not a win. Possible
causes to discuss in interview (not proven here): OOF score distribution mismatch,
over-concentration on a narrow hard-negative manifold, or interaction with the fixed
500k negative budget — further tuning was not pursued because GAT already covers the
structural gap more cheaply.
