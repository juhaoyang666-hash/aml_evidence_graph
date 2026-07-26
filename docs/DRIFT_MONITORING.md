# Drift / rolling monitoring drill

**Status**: offline drill on frozen main-line scores (no monthly retrain).  
**Artifacts**: `artifacts/drift_monitoring/`  
**Command**:

```bash
export PYTHONPATH=src
python scripts/run_drift_monitoring.py
```

## Protocol

- Scores: CatBoost (`table_baseline_rules`), GAT (`artifacts/gat`), fusion (`fusion_test_cb_gat`).
- Test months: 2023-07, 2023-08 (fixed time-out split).
- Operational alert fraction: **0.5%** (validation-selected), matching fusion policy.
- Recalibration fits **isotonic + threshold on validation only**; test labels are evaluation-only.

## Monthly ranking & ops curves

Frozen validation quantile / policy threshold applied month-by-month:

| Model | Month | n | Pos rate | PR-AUC | ECE@10 | Alert rate | P@thr | R@thr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | 2023-07 | 902,239 | 0.109% | 0.803 | 0.039 | 0.462% | 0.205 | 0.869 |
| CatBoost | 2023-08 | 656,582 | 0.126% | 0.817 | 0.048 | 0.594% | 0.186 | 0.873 |
| GAT | 2023-07 | 902,239 | 0.109% | 0.944 | 0.003 | 0.448% | 0.239 | 0.981 |
| GAT | 2023-08 | 656,582 | 0.126% | 0.953 | 0.003 | 0.456% | 0.272 | 0.981 |
| fusion CB+GAT | 2023-07 | 902,239 | 0.109% | 0.915 | ≈0 | 0.471% | 0.228 | 0.985 |
| fusion CB+GAT | 2023-08 | 656,582 | 0.126% | 0.921 | ≈0 | 0.517% | 0.242 | 0.992 |

Files: `drift_monthly_curves.csv` / `.parquet`.

**Read**: ranking quality is stable Jul→Aug (slightly up). CatBoost alert rate under a frozen raw quantile drifts up (~0.46%→0.59%) as the score mass shifts — the monitoring signal is **ops volume / precision**, not only PR-AUC.

## Threshold recalibration drill (CatBoost / GAT)

Compare three policies on the **full test set**:

| Model | Policy | Test PR-AUC | Alert rate | Precision | Recall | ECE@10 |
|---|---|---:|---:|---:|---:|---:|
| CatBoost | stale (fit May only) | 0.800 | **0.96%** | 0.110 | 0.904 | ≈0 |
| CatBoost | fresh (full val) | 0.801 | 0.54% | 0.189 | 0.872 | ≈0 |
| CatBoost | raw val quantile (no isotonic) | **0.809** | 0.52% | 0.196 | 0.871 | 0.043 |
| GAT | stale (May only) | 0.944 | 0.55% | 0.209 | 0.985 | ≈0 |
| GAT | fresh (full val) | 0.945 | 0.51% | 0.225 | 0.983 | ≈0 |
| GAT | raw val quantile | **0.948** | 0.45% | 0.253 | 0.981 | 0.003 |

Expanding-window CSV: `drift_expanding_window.csv` (fit through month *M*, score next month).

**Interview line**: “When the calibrator/threshold goes stale, ranking barely moves but **alert volume and precision do** — so the first lever is re-fit calibration on recent labeled validation, not a full model retrain.”

## Fusion frozen vs month-oracle quantile

`drift_fusion_oracle_vs_frozen.csv` — oracle retunes the 0.5% cut **using that month’s scores** (cheating upper bound for ops, not for ranking):

| Month | Frozen alert rate | Oracle alert rate | Frozen P | Oracle P |
|---|---:|---:|---:|---:|
| 2023-07 | 0.471% | 0.552% | 0.228 | 0.195 |
| 2023-08 | 0.517% | 0.517% | 0.242 | 0.242 |

August already sits on the budget; July’s frozen cut is slightly tighter than a month-local quantile.

## Honest boundary

- Synthetic SAML-D; not production monitoring or third-party labels.
- **No monthly retrain** — frozen CatBoost/GAT/fusion scores only.
- Does not overturn main-line PR-AUC (CatBoost 0.809 / GAT 0.948 / fusion 0.918).
