# Nonlinear fusion ablation (Tier B)

**Status**: ablation — logistic OOF fusion remains main-line.  
**Artifacts**: `artifacts/nonlinear_fusion/`  
**Command**:

```bash
export PYTHONPATH=src
python scripts/run_nonlinear_fusion.py
```

## Protocol

- Components: CatBoost + GAT OOF (`graphsage` column) — 4,524,912 train OOF rows.
- Heads fitted on OOF only: logistic (reproduced), HistGradientBoosting, MLP `(16,8)`.
- Isotonic calibration + 0.5% threshold on validation; test evaluation only.

## Test results

| Fusion head | Test PR-AUC | 0.1% P | 0.1% R |
|---|---:|---:|---:|
| **Logistic (main-line style)** | **0.9177** | 0.934 | 0.803 |
| MLP | 0.9073 | 0.912 | 0.784 |
| HistGBDT | 0.0410 | 0.178 | 0.153 |
| Reference artifact `fusion_test_cb_gat` | 0.9175 | 0.936 | 0.805 |
| GAT alone | 0.9483 | 0.974 | 0.838 |

**Verdict**: nonlinear heads do **not** beat logistic fusion; HistGBDT collapses under class imbalance with default settings. Keep logistic. Still below GAT alone.

## Relation to A3 (heterogeneous / relation-aware)

Neural relation ablations (see [RELATION_ABLATION.md](RELATION_ABLATION.md) / RESULTS):

| Model | Test PR-AUC |
|---|---:|
| GAT | 0.9483 |
| RGCN single-rel | 0.9031 |
| RGCN R=4 | 0.8873 |

This file covers the fusion-head upgrade (B3) only.

## Honest boundary

Synthetic SAML-D · does not change the reported main-line fusion (0.9175) · fusion still does not beat GAT alone.
