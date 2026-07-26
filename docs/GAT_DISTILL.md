# GAT → CatBoost feature distillation (Tier B2)

**Artifact**: `artifacts/table_baseline_gat_distill/`  
**Command**: `PYTHONPATH=src python scripts/run_gat_distill_catboost.py`  
**tmux**: `aml_b2_distill`

## Protocol

| Split | `gat_oof` source |
|---|---|
| Train | `artifacts/graph_oof_gat` (expanding OOF — leakage-safe) |
| Val / Test | Frozen full-model GAT scores from `artifacts/gat` |

CatBoost trains on PIT features (excluding `graph_*`) **plus** `gat_oof`. Same 500k negative downsample as main-line table training.

## Result

| Model | Test PR-AUC |
|---|---:|
| CatBoost (main-line) | 0.8092 |
| **CatBoost + GAT score feature** | **0.9656** |
| GAT alone | 0.9483 |

`beats_catboost=True`; also ranks above GAT alone on this split because the tabular head can blend residual PIT features with the frozen graph score.

## How to talk about it

This is **not** “CatBoost suddenly learned neighborhoods.” It is a **two-stage scorer**: run GAT, feed the probability into CatBoost with PIT features. Valid ops pattern when graph inference is already in the pipeline; **not** an independent tabular breakthrough.

## Honest boundary

Synthetic SAML-D. Does not replace reporting GAT alone or the OOF logistic fusion (0.918). If the interview asks for pure tabular, quote 0.809 without `gat_oof`.
