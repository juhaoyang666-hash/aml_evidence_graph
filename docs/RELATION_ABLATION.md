# Relation-aware ablation (Tier A3)

**Artifacts**: `artifacts/relation_ablation/` · `artifacts/rgcn_rel/`  
**Config**: `configs/models.rgcn_rel.yaml` (`num_relations=4`)  
**CPU ablation**: `PYTHONPATH=src python scripts/run_relation_ablation.py`  
**Full R=4 RGCN**: `python -m aml_evidence_graph.training.run_graphsage --features artifacts/pit_features --output artifacts/rgcn_rel --model-config configs/models.rgcn_rel.yaml --device cuda --max-gpus 4 --overwrite`

## Relation map

| id | Meaning |
|---:|---|
| 0 | domestic × same currency |
| 1 | domestic × conversion |
| 2 | cross-border × same currency |
| 3 | cross-border × conversion |

## Full multi-rel RGCN (done)

| Model | Test PR-AUC | Val PR-AUC | 0.1% P | 0.1% R | run_id |
|---|---:|---:|---:|---:|---|
| **GAT** (main-line) | **0.9483** | — | 0.974 | 0.838 | `20260726T023031Z-5ccb81baec` |
| RGCN single-rel | 0.9031 | — | 0.905 | 0.778 | `20260726T062636Z-ef6be55956` |
| **RGCN R=4** (`rgcn_rel`) | **0.8873** | 0.8412 | 0.881 | 0.757 | `20260726T135457Z-045b0b06ab` |

Wall time ≈ **28.5 min**; early-stopped at 7 epochs; GPU peak ≈ 1.1 GB (4×3090 fan-out).

**Verdict**: discrete relation routing on history edges **does not beat** single-rel RGCN or GAT under the same PIT / time-split protocol. Keep GAT as primary graph model; treat R=4 as a negative ablation.

## CPU results (earlier)

### Score slices (existing GAT / single-rel RGCN)

| Relation | GAT PR-AUC | RGCN PR-AUC | n (test) | positives |
|---:|---:|---:|---:|---:|
| 0 | 0.932 | 0.872 | 1,380,504 | 1,100 |
| 1 | 0.975 | 0.936 | 25,534 | 121 |
| 2 | 1.000* | 1.000* | 770 | 7 |
| 3 | 0.973 | 0.957 | 152,013 | 585 |

\*tiny positive count — treat as unstable.

### Edge MLP ± relation embedding

| Model | Test PR-AUC |
|---|---:|
| MLP without relation emb | 0.0379 |
| MLP with relation emb | 0.0363 |
| Helps? | **No** |

## Honest boundary

Synthetic SAML-D. Relation ids reuse PIT flags already available as numeric edge features; routing them through RGCNConv did not help. Does not overturn GAT main-line (0.948).
