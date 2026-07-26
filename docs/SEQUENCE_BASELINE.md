# Sequence GRU baseline (Tier B1)

**Artifact**: `artifacts/sequence_baseline/`  
**Command**: `PYTHONPATH=src python scripts/run_sequence_baseline.py`  
**tmux**: `aml_b1_seq`

## Protocol

Sender outgoing history (last **K=32** events before `t`) from `prepared_transactions` → GRU → edge laundering score. Features per step: `log1p(amount)`, `log1p(Δt_hours)`, payment_type id, cross-border, currency conversion. Negatives hash-downsampled (~200k/split).

## Result (sampled)

| Split rows | Val PR-AUC | Test PR-AUC | Test ROC-AUC | 0.1% Precision |
|---:|---:|---:|---:|---:|
| train 206,170 / test 201,813 | 0.011 | **0.010** | 0.514 | 0.035 |

Reference: CatBoost **0.809** · GAT **0.948** (full test).

## Reading

Near-chance ranking on this short outgoing-only history — PIT window aggregates + GNN already capture most usable sequential signal. Useful as a **negative / modest** sequence baseline for JD talk tracks, not a replacement.

## Honest boundary

CPU · downsampled · sender-outgoing only · synthetic SAML-D. Does not change main-line numbers.
