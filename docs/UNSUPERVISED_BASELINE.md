# Unsupervised anomaly baseline

**Status**: parallel baseline — does **not** replace CatBoost / GAT.  
**Artifacts**: `artifacts/unsupervised_baseline/`  
**Command**:

```bash
export PYTHONPATH=src
python scripts/run_unsupervised_baseline.py
# optional: --no-autoencoder
```

## Protocol

- Features: numeric PIT columns (76) from `artifacts/pit_features`.
- Fit: train-split partitions, **negatives only**, capped at **400k** rows (partition-sequential sample, not a full scan).
- Score: test-split sample **400k** rows (468 positives, rate 0.117%).
- Models:
  - **Isolation Forest** (`n_estimators=200`, `contamination=auto`)
  - **Shallow Autoencoder** (MLP recon, 8 epochs, CUDA if available) — anomaly score = scaled reconstruction MSE

Scores min-maxed to [0, 1] for the same `evaluate_binary_risk_scores` harness as supervised models.

## Results (sampled test)

| Model | PR-AUC | ROC-AUC | 0.1% Precision | 0.1% Recall |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.0009 | 0.418 | 0.000 | 0.000 |
| Autoencoder | 0.0053 | 0.479 | 0.005 | ~0 |
| Chance ≈ positive rate | ~0.0012 | 0.5 | — | — |
| **CatBoost (full test, main line)** | **0.809** | 0.996 | 0.858 | 0.738 |
| **GAT (full test, main line)** | **0.948** | 1.000 | 0.974 | 0.838 |

Files: `unsupervised_summary.json`, `unsupervised_test_scores.parquet`, `feature_columns.json`.

## Reading

On this synthetic PIT feature space, classical unsupervised detectors are **near or below chance** at ranking laundering edges. That is a useful negative result for JD “anomaly detection” talk tracks: unsupervised is cheap to stand up, but here it does not compete with supervised tabular/graph models under the same feature contract.

Likely contributors (not proven causal): extreme rarity, laundering typology that is not a generic density outlier in PIT aggregates, and sample caps that under-represent rare patterns.

## Honest boundary

- Sampled partitions — not full 9.5M-row unsupervised training.
- Parallel baseline only; main-line numbers unchanged.
- Synthetic SAML-D; not production anomaly monitoring.
