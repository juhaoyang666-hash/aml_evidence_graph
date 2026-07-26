# Batch feature replay (Tier B4)

**Artifact**: `artifacts/batch_feature_replay/`  
**Command**: `PYTHONPATH=src python scripts/run_batch_feature_replay.py`

Replays **`sender_outgoing_count_7d`** with DuckDB window + Polars rolling against official PIT (read-only).

## Result (5 scored days + 7d lookback)

| Engine | Match rate vs PIT | Approx rows/s |
|---|---:|---:|
| DuckDB `RANGE` window | **1.0** | ~3.7e5 |
| Polars rolling | **1.0** | ~9.2e4 |

## Honest boundary

Does **not** replace official Python PIT. Demonstrates the same protocol on a vectorized engine for interview / migration talk tracks. See also [BATCH_FEATURE_NOTE.md](BATCH_FEATURE_NOTE.md).
