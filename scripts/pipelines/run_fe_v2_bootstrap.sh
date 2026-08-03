#!/usr/bin/env bash
# FE v2 frozen-fusion test with 200 stratified bootstrap iterations.
# Does not overwrite artifacts/fusion_test_fe_v2; writes sidecar bootstrap dir.
set -euo pipefail
ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
mkdir -p logs artifacts/logs
export PATH="/data1/yangjuhao/envs/risk/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
PY=/data1/yangjuhao/envs/risk/bin/python
LOG=logs/fe_v2_fusion_bootstrap.log
STATUS=artifacts/logs/fe_v2_bootstrap_status.json
OUT=artifacts/fusion_test_fe_v2_bootstrap

echo "{\"step\":\"fusion_bootstrap\",\"state\":\"running\",\"started_at\":\"$(date -Is)\"}" > "$STATUS"
{
  echo "[$(date -Is)] START fe_v2 fusion bootstrap -> $OUT"
  if [[ -f "$OUT/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP already has metrics.json"
  else
    stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.evaluate_fusion \
      --fusion-dir artifacts/fusion_fe_v2 \
      --test artifacts/table_baseline_fe_v2/scores/table_test_scores.parquet \
      --test artifacts/gat_fe_v2/scores/graphsage_test_scores.parquet \
      --features artifacts/pit_features_fe_v2 \
      --output "$OUT" \
      --bootstrap-iterations 200 \
      --overwrite
  fi
  echo "[$(date -Is)] DONE fe_v2 fusion bootstrap"
  echo "{\"step\":\"fusion_bootstrap\",\"state\":\"ok\",\"finished_at\":\"$(date -Is)\",\"output\":\"$OUT\"}" > "$STATUS"
  date -Is > logs/fe_v2_fusion_bootstrap_done.flag
} 2>&1 | tee -a "$LOG"
