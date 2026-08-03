#!/usr/bin/env bash
# FE v2 frozen-fusion investigation views (sidecar).
set -euo pipefail
ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
mkdir -p logs artifacts/logs
export PATH="/data1/yangjuhao/envs/risk/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
PY=/data1/yangjuhao/envs/risk/bin/python
LOG=logs/fe_v2_investigation_views.log
STATUS=artifacts/logs/fe_v2_investigation_views_status.json
OUT=artifacts/test_investigation_views_fe_v2

echo "{\"step\":\"investigation_views\",\"state\":\"running\",\"started_at\":\"$(date -Is)\"}" > "$STATUS"
{
  echo "[$(date -Is)] START fe_v2 investigation views -> $OUT"
  if [[ -f "$OUT/summary.json" ]]; then
    echo "[$(date -Is)] SKIP summary.json present"
  else
    stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.aggregation.views \
      --transactions artifacts/pit_features_fe_v2 \
      --scores artifacts/fusion_test_fe_v2/test_fusion_scores.parquet \
      --score-column fusion_calibrated_probability \
      --as-of-ts 2023-08-23T23:59:59Z \
      --output "$OUT"
  fi
  echo "[$(date -Is)] DONE fe_v2 investigation views"
  echo "{\"step\":\"investigation_views\",\"state\":\"ok\",\"finished_at\":\"$(date -Is)\",\"output\":\"$OUT\"}" > "$STATUS"
  date -Is > logs/fe_v2_investigation_views_done.flag
} 2>&1 | tee -a "$LOG"
