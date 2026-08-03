#!/usr/bin/env bash
# Full-protocol second-seed GAT stability run on one explicitly selected GPU.
set -euo pipefail

ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
mkdir -p logs artifacts/logs
export PATH="/data1/yangjuhao/envs/risk/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

PY=/data1/yangjuhao/envs/risk/bin/python
OUTPUT=artifacts/gat_seed_20260723
LOG=logs/gat_seed_20260723.log
STATUS=artifacts/logs/gat_seed_20260723_status.json

echo "{\"experiment\":\"gat_second_seed\",\"state\":\"running\",\"seed\":20260723,\"started_at\":\"$(date -Is)\"}" > "$STATUS"
{
  echo "[$(date -Is)] START GAT second seed 20260723 on cuda:3"
  if [[ -f "$OUTPUT/metrics.json" && -f "$OUTPUT/graphsage.pt" ]]; then
    echo "[$(date -Is)] SKIP complete output already exists"
  else
    "$PY" -u -m aml_evidence_graph.training.run_graphsage \
      --features artifacts/pit_features \
      --output "$OUTPUT" \
      --model-config configs/models.gat.yaml \
      --device cuda:3 \
      --max-gpus 1 \
      --random-seed 20260723 \
      --overwrite
  fi
  echo "[$(date -Is)] DONE GAT second seed"
  echo "{\"experiment\":\"gat_second_seed\",\"state\":\"ok\",\"seed\":20260723,\"finished_at\":\"$(date -Is)\",\"output\":\"$OUTPUT\"}" > "$STATUS"
} 2>&1 | tee -a "$LOG"
