#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
export PATH="/data1/yangjuhao/envs/risk/bin:$HOME/.local/bin:/usr/bin:$PATH"
export PY="/data1/yangjuhao/envs/risk/bin/python"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-aml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p logs

log() { echo "[$(date -Is)] $*"; }

find_table_pid() {
  local pid cmd
  for pid in $(pgrep -f '/data1/yangjuhao/envs/risk/bin/python' || true); do
    cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
    if [[ "${cmd}" == *"-m aml_evidence_graph.training.table_baseline"* ]]; then
      echo "${pid}"
      return 0
    fi
  done
  return 1
}

while TABLE_PID=$(find_table_pid); do
  log "waiting for table_baseline pid=${TABLE_PID}"
  while kill -0 "${TABLE_PID}" 2>/dev/null; do
    sleep 30
  done
  sleep 5
done

if [[ ! -f artifacts/table_baseline/metrics.json ]]; then
  log "ERROR: table_baseline finished but metrics.json missing"
  tail -100 logs/table_baseline.log || true
  exit 1
fi
log "table_baseline OK; continuing"

run_logged() {
  local name="$1"; shift
  log "START ${name}"
  "$@" 2>&1 | tee "logs/${name}.log"
  log "OK ${name}"
}

if [[ ! -f artifacts/graphsage/metrics.json ]]; then
  run_logged graphsage "$PY" -m aml_evidence_graph.training.run_graphsage \
    --features artifacts/pit_features --output artifacts/graphsage \
    --model-config configs/models.yaml --device auto --overwrite
else
  log "SKIP graphsage"
fi

if [[ ! -f artifacts/table_oof/table_oof_scores.parquet ]]; then
  run_logged table_oof "$PY" -m aml_evidence_graph.training.oof \
    --features artifacts/pit_features --output artifacts/table_oof \
    --model table --model-config configs/models.yaml \
    --splits 3 --minimum-training-months 2 --overwrite
else
  log "SKIP table_oof"
fi

if [[ ! -f artifacts/graph_oof/graphsage_oof_scores.parquet ]]; then
  run_logged graph_oof "$PY" -m aml_evidence_graph.training.oof \
    --features artifacts/pit_features --output artifacts/graph_oof \
    --model graphsage --model-config configs/models.yaml \
    --splits 3 --minimum-training-months 2 --overwrite
else
  log "SKIP graph_oof"
fi

if [[ ! -f artifacts/fusion/_run_manifest.json ]]; then
  run_logged fusion "$PY" -m aml_evidence_graph.training.fusion \
    --oof artifacts/table_oof/table_oof_scores.parquet \
    --oof artifacts/graph_oof/graphsage_oof_scores.parquet \
    --validation artifacts/table_baseline/scores/table_validation_scores.parquet \
    --validation artifacts/graphsage/scores/graphsage_validation_scores.parquet \
    --components catboost,graph_stats_catboost,graphsage \
    --output artifacts/fusion --model-config configs/models.yaml --overwrite
else
  log "SKIP fusion"
fi

if [[ ! -f artifacts/fusion_test/metrics.json && ! -f artifacts/fusion_test/test_fusion_scores.parquet ]]; then
  run_logged fusion_test "$PY" -m aml_evidence_graph.training.evaluate_fusion \
    --fusion-dir artifacts/fusion \
    --test artifacts/table_baseline/scores/table_test_scores.parquet \
    --test artifacts/graphsage/scores/graphsage_test_scores.parquet \
    --features artifacts/pit_features --output artifacts/fusion_test --overwrite
else
  log "SKIP fusion_test"
fi

if [[ ! -d artifacts/test_investigation_views ]] || [[ -z "$(ls -A artifacts/test_investigation_views 2>/dev/null || true)" ]]; then
  run_logged investigation_views "$PY" -m aml_evidence_graph.aggregation.views \
    --transactions artifacts/pit_features \
    --scores artifacts/fusion_test/test_fusion_scores.parquet \
    --score-column fusion_calibrated_probability \
    --as-of-ts 2023-08-23T23:59:59Z \
    --output artifacts/test_investigation_views
else
  log "SKIP investigation_views"
fi

log "REMAINING CHAIN COMPLETE"
date -Is > logs/remaining_chain_done.flag
