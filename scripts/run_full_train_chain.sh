#!/usr/bin/env bash
# Full post-PIT training chain (formal configs; no smoke params).
set -euo pipefail

ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
export PATH="/data1/yangjuhao/envs/risk/bin:$HOME/.local/bin:/usr/bin:$PATH"
export PY="/data1/yangjuhao/envs/risk/bin/python"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-aml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p logs artifacts

log() { echo "[$(date -Is)] $*"; }

run_step() {
  local name="$1"
  shift
  local logfile="logs/${name}.log"
  log "START ${name}" | tee -a "$logfile"
  if "$@"; then
    log "OK ${name}" | tee -a "$logfile"
  else
    local code=$?
    log "FAIL ${name} exit=${code}" | tee -a "$logfile"
    return "$code"
  fi
}

# 2. table_baseline (skip if complete metrics present)
if [[ -f artifacts/table_baseline/metrics.json && -f artifacts/table_baseline/scores/table_validation_scores.parquet ]]; then
  log "SKIP table_baseline (artifacts present)"
else
  run_step table_baseline \
    "$PY" -m aml_evidence_graph.training.table_baseline \
      --features artifacts/pit_features \
      --output artifacts/table_baseline \
      --model-config configs/models.yaml \
      --overwrite \
    2>&1 | tee -a logs/table_baseline.log
fi

# 3. GraphSAGE
if [[ -f artifacts/graphsage/metrics.json && -f artifacts/graphsage/scores/graphsage_validation_scores.parquet ]]; then
  log "SKIP graphsage (artifacts present)"
else
  run_step graphsage \
    "$PY" -m aml_evidence_graph.training.run_graphsage \
      --features artifacts/pit_features \
      --output artifacts/graphsage \
      --model-config configs/models.yaml \
      --device auto \
      --overwrite \
    2>&1 | tee -a logs/graphsage.log
fi

# 4a. table OOF
if [[ -f artifacts/table_oof/table_oof_scores.parquet ]]; then
  log "SKIP table_oof (artifacts present)"
else
  run_step table_oof \
    "$PY" -m aml_evidence_graph.training.oof \
      --features artifacts/pit_features \
      --output artifacts/table_oof \
      --model table \
      --model-config configs/models.yaml \
      --splits 3 \
      --minimum-training-months 2 \
      --overwrite \
    2>&1 | tee -a logs/table_oof.log
fi

# 4b. graph OOF
if [[ -f artifacts/graph_oof/graphsage_oof_scores.parquet ]]; then
  log "SKIP graph_oof (artifacts present)"
else
  run_step graph_oof \
    "$PY" -m aml_evidence_graph.training.oof \
      --features artifacts/pit_features \
      --output artifacts/graph_oof \
      --model graphsage \
      --model-config configs/models.yaml \
      --splits 3 \
      --minimum-training-months 2 \
      --overwrite \
    2>&1 | tee -a logs/graph_oof.log
fi

# 5. fusion
if [[ -f artifacts/fusion/fusion_model.joblib || -f artifacts/fusion/metrics.json || -f artifacts/fusion/_run_manifest.json ]]; then
  # Prefer concrete score/model outputs if present
  if [[ -f artifacts/fusion/_run_manifest.json ]]; then
    log "SKIP fusion (manifest present)"
  else
    run_step fusion \
      "$PY" -m aml_evidence_graph.training.fusion \
        --oof artifacts/table_oof/table_oof_scores.parquet \
        --oof artifacts/graph_oof/graphsage_oof_scores.parquet \
        --validation artifacts/table_baseline/scores/table_validation_scores.parquet \
        --validation artifacts/graphsage/scores/graphsage_validation_scores.parquet \
        --components catboost,graph_stats_catboost,graphsage \
        --output artifacts/fusion \
        --model-config configs/models.yaml \
        --overwrite \
      2>&1 | tee -a logs/fusion.log
  fi
else
  run_step fusion \
    "$PY" -m aml_evidence_graph.training.fusion \
      --oof artifacts/table_oof/table_oof_scores.parquet \
      --oof artifacts/graph_oof/graphsage_oof_scores.parquet \
      --validation artifacts/table_baseline/scores/table_validation_scores.parquet \
      --validation artifacts/graphsage/scores/graphsage_validation_scores.parquet \
      --components catboost,graph_stats_catboost,graphsage \
      --output artifacts/fusion \
      --model-config configs/models.yaml \
      --overwrite \
    2>&1 | tee -a logs/fusion.log
fi

# 6. evaluate fusion on frozen test
if [[ -f artifacts/fusion_test/test_fusion_scores.parquet || -f artifacts/fusion_test/metrics.json ]]; then
  log "SKIP evaluate_fusion (artifacts present)"
else
  run_step fusion_test \
    "$PY" -m aml_evidence_graph.training.evaluate_fusion \
      --fusion-dir artifacts/fusion \
      --test artifacts/table_baseline/scores/table_test_scores.parquet \
      --test artifacts/graphsage/scores/graphsage_test_scores.parquet \
      --features artifacts/pit_features \
      --output artifacts/fusion_test \
      --overwrite \
    2>&1 | tee -a logs/fusion_test.log
fi

# 7. investigation views
if [[ -d artifacts/test_investigation_views ]] && compgen -G "artifacts/test_investigation_views/*" > /dev/null; then
  log "SKIP investigation_views (artifacts present)"
else
  run_step investigation_views \
    "$PY" -m aml_evidence_graph.aggregation.views \
      --transactions artifacts/pit_features \
      --scores artifacts/fusion_test/test_fusion_scores.parquet \
      --score-column fusion_calibrated_probability \
      --as-of-ts 2023-08-23T23:59:59Z \
      --output artifacts/test_investigation_views \
    2>&1 | tee -a logs/investigation_views.log
fi

log "FULL CHAIN COMPLETE"
echo "FULL CHAIN COMPLETE" > logs/full_chain_done.flag
