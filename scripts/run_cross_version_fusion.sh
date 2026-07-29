#!/usr/bin/env bash
# CatBoost FE v2 + historical GAT v1 fusion (this Linux machine has v1 artifacts).
# Sidecar only; does not overwrite pit_features / gat / fusion_cb_gat.
set -euo pipefail
ROOT="/data1/yangjuhao/反洗钱/aml_evidence_graph"
cd "$ROOT"
mkdir -p logs artifacts/logs
export PATH="/data1/yangjuhao/envs/risk/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
PY=/data1/yangjuhao/envs/risk/bin/python
LOG=logs/cross_version_cb_fe_v2_gat_v1.log
STATUS=artifacts/logs/cross_version_fusion_status.json
FUSION=artifacts/fusion_catboost_fe_v2_gat_v1
FTEST=artifacts/fusion_test_catboost_fe_v2_gat_v1
FBOOT=artifacts/fusion_test_catboost_fe_v2_gat_v1_bootstrap

echo "{\"pipeline\":\"cross_version_cb_fe_v2_gat_v1\",\"state\":\"running\",\"note\":\"uses historical artifacts/gat + graph_oof_gat (not Windows local_replay)\",\"started_at\":\"$(date -Is)\"}" > "$STATUS"
{
  echo "[$(date -Is)] START cross-version CatBoost FE v2 + GAT v1 (historical scores)"
  echo "[$(date -Is)] NOTE: not gat_v1_local_replay; this machine has authoritative v1 GAT artifacts"

  if [[ -f "$FUSION/run_manifest.json" ]]; then
    echo "[$(date -Is)] SKIP fusion fit (manifest present)"
  else
    echo "[$(date -Is)] STEP fusion fit"
    stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.fusion \
      --oof artifacts/table_oof_fe_v2/table_oof_scores.parquet \
      --oof artifacts/graph_oof_gat/graphsage_oof_scores.parquet \
      --validation artifacts/table_baseline_fe_v2/scores/table_validation_scores.parquet \
      --validation artifacts/gat/scores/graphsage_validation_scores.parquet \
      --components catboost,graphsage \
      --output "$FUSION" \
      --model-config configs/models.yaml \
      --overwrite
  fi

  if [[ -f "$FTEST/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP fusion test (metrics present)"
  else
    echo "[$(date -Is)] STEP fusion test"
    stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.evaluate_fusion \
      --fusion-dir "$FUSION" \
      --test artifacts/table_baseline_fe_v2/scores/table_test_scores.parquet \
      --test artifacts/gat/scores/graphsage_test_scores.parquet \
      --features artifacts/pit_features_fe_v2 \
      --output "$FTEST" \
      --overwrite
  fi

  if [[ -f "$FBOOT/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP fusion bootstrap (metrics present)"
  else
    echo "[$(date -Is)] STEP fusion bootstrap 200"
    stdbuf -oL -eL "$PY" -u -m aml_evidence_graph.training.evaluate_fusion \
      --fusion-dir "$FUSION" \
      --test artifacts/table_baseline_fe_v2/scores/table_test_scores.parquet \
      --test artifacts/gat/scores/graphsage_test_scores.parquet \
      --features artifacts/pit_features_fe_v2 \
      --output "$FBOOT" \
      --bootstrap-iterations 200 \
      --overwrite
  fi

  echo "[$(date -Is)] SUMMARY"
  "$PY" - <<'PY'
import json
from pathlib import Path
ft = json.loads(Path("artifacts/fusion_test_catboost_fe_v2_gat_v1/metrics.json").read_text())
fb = json.loads(Path("artifacts/fusion_test_catboost_fe_v2_gat_v1_bootstrap/metrics.json").read_text())
gat = json.loads(Path("artifacts/gat/metrics.json").read_text())
cb = json.loads(Path("artifacts/table_baseline_fe_v2/metrics.json").read_text())
print("catboost_fe_v2", cb["test_metrics"]["catboost"]["pr_auc"])
print("gat_v1_hist", gat["test_metrics"]["pr_auc"])
print("cross_fusion", ft["test_metrics"]["pr_auc"])
print("bootstrap", fb.get("test_bootstrap_intervals"))
PY
  echo "[$(date -Is)] DONE cross-version fusion chain"
  echo "{\"pipeline\":\"cross_version_cb_fe_v2_gat_v1\",\"state\":\"ok\",\"finished_at\":\"$(date -Is)\",\"outputs\":{\"fusion\":\"$FUSION\",\"fusion_test\":\"$FTEST\",\"bootstrap\":\"$FBOOT\"}}" > "$STATUS"
  date -Is > logs/cross_version_cb_fe_v2_gat_v1_done.flag
} 2>&1 | tee -a "$LOG"
