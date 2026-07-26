# 全量 PIT 完成后的正式运行清单

> 用途：`artifacts/pit_features` 写出 `_run_manifest.json` 且分区完整后执行。  
> **不要**沿用烟雾路径的 `--splits 1 --minimum-training-months 1` 或 `models.smoke.yaml`。  
> 下列指标可写入模型卡；烟雾指标不可引用。
>
> **环境**：Linux + conda `risk`。详见 [ENVIRONMENT.md](ENVIRONMENT.md)。  
> **主线报告**：`catboost` + **GAT**（`catboost + GAT` 融合）；`graph_stats` 与原
> GraphSAGE 融合仅作对照。数字见 [RESULTS.md](RESULTS.md)。

```bash
export PY=/data1/yangjuhao/envs/risk/bin/python
export PYTHONPATH=src
cd /data1/yangjuhao/反洗钱/aml_evidence_graph
# 长任务请放在 tmux 中
```

成套脚本：`scripts/run_full_train_chain.sh`、`scripts/run_remaining_gpu.sh`、
`scripts/run_arch_comparison.sh`、`scripts/run_table_baseline_with_rules.sh`。

## 0. 开工前检查

```bash
test -f artifacts/pit_features/_run_manifest.json
ls -d artifacts/pit_features/event_date=* | wc -l
# 预期：存在 manifest；分区约 300+（全量 321）
```

## 1. 表格基线（含规则 KPI 时）

先确认规则 v2026.2 与 `rule_*_hit`（可用 `scripts/backfill_rule_hits.py`）：

```bash
$PY -m aml_evidence_graph.training.table_baseline \
  --features artifacts/pit_features \
  --output artifacts/table_baseline_rules \
  --model-config configs/models.yaml \
  --overwrite
```

## 2. 图边分类：主线 GAT

```bash
$PY -m aml_evidence_graph.training.run_graphsage \
  --features artifacts/pit_features \
  --output artifacts/gat \
  --model-config configs/models.gat.yaml \
  --device cuda --max-gpus 4 \
  --overwrite
```

同协议对照（可选）：`scripts/run_arch_comparison.sh` → GraphSAGE / RGCN / PNA。

## 3. Expanding-time OOF（正式参数）

```bash
$PY -m aml_evidence_graph.training.oof \
  --features artifacts/pit_features \
  --output artifacts/table_oof \
  --model table \
  --model-config configs/models.yaml \
  --splits 3 --minimum-training-months 2 \
  --overwrite

# 主线：GAT OOF（CLI --model 仍为 graphsage，架构由 model-config 决定；
# 写出列名仍为 graphsage，融合组件名沿用 graphsage）
$PY -m aml_evidence_graph.training.oof \
  --features artifacts/pit_features \
  --output artifacts/graph_oof_gat \
  --model graphsage \
  --model-config configs/models.gat.yaml \
  --splits 3 --minimum-training-months 2 \
  --device cuda --max-gpus 4 \
  --overwrite
```

## 4. 主线融合：catboost + GAT

```bash
$PY -m aml_evidence_graph.training.fusion \
  --oof artifacts/table_oof/table_oof_scores.parquet \
  --oof artifacts/graph_oof_gat/graphsage_oof_scores.parquet \
  --validation artifacts/table_baseline/scores/table_validation_scores.parquet \
  --validation artifacts/gat/scores/graphsage_validation_scores.parquet \
  --components catboost,graphsage \
  --output artifacts/fusion_cb_gat \
  --model-config configs/models.yaml \
  --overwrite
```

> 不要把 `graph_stats_catboost` 放进主线 `--components`。

## 5. 冻结测试评估

```bash
$PY -m aml_evidence_graph.training.evaluate_fusion \
  --fusion-dir artifacts/fusion_cb_gat \
  --test artifacts/table_baseline/scores/table_test_scores.parquet \
  --test artifacts/gat/scores/graphsage_test_scores.parquet \
  --features artifacts/pit_features \
  --output artifacts/fusion_test_cb_gat \
  --overwrite
```

需要置信区间时再加 `--bootstrap-iterations 200`（更慢）。

## 6. 调查视图（后评分聚合）

```bash
$PY -m aml_evidence_graph.aggregation.views \
  --transactions artifacts/pit_features \
  --scores artifacts/fusion_test_cb_gat/test_fusion_scores.parquet \
  --score-column fusion_calibrated_probability \
  --as-of-ts 2023-08-23T23:59:59Z \
  --output artifacts/test_investigation_views_gat
```

## 7. 文档回填

全量跑完后更新（必须带 run_id，并声明数据为公开合成 SAML-D）：

1. `docs/RESULTS.md` / `docs/MODEL_CARD.md`
2. `docs/IMPLEMENTATION_STATUS.md` / `docs/COMPLETION_AUDIT.md`

## 与烟雾路径对照

| 项 | 烟雾（工程验证） | 正式（本清单） |
|---|---|---|
| 特征根 | `pit_features_smoke` | `pit_features` |
| 模型配置 | `models.smoke.yaml` | `models.yaml` / `models.gat.yaml` |
| OOF | `--splits 1 --minimum-training-months 1` | `--splits 3 --minimum-training-months 2` |
| 图 / 融合 | GraphSAGE 烟雾 | **GAT** + `fusion_cb_gat` |
| 指标用途 | 仅调试 | 可写模型卡（合成基准） |
