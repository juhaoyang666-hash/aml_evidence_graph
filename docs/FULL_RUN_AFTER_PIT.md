# 全量 PIT 完成后的正式运行清单

> 用途：`artifacts/pit_features` 写出 `_run_manifest.json` 且分区完整后执行。  
> **不要**沿用烟雾路径的 `--splits 1 --minimum-training-months 1` 或 `models.smoke.yaml`。  
> 下列指标可写入模型卡；烟雾指标不可引用。

## 0. 开工前检查

```powershell
$pit = "artifacts\pit_features"
Test-Path "$pit\_run_manifest.json"
@(Get-ChildItem $pit -Directory -Filter "event_date=*").Count
# 预期：存在 manifest；分区数覆盖 train/val/test 全协议日期（约 300+ 天）
```

可选：确认全量 PIT 进程已结束，且未与烟雾输出互相覆盖。

## 1. 环境

```powershell
cd d:\homework\反洗钱\aml_evidence_graph
$env:PYTHONPATH = "src"   # 若安装锁住 CLI，可继续用模块方式
$py = "D:\Miniconda3\envs\aml-evidence\python.exe"
# 或使用已安装的 aml-*.exe
```

配置：默认 `configs/models.yaml`（非 `models.smoke.yaml`）。

## 2. 表格基线

```powershell
& $py -m aml_evidence_graph.training.table_baseline `
  --features artifacts/pit_features `
  --output artifacts/table_baseline `
  --model-config configs/models.yaml `
  --overwrite
```

## 3. GraphSAGE（GPU 默认 auto）

```powershell
& $py -m aml_evidence_graph.training.run_graphsage `
  --features artifacts/pit_features `
  --output artifacts/graphsage `
  --model-config configs/models.yaml `
  --device auto `
  --overwrite
```

成熟版多架构对比（可选，改 `configs/models.yaml` 的 `graphsage.architecture` 为
`gat` / `rgcn` / `pna` 后分别输出到独立目录，例如 `artifacts/gat`）。

## 4. Expanding-time OOF（正式参数）

```powershell
& $py -m aml_evidence_graph.training.oof `
  --features artifacts/pit_features `
  --output artifacts/table_oof `
  --model table `
  --model-config configs/models.yaml `
  --splits 3 `
  --minimum-training-months 2 `
  --overwrite

& $py -m aml_evidence_graph.training.oof `
  --features artifacts/pit_features `
  --output artifacts/graph_oof `
  --model graphsage `
  --model-config configs/models.yaml `
  --splits 3 `
  --minimum-training-months 2 `
  --overwrite
```

## 5. 融合拟合（仅 OOF + 验证集）

```powershell
& $py -m aml_evidence_graph.training.fusion `
  --oof artifacts/table_oof/table_oof_scores.parquet `
  --oof artifacts/graph_oof/graphsage_oof_scores.parquet `
  --validation artifacts/table_baseline/scores/table_validation_scores.parquet `
  --validation artifacts/graphsage/scores/graphsage_validation_scores.parquet `
  --components catboost,graph_stats_catboost,graphsage `
  --output artifacts/fusion `
  --model-config configs/models.yaml `
  --overwrite
```

## 6. 冻结测试评估（一次性）

```powershell
& $py -m aml_evidence_graph.training.evaluate_fusion `
  --fusion-dir artifacts/fusion `
  --test artifacts/table_baseline/scores/table_test_scores.parquet `
  --test artifacts/graphsage/scores/graphsage_test_scores.parquet `
  --features artifacts/pit_features `
  --output artifacts/fusion_test `
  --overwrite
```

需要置信区间时再加 `--bootstrap-iterations 200`（更慢）。

## 7. 调查视图（后评分聚合）

```powershell
& $py -m aml_evidence_graph.aggregation.views `
  --transactions artifacts/pit_features `
  --scores artifacts/fusion_test/test_fusion_scores.parquet `
  --score-column fusion_calibrated_probability `
  --as-of-ts 2023-08-23T23:59:59Z `
  --output artifacts/test_investigation_views
```

## 8. 文档回填

全量跑完后更新（必须带 run_id 与数据为合成 SAML-D 的声明）：

1. `docs/MODEL_CARD.md`：正式 PR-AUC、告警削减率等  
2. `docs/IMPLEMENTATION_STATUS.md` / `docs/COMPLETION_AUDIT.md`  
3. 上级进展文档（若使用）`docs/AML_CURRENT_PROGRESS_*.md`

## 与烟雾路径对照

| 项 | 烟雾（已完成） | 正式（本清单） |
|---|---|---|
| 特征根 | `pit_features_smoke` | `pit_features` |
| 模型配置 | `models.smoke.yaml` | `models.yaml` |
| OOF | `--splits 1 --minimum-training-months 1` | `--splits 3 --minimum-training-months 2` |
| 产物根 | `artifacts/models_smoke/` | `artifacts/{table_baseline,graphsage,...}` |
| 指标用途 | 仅工程调试 | 可写模型卡（合成基准） |
