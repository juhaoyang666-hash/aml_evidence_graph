# FE v2 后续实验安排（2026-07-28）

## 当前事实

- `artifacts/pit_features_fe_v2` 已完成：321 个日期分区、9,504,852 行。
- `artifacts/table_baseline_fe_v2` 已完成；正式真值来源为其 `metrics.json`：
  - validation PR-AUC：`0.8660898669`
  - test PR-AUC：`0.8754139061`
  - 相对 v1 CatBoost `0.8092`：`+0.0662`
- 旧 `fe_v2_pipeline_status.json` 曾误记 `0.9996`，已按正式 `metrics.json` 纠正。
- FE v2 GAT、OOF、融合、置信区间和调查视图此前均未完成。

## P0：必须补齐的同协议实验

本机使用 `scripts/run_fe_v2_gat_oof_fusion.ps1`，按单卡串行、产物可恢复的方式执行：

| 顺序 | 实验 | 输出目录 | 完成标准 |
|---|---|---|---|
| 1 | FE v2 GAT 全量训练与时间外测试 | `artifacts/gat_fe_v2` | `metrics.json` |
| 2 | FE v2 CatBoost expanding-time OOF | `artifacts/table_oof_fe_v2` | `table_oof_scores.parquet` |
| 3 | FE v2 GAT expanding-time OOF | `artifacts/graph_oof_fe_v2` | `graphsage_oof_scores.parquet` |
| 4 | CatBoost + GAT OOF 融合及验证期校准 | `artifacts/fusion_fe_v2` | `run_manifest.json` |
| 5 | 冻结测试期一次性评估 | `artifacts/fusion_test_fe_v2` | `metrics.json` |
| 6 | 融合测试 200 次分层 Bootstrap | `artifacts/fusion_test_fe_v2_bootstrap` | `metrics.json` 含置信区间 |
| 7 | 冻结融合分调查视图 | `artifacts/test_investigation_views_fe_v2` | `summary.json` |

运行协议保持与 v1 主线一致：GAT 使用 `configs/models.gat.yaml`，OOF 为 3 折、至少 2 个
训练月，融合只使用 `catboost,graphsage` 两列；其中 `graphsage` 列实际承载 GAT 分数。
本机硬件为单张 RTX 2060 6GB，因此设置 `--max-gpus 1`，不并发运行 table OOF 与图训练。

## 判定问题

最终必须同时回答：

1. FE v2 GAT 是否超过 v1 GAT test PR-AUC `0.9483`？
2. FE v2 融合是否超过 v1 `catboost + GAT` 的 `0.9175`？
3. FE v2 融合是否超过 FE v2 GAT 单模型？
4. 提升是否超出 200 次 Bootstrap 置信区间的不确定性？
5. 固定 0.1% / 0.5% 告警预算下 Precision、Recall 是否同步改善？
6. 调查视图的账户、资金路径和案件数量是否出现不可解释的膨胀？

在上述问题回答前，FE v2 保持 sidecar，不覆盖 README / RESULTS 中的 v1 主线数字。

## P1：主链完成后再决定的消融

- P0 特征族 vs P0+P1 全量特征族，用于判断 `+0.0662` 的主要来源。
- 至少一个额外随机种子复跑最优配置，排除单次优化波动。
- 若 GAT 或融合提升明显，再做月度稳定性、固定阈值告警量和校准漂移对比。
- 若 CatBoost 再次出现接近完美的异常结果，优先做字段/时间可用性审计，不继续堆模型。

这些消融不与 P0 主链并发，避免当前 32GB 内存和 6GB GPU 发生资源争用。

## 运行与监控

后台运行器：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run_fe_v2_gat_oof_fusion.ps1 `
  -PythonExecutable D:\Miniconda3\envs\aml-evidence\python.exe `
  -Device cuda -MaxGpus 1
```

状态与日志：

- `artifacts/logs/fe_v2_gat_oof_fusion_status.json`
- `artifacts/logs/fe_v2_gat_oof_fusion_windows.log`
- `artifacts/logs/fe_v2_gat_oof_fusion_runner.pid`

重启同一命令时，已有完整产物会自动跳过；只有传入 `-ForceRetrain` 才会强制重跑。
