# CatBoost 主线（PR-AUC 0.809）端到端技术路线

本文只讲 **CatBoost 主线表格模型** 这条链路：从原始 SAML-D CSV 到测试集预测与评估。
对应结果来自 `docs/RESULTS.md` 中的 `catboost = 0.8092`（`artifacts/table_baseline_rules`）。

## 1) 目标与边界

- 任务：交易级二分类（`is_laundering`），输出风险概率用于排序。
- 数据：公开合成 SAML-D（约 950 万交易），不是私有真实生产数据。
- 防泄漏原则：
  - 固定时间外切分（train/validation/test）。
  - PIT 特征仅使用预测时点之前历史。
  - 训练不读取测试标签进行拟合或调参。

## 2) 时间切分协议（固定）

来自 `configs/data.yaml` 与 `src/aml_evidence_graph/data/splits.py`：

- train: `2022-10-07` ~ `2023-04-30`
- validation: `2023-05-01` ~ `2023-06-30`
- test: `2023-07-01` ~ `2023-08-23`

这是全项目统一评估协议，CatBoost 也必须遵守。

## 3) 原始数据 -> 规范化分区数据（prepared）

入口：`src/aml_evidence_graph/ingestion/prepare.py`

核心动作：

1. 按 chunk 读取 CSV（默认 250k 行/块）。
2. 标准化字段（`normalize_transaction_chunk`）得到 canonical schema。
3. 生成 `event_date`，并用 `assign_time_split` 写入 `split`。
4. 落盘为 Hive 分区 Parquet：
   - `event_date=YYYY-MM-DD/split=train|validation|test`
5. 产出运行清单 `_run_manifest.json`、汇总 `_conversion_summary.json`。

典型命令（Linux）：

```bash
export PY=/data1/yangjuhao/envs/risk/bin/python
export PYTHONPATH=src
$PY -m aml_evidence_graph.ingestion.prepare \
  --input ../data/SAML-D.csv \
  --output artifacts/prepared_transactions \
  --overwrite
```

## 4) prepared -> PIT 特征数据（pit_features）

入口：`src/aml_evidence_graph/features/build.py`

核心动作（按天分区迭代）：

1. 扫描 `event_date=*` 分区，按时间顺序处理。
2. `PITFeatureBuilder` 计算窗口/时序特征（仅用历史）。
3. `CausalGraphStatisticsBuilder` 计算图统计特征（仍是历史视角）。
4. 应用规则引擎 `apply_rules` 生成 `rule_*_hit`（用于规则基线/KPI分母）。
5. 校验特征元数据（`features.yaml` + 规则特征登记）。
6. 写回分区 Parquet，并生成：
   - `_feature_build_summary.json`
   - `_feature_registry.json`
   - `_run_manifest.json`

典型命令：

```bash
$PY -m aml_evidence_graph.features.build \
  --input artifacts/prepared_transactions \
  --output artifacts/pit_features \
  --rules configs/rules/default.yaml \
  --overwrite
```

## 5) PIT 特征 -> CatBoost 训练与打分

入口：`src/aml_evidence_graph/training/table_baseline.py`

### 5.1 读数与切分

- `load_feature_split()` 从 `pit_features` 读取 `split=train/validation/test`。
- train 仅用于拟合；validation 用于选择与监控；test 仅用于最终报告。

### 5.2 训练集负样本下采样（可复现）

- 默认策略：`deterministic_negative_downsample()`
- 正样本全保留；负样本按 `source_row_number` 稳定哈希取前 N（默认最多 500k）。
- 目的：降算力开销、保持可复现，不引入随机波动。

### 5.3 模型参数

来自 `configs/models.yaml`：

- `iterations: 800`
- `depth: 8`
- `learning_rate: 0.05`
- `loss_function: Logloss`
- `eval_metric: PRAUC`

### 5.4 训练与产物

- 模型拟合：`fit_table_models()`
- 输出目录：`artifacts/table_baseline_rules`（主线结果来源）
- 关键产物：
  - `models/table_baselines/catboost.cbm`
  - `scores/table_validation_scores.parquet`
  - `scores/table_test_scores.parquet`
  - `run_manifest.json`
  - `metrics.json`（指标总表）

典型命令：

```bash
$PY -m aml_evidence_graph.training.table_baseline \
  --features artifacts/pit_features \
  --output artifacts/table_baseline_rules \
  --model-config configs/models.yaml \
  --overwrite
```

## 6) 评估逻辑（为什么是 0.809）

`table_baseline.py` 会调用：

- `evaluate_binary_risk_scores()`：PR-AUC、ROC-AUC、预算 Precision/Recall、固定 FPR Recall 等。
- `compare_alert_volume_at_fixed_recall()`：同召回相对规则基线的告警量削减。
- 切片监控：月度稳定性、新账户、支付方式、地区对、币种对等。

在 `docs/RESULTS.md` 对应主线结果（test）：

- CatBoost PR-AUC: **0.8092**
- 0.1% 告警预算：Precision **0.8576** / Recall **0.7375**
- 对照规则（v2026.2）在同召回下可实现约 99.9% 告警量削减（详见 RESULTS/RULE_BASELINE）

## 7) 从预测到下游（CatBoost 线）

CatBoost 分数会进入两类下游：

1. **独立表格报告**（就是 0.809 这条线本身）。
2. **融合输入**（与图模型 OOF 分数组合，形成 `catboost + GAT` 主线融合）。

注意：本文件聚焦 CatBoost 单模型链路；融合链路见 `docs/FULL_RUN_AFTER_PIT.md` 与 `docs/RESULTS.md`。

## 8) 一句话总结

`SAML-D.csv -> prepared_transactions -> pit_features -> table_baseline_rules(catboost)`  
在严格时间外切分 + PIT 防泄漏协议下，得到 test PR-AUC **0.8092**。这条线是主线表格基座，也是后续图融合的 tabular 侧输入。

