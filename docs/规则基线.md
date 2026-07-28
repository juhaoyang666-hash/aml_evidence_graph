# 规则基线定阈证据（`configs/rules/default.yaml` v2026.2）

规则基线的用途是给「固定召回下的告警削减率」提供一个可审计的分母，模拟传统
阈值型交易监控。它不是模型候选，也不追求高精度。

## 定阈协议

- 数据：`artifacts/pit_features`（公开合成 SAML-D，PIT 特征，
  `run_id=20260724T000855Z-e3d573661f`）
- **仅使用训练期分区**（`split=train`，2022-10-07..2023-04-30，6,131,187 行 / 6,170 正例）
- 验证期与测试期在定阈过程中未被读取
- 阈值取训练期经验分位数，再固定写入 YAML；不做逐期重搜

## 单规则表现（训练期）

| rule_id | 特征 | 阈值 | 命中率 | 召回 | Lift |
|---|---|---|---|---|---|
| R-HIGH-AMOUNT-001 | `amount` | ≥ 45,225.28（q99.0） | 1.00% | 0.048 | 4.78× |
| R-NEW-SENDER-002 | `is_new_sender_account` | ≥ 1 | 4.99% | 0.200 | 4.01× |
| R-FAN-IN-VELOCITY-003 | `receiver_incoming_unique_counterparties_7d` | ≥ 20（q99.0） | 2.63% | 0.002 | 0.07× |

## 组合基线（任一规则命中即告警）

| 切分 | 告警率 | 告警数 | 召回 | 精确率 |
|---|---|---|---|---|
| train | 8.36% | 512,386 | 0.235 | 0.00283 |
| test | 8.79% | 136,985 | 0.215 | 0.00284 |

这是典型的传统规则表现形态：召回约 20%，告警量接近总交易量的 9%，精确率约
0.28%（约 352 条告警才命中 1 例）。模型侧的价值应表述为**同召回下把告警量压到多少**。

## 关于被淘汰的候选规则

初版 `default.yaml` 的 `R-CROSS-BORDER-FREQUENCY-001` 阈值为 `null` 且
`status: draft`，因此 `rules/engine.py` 不生成 `rule_*_hit` 列，
`table_baseline` 的 `alert_reduction_vs_rules` 恒为 `{}`。

在训练期扫描后确认，以下高分位速度类阈值在本合成数据集上召回≈0，不适合单独作为基线：

| 候选特征 | q99.5 阈值 | 命中率 | 召回 |
|---|---|---|---|
| `sender_outgoing_cross_border_count_14d` | 108 | 0.50% | 0.0002 |
| `sender_outgoing_count_1d` | 231 | 0.51% | 0.0000 |

`R-FAN-IN-VELOCITY-003` 单独召回也很低，保留它的理由是 typology 覆盖
（Layered_Fan_In / Gather-Scatter 在测试期各有约 113 / 85 例），
且真实场景中此类扇入速度规则普遍存在；其局限已在此文档记录，不做效果宣称。

## 复现

```bash
export PYTHONPATH=src
PY=/data1/yangjuhao/envs/risk/bin/python

# 1) 定阈扫描（只读训练期）
$PY -c "
import pyarrow.dataset as ds, pandas as pd, numpy as np
d=ds.dataset('artifacts/pit_features',format='parquet',partitioning='hive')
df=d.to_table(columns=['amount','is_laundering'],filter=ds.field('split')=='train').to_pandas()
print(np.percentile(pd.to_numeric(df.amount).astype(float),99.0))
"

# 2) 规则命中列需重建 PIT（rule_*_hit 由 features.build 写出）
$PY -m aml_evidence_graph.features.build \
  --input artifacts/prepared_transactions \
  --output artifacts/pit_features \
  --rules configs/rules/default.yaml --overwrite
```

## 改规则后不必重建 PIT

`rule_*_hit` 列原本只在 `features.build` 中生成，因此「改规则 → 重建 PIT」会付出
约 8 小时的全量代价。但阈值规则是**已有列的纯函数**，与因果历史重放无关，
所以可以用后处理脚本直接补写：

```bash
export PYTHONPATH=src
PY=/data1/yangjuhao/envs/risk/bin/python

# 先干跑，确认命中量与列可用性
$PY scripts/backfill_rule_hits.py \
  --features artifacts/pit_features \
  --rules configs/rules/default.yaml --dry-run

# 实际写入（原地重写分区 + 生成 _rule_evidence）
$PY scripts/backfill_rule_hits.py \
  --features artifacts/pit_features \
  --rules configs/rules/default.yaml
```

v2026.2 实测：321 分区 / 9,504,852 行，干跑 **47 秒**，写入 **1 分 48 秒**
（对比全量 PIT 重建约 8 小时，约 **250 倍**）。产出
`_rule_backfill_summary.json` 记录 rule_version 与 rule_ids 以便追溯。

**边界**：当规则引用的特征不在 PIT 产物列中时，脚本会直接报错并要求全量重建，
避免用后处理伪造需要历史重放的特征。若新增的是窗口/图统计类特征，仍须重建 PIT。
