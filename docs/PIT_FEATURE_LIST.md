# PIT 特征清单（当前 `main`，features-v2）

本文列出 `aml_evidence_graph` 当前主线在 `features.build` 阶段生成的 **PIT 模型特征**。

口径说明：

- 来源代码：
  - `src/aml_evidence_graph/features/pit.py`
  - `src/aml_evidence_graph/features/graph_stats.py`
  - `src/aml_evidence_graph/features/registry.py`
  - `src/aml_evidence_graph/features/engineering_config.py`
  - `configs/features.yaml`（`features-v2`）
  - `configs/feature_engineering.yaml`（高风险国家 / 支付类型 / 阈值常数）
  - `configs/rules/default.yaml`
- 窗口集合：`1h / 1d / 7d / 14d / 30d`
- 本文仅列“模型特征列”，不含原始字段（如 `transaction_id`, `event_ts`, `is_laundering` 等）。
- FE 试验请写出 sidecar（例如 `artifacts/pit_features_fe_v2`），勿覆盖正式 `artifacts/pit_features`。

## 1. 当前交易即时特征（5）

1. `is_new_sender_account`
2. `is_new_receiver_account`
3. `is_cross_border_current_transaction`
4. `amount_log1p`
5. `is_currency_conversion`

## 2. 账户与关系历史窗口特征（60）

三组前缀：

- `sender_outgoing_*`
- `receiver_incoming_*`
- `relationship_*`

四类统计后缀：

- `count_*`
- `same_currency_amount_sum_*`
- `unique_counterparties_*`
- `cross_border_count_*`

五个窗口：

- `_1h`, `_1d`, `_7d`, `_14d`, `_30d`

### 2.1 sender_outgoing（20）

- `sender_outgoing_count_1h`
- `sender_outgoing_count_1d`
- `sender_outgoing_count_7d`
- `sender_outgoing_count_14d`
- `sender_outgoing_count_30d`
- `sender_outgoing_same_currency_amount_sum_1h`
- `sender_outgoing_same_currency_amount_sum_1d`
- `sender_outgoing_same_currency_amount_sum_7d`
- `sender_outgoing_same_currency_amount_sum_14d`
- `sender_outgoing_same_currency_amount_sum_30d`
- `sender_outgoing_unique_counterparties_1h`
- `sender_outgoing_unique_counterparties_1d`
- `sender_outgoing_unique_counterparties_7d`
- `sender_outgoing_unique_counterparties_14d`
- `sender_outgoing_unique_counterparties_30d`
- `sender_outgoing_cross_border_count_1h`
- `sender_outgoing_cross_border_count_1d`
- `sender_outgoing_cross_border_count_7d`
- `sender_outgoing_cross_border_count_14d`
- `sender_outgoing_cross_border_count_30d`

### 2.2 receiver_incoming（20）

- `receiver_incoming_count_1h`
- `receiver_incoming_count_1d`
- `receiver_incoming_count_7d`
- `receiver_incoming_count_14d`
- `receiver_incoming_count_30d`
- `receiver_incoming_same_currency_amount_sum_1h`
- `receiver_incoming_same_currency_amount_sum_1d`
- `receiver_incoming_same_currency_amount_sum_7d`
- `receiver_incoming_same_currency_amount_sum_14d`
- `receiver_incoming_same_currency_amount_sum_30d`
- `receiver_incoming_unique_counterparties_1h`
- `receiver_incoming_unique_counterparties_1d`
- `receiver_incoming_unique_counterparties_7d`
- `receiver_incoming_unique_counterparties_14d`
- `receiver_incoming_unique_counterparties_30d`
- `receiver_incoming_cross_border_count_1h`
- `receiver_incoming_cross_border_count_1d`
- `receiver_incoming_cross_border_count_7d`
- `receiver_incoming_cross_border_count_14d`
- `receiver_incoming_cross_border_count_30d`

### 2.3 relationship（20）

- `relationship_count_1h`
- `relationship_count_1d`
- `relationship_count_7d`
- `relationship_count_14d`
- `relationship_count_30d`
- `relationship_same_currency_amount_sum_1h`
- `relationship_same_currency_amount_sum_1d`
- `relationship_same_currency_amount_sum_7d`
- `relationship_same_currency_amount_sum_14d`
- `relationship_same_currency_amount_sum_30d`
- `relationship_unique_counterparties_1h`
- `relationship_unique_counterparties_1d`
- `relationship_unique_counterparties_7d`
- `relationship_unique_counterparties_14d`
- `relationship_unique_counterparties_30d`
- `relationship_cross_border_count_1h`
- `relationship_cross_border_count_1d`
- `relationship_cross_border_count_7d`
- `relationship_cross_border_count_14d`
- `relationship_cross_border_count_30d`

## 3. 因果图统计特征（7）

1. `graph_sender_historical_out_degree`
2. `graph_sender_historical_in_degree`
3. `graph_receiver_historical_out_degree`
4. `graph_receiver_historical_in_degree`
5. `graph_directed_edge_prior_count`
6. `graph_reverse_edge_prior_count`
7. `graph_prior_reciprocal_relationship`

## 4. 规则命中特征（3 + 2 交互）

来自 `configs/rules/default.yaml`（v2026.2，`approved`）：

1. `rule_R-HIGH-AMOUNT-001_hit`
2. `rule_R-NEW-SENDER-002_hit`
3. `rule_R-FAN-IN-VELOCITY-003_hit`

规则交互（P1，由各 `rule_*_hit` 聚合）：

4. `any_rule_hit`
5. `rule_hit_count`

## 5. Typology / 金额形态代理（P0 + P1，features-v2）

常数来自 `configs/feature_engineering.yaml`（版本写入 run manifest）。

### 5.1 P0 — 地理 / 支付 / Structuring / Smurfing / 相对金额（11）

1. `is_high_risk_sender_location`
2. `is_high_risk_receiver_location`
3. `is_high_risk_corridor`
4. `is_cash_like_payment`
5. `is_cross_border_payment_type`
6. `is_round_amount`
7. `is_just_below_reporting_threshold`
8. `sender_small_amount_unique_receivers_7d`
9. `receiver_small_amount_unique_senders_7d`
10. `amount_to_sender_outgoing_mean_ratio_30d`
11. `amount_zscore_vs_sender_outgoing_30d`

### 5.2 P1 — Deposit-Send / 行为变化 / 时间周期（10）

1. `seconds_since_last_outgoing`（sender 账户）
2. `seconds_since_last_incoming`（sender 账户作为收款方的上次入账）
3. `cash_in_then_out_within_window`
4. `sender_outgoing_count_1d_over_30d`
5. `receiver_incoming_count_1d_over_30d`
6. `sender_outgoing_unique_counterparties_1d_over_30d`
7. `receiver_incoming_unique_counterparties_1d_over_30d`
8. `hour_of_day`
9. `day_of_week`
10. `is_weekend`

另见第 4 节规则交互 `any_rule_hit` / `rule_hit_count`。

## 6. 总数校验

- 非规则 PIT 特征：`5 + 60 + 7 + 11 (P0) + 10 (P1 行为/时间) = 93`
- 规则命中特征：`3`
- 规则交互：`2`
- 合计模型特征：**98**

`_feature_build_summary.json` 中更大的 `feature_column_count` 会包含标签与基础字段列，因此会高于 98，这属于预期。

## 7. 全量重建与表格基线重训（sidecar）

```bash
# 1) 写出 FE sidecar，勿覆盖正式 pit_features
$PY -m aml_evidence_graph.features.build \
  --input artifacts/prepared_transactions \
  --output artifacts/pit_features_fe_v2 \
  --rules configs/rules/default.yaml \
  --feature-registry configs/features.yaml \
  --feature-engineering-config configs/feature_engineering.yaml \
  --overwrite

# 2) 仅重训表格基线对比 PR-AUC / 告警削减
$PY -m aml_evidence_graph.training.table_baseline \
  --features artifacts/pit_features_fe_v2 \
  --output artifacts/table_baseline_fe_v2
```

烟雾子集可用 `--max-dates N` / `--start-date` / `--end-date`。
