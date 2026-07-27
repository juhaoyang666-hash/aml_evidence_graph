# SAML-D 特征工程调研与本项目改进建议

> 对照公开合成数据集 SAML-D（Oztas et al., IEEE ICEBE 2023）上社区常见做法，
> 与本仓库当前 PIT 特征（见 [PIT_FEATURE_LIST.md](PIT_FEATURE_LIST.md)）的差距与可落地建议。  
> **口径**：建议仅在严格 PIT / 时间外切分下试验；不自动改变主线数字。

## 1. 数据集本身在“暗示”什么特征

官方论文与数据说明（[Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)、
[IEEE ICEBE 2023](https://doi.org/10.1109/ICEBE59045.2023.00028)、
[作者 GitHub](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D)）：

| 设计点 | 含义 |
|---|---|
| 12 个原始字段 | Date/Time、Sender/Receiver、Amount、Payment type、Locations、Currencies、Is_laundering、Type |
| **17 可疑 + 11 正常 typologies** | Structuring / Smurfing、Cash Withdrawal、Deposit-Send、Layered Fan-In/Out、Cycle、Behavioural Change… |
| **15 种图结构** | fan-in/out、layering、bipartite、scatter-gather 等；表格聚合难以完全表达多跳结构 |
| 高风险地理 / 支付 | 文中点名 Mexico / Turkey / Morocco / UAE；Cash、Cross-border 等支付类型 |

论文实验结论：原始字段 + 简单编码的经典 ML 在 SAML-D 上比在 AMLSim 更难；作者明确建议
**做特征工程 + 更强模型**。本项目的 PIT + CatBoost + GAT 正是在回应这一点。

## 2. 别人怎么做特征（公开仓库 / 流水线）

### 2.1 [nashcastillo/aml-feature-engineering](https://github.com/nashcastillo/aml-feature-engineering)

- 样本：约 80 万行子集；时间切分 + held-out 定阈。
- 特征侧重：
  - 账户频次（all-time / train 内）
  - **高风险国家清单**（FATF/OFAC/EU + 作者点名四国）
  - **高风险支付类型**（Cash Deposit / Withdrawal / Cross-border）
  - **Smurfing score / Fan-out score**（小额 + 多对端）
  - NetworkX：**PageRank + 度**（图只在 train 上建）
  - 账户 **Autoencoder embedding**（sender/receiver 各 8 维）→ 共约 34 维
- 模型：LightGBM + 校准；规则基线对比告警量。

### 2.2 [Vamshi-27072001/aml-detection-saml-d](https://github.com/Vamshi-27072001/aml-detection-saml-d)

| 组 | 代表特征 |
|---|---|
| Amount | `log_amount`, `amount_zscore_sender`, `amount_to_sender_mean_ratio`, `is_round_number`, `is_just_below_threshold` |
| Velocity | `txn_count_1h/24h`, `txn_volume_24h`, `unique_receivers_24h` |
| Geography | `high_risk_sender/receiver/corridor`, `cross_border` |
| Currency | `currency_mismatch` |
| Graph | out/in degree, **betweenness**, hub 标志 |
| Anomaly | IsolationForest score |
| Categorical | **Bayesian target encoding**（Payment_type / 币种 / 地区） |

高风险司法辖区示例：Mexico, Turkey, UAE, Morocco, Nigeria, Iran, Myanmar。

### 2.3 数据集作者 starter / 论文基线

- 多为 one-hot / label encode + 标准化 + 经典分类器；**几乎不做 PIT 窗口**。
- AUC 约 0.81 量级、高 FPR——与本项目「规则告警量巨大、模型大幅削减」叙事一致。

### 2.4 横向归纳：社区高频特征族

1. **速度 / 窗口聚合**（1h–30d）  
2. **金额形态**（取对数、相对账户均值、整数金额、略低于阈值）  
3. **地理 / 走廊 / 高风险清单**  
4. **币种错配**  
5. **typology 代理分**（smurfing / fan-out / structuring）  
6. **图统计或中心性**（度、PageRank、betweenness）  
7. **账户表征**（AE / 无监督分数）  
8. **类别编码**（target encode 或 GBDT 原生类别）

## 3. 本项目现状（相对优势）

已有且协议更严（见 `features/pit.py`、`graph_stats.py`、`PIT_FEATURE_LIST.md`）：

| 能力 | 本项目 | 多数公开 notebook |
|---|---|---|
| 多窗口速度（1h/1d/7d/14d/30d） | ✅ 三组前缀共 60 维 | 常只有 1h/24h |
| 同秒批次隔离 PIT | ✅ | 少见严格实现 |
| 关系对 (sender,receiver) 窗口 | ✅ | 少见 |
| 因果图度 / 先验边计数 / 互惠 | ✅ | 常见度，少见因果边计数 |
| 图消息传递（GAT） | ✅ 主线 PR-AUC 0.948 | 多为 NetworkX 手工图特征 |
| 固定时间外切分 + OOF 融合 | ✅ | 常 stratified 或弱时间切分 |

**结论**：你们在「泄漏控制 + 窗口完备性 + 图学习」上已领先多数开源 SAML-D 流水线；社区多出的是
**业务语义代理特征**和**金额/地理形态特征**，而不是更复杂的滚动窗口。

`CATBOOST_GAP_DIAGNOSIS.md` 也表明：CatBoost 已吃到 `relationship_*`，与 GAT 的差距主要来自
**多跳结构**，不宜指望再堆窗口统计就追上 GAT。

## 4. 改进建议（按优先级）

### P0 — 低成本、对齐数据集生成逻辑（优先试）

这些与论文 typologies / 高风险设计直接对齐，且可用已有 PIT 状态或当前行算出，**不必重建图模型**。

1. **高风险地理 / 走廊**（作者点名四国至少先做）  
   - `is_high_risk_sender_location` / `is_high_risk_receiver_location`  
   - `is_high_risk_corridor`（sender∈HR 或 receiver∈HR，或双方）  
   - 清单版本写入 `configs/` + registry，禁止 silent 硬编码。

2. **支付类型风险旗标**  
   - `is_cash_like_payment`、`is_cross_border_payment_type`（与 `Payment_type` 枚举对齐）  
   - CatBoost importance 里 `payment_type` 已很靠前；显式风险旗标利于规则与 SHAP 叙事。

3. **Structuring / 阈值邻域**  
   - `is_round_amount`、`is_just_below_reporting_threshold`（阈值仅用 **train 分位数或业务常数**，与规则 v2026.2 一致）  
   - 可与现有 `R-HIGH-AMOUNT-001` 互补（规则看高额，特征看「刚好低于」）。

4. **小额扇入 / 扇出代理（Smurfing / Fan-out score）**  
   - 在已有 `unique_counterparties_*` + `same_currency_amount_sum_*` 上衍生：  
     - 小额定义：`amount < train_q_p` 或固定业务阈值  
     - `receiver_small_amount_unique_senders_7d`、`sender_small_amount_unique_receivers_7d`  
   - 对齐社区 smurfing/fan-out score，且严格 PIT。

5. **金额相对历史**（社区常用，你们目前只有绝对 sum）  
   - `amount / (1 + sender_outgoing_same_currency_amount_sum_30d / max(count,1))`  
   - 或 `amount_zscore_vs_sender_30d`（需存均值/方差于 RollingHistory）。

### P1 — 中等成本、补行为类 typologies

6. **Deposit-Send / 快速进出**  
   - `seconds_since_last_incoming`、`seconds_since_last_outgoing`  
   - `cash_in_then_out_within_Xh` 类布尔（需识别 cash 支付类型）。

7. **Behavioural Change 代理**  
   - 短窗 vs 长窗速度比：`count_1d / (1+count_30d)`  
   - 新对端占比：`unique_counterparties_1d / (1+unique_counterparties_30d)`  
   - 对齐论文 Behavioural Change 1/2。

8. **时间周期**  
   - `hour_of_day`、`day_of_week`、`is_weekend`（当前行时间戳即可）。

9. **规则特征交互**  
   - 已有 3 个 `rule_*_hit`；可加 `any_rule_hit`、命中条数，供表格模型学习规则互补区。

### P2 — 可选 / 性价比低（主线已有更好替代）

10. **PageRank / betweenness / hub**  
    - 社区常用，但本项目 **GAT 已覆盖结构排序**；再做全局中心性要小心泄漏与算力。  
    - 若做：仅 train 图快照、按日 PIT 更新，且与 GAT 对照 PR-AUC，预期增益有限。

11. **账户 Autoencoder / IsolationForest 分数当特征**  
    - 可作为 ablation；拟合必须只在 train，推断用冻结编码器。  
    - 不如优先 P0 语义特征清晰、可审计。

12. **Target Encoding 类别**  
    - CatBoost 已原生处理类别；OOF target encode 收益通常不大，还增加泄漏面。

13. **继续加长窗口堆 count_***  
    - 边际收益低；gap 诊断已指向结构而非「再多一个 30d count」。

## 5. 建议实验协议（避免污染主线）

1. 在 `pit.py` / registry 增加候选特征 → 写出 `artifacts/pit_features_fe_v2` 或增量列脚本（类似 `backfill_rule_hits.py`）。  
2. 仅重训 `table_baseline_*`（hash 负采样），对比 test PR-AUC / 0.1% 预算 / 告警削减。  
3. **不要**指望关掉 GAT；目标是缩小 CatBoost↔GAT 的部分缺口、增强可解释性与规则互补。  
4. 若 CatBoost 提升 < ~0.01 PR-AUC：写入负向消融，保持当前主线不变（与 hard-negative 结论同一纪律）。

## 6. 一句话对照

| 维度 | 社区 SAML-D 流水线 | 本项目 | 建议 |
|---|---|---|---|
| 速度窗口 | 浅 | **更深更严** | 保持 |
| Typology / 金额形态 / 高风险地理 | **强** | 弱 | **P0 补齐** |
| 手工图中心性 | 常见 | 弱（有因果度） | P2，可选 |
| 图学习 | 少 | **GAT 主线** | 保持 |
| 防泄漏 | 参差 | **强** | 新特征必须沿用 PIT |

## 6.1 FE v2 sidecar 实验回填（2026-07-27）

- 本调研建议中的 P0/P1 特征已在 `fe_v2` sidecar 链路完成一次端到端实现与评估
  （Polars + PIT sidecar + table CatBoost）。
- 指标唯一真值来源：`artifacts/table_baseline_fe_v2/metrics.json`。
- 运行状态时间来源：`artifacts/logs/fe_v2_pipeline_status.json`（仅状态/时间）。
- 结果（sidecar 口径）：validation PR-AUC **0.8660898669**、validation ROC-AUC **0.9982223689**、
  test PR-AUC **0.8754139061**、test ROC-AUC **0.9984022445**。
- 相对主线 CatBoost test PR-AUC 0.8092 的差值为 **+0.0662**（四舍五入）。
- 边界：该结果仅作特征路线 sidecar 补充，sidecar 不覆盖主线；smoke 路径只用于链路验证；
  后续仍需复核是否存在过拟合或数据口径差异。
