# 规划复核与修订（2026-07-25）

本文基于 `artifacts/table_baseline`（`run_id=20260725T073033Z-bbf71e1c02`）与
`artifacts/graphsage` 的全量结果，复核原规划并给出修订项。原规划的 **PIT / 时间外切分 /
无标签图推理** 框架保留；修订集中在：规则基线 KPI、交付物取舍、以及面向岗位的表达口径。

## 一、结论摘要

| 维度 | 原规划 | 复核结论 |
|---|---|---|
| PIT / 时间外切分 / 无标签图推理 | 严格 | **保留**，这是本项目最强的差异点 |
| 指标口径（PR-AUC 为主 + 告警预算） | 合理 | **保留**；对外数字须带数据集、run_id、切分协议 |
| `graph_stats_catboost` | 平行候选 | **保留为候选基线**；主线报告以 `catboost` 与 `graphsage` 为主 |
| 规则基线与告警削减 KPI | 核心 KPI | 初版空转，已用训练期分位数修好，见 §2 |
| GAT/RGCN/PNA 全量对比 | 成熟版目标 | **已完成**；GAT 升为主线图模型，主线融合改为 `catboost + GAT`，见 §4 / RESULTS |
| 100 例人工 Golden | 验收项 | **缩减到 30 例**，见 §4 |

## 二、告警削减 KPI：规则基线须可运行

首轮 `artifacts/table_baseline/metrics.json` 的 `alert_reduction_vs_rules` 曾为 `{}`，
`test_metrics` 中无 `rules` 项。根因链：

1. `configs/rules/default.yaml` 中部分规则的 `parameters.threshold` 为 `null`，`status: draft`；
2. `rules/engine.py` 对 threshold 为空的规则不生成命中列；
3. PIT 产物中 `rule_*_hit` 列数为 0，`_rule_evidence` 未产生；
4. `table_baseline.rule_baseline_scores()` 因此返回 `None`，
   `compare_alert_volume_at_fixed_recall` 从未被调用。

「固定召回下相对规则基线的告警削减率」是原规划强调的核心 KPI。修订后把规则基线补齐为
可运行的**三条阈值规则**，阈值只在**训练期**分位数上确定，并写入 `backtest_summary`：

| 规则 | 特征 | 定阈方式 |
|---|---|---|
| R-CROSS-BORDER-FREQUENCY | `sender_outgoing_cross_border_count_14d` | 训练期 99.5 分位 |
| R-STRUCTURING-BURST | `sender_outgoing_count_1d` | 训练期 99.5 分位 |
| R-FAN-IN | `receiver_incoming_unique_counterparties_7d` | 训练期 99 分位 |

定阈只读训练期分区，禁止使用验证/测试期。完成后重跑 `table_baseline`，
`alert_reduction_vs_rules` 应给出「同召回下告警量下降 X%」。细节见
[RULE_BASELINE.md](RULE_BASELINE.md)。

## 三、指标表达口径（面向简历与面试）

现有 `metrics.json` 已含 PR-AUC、ROC-AUC、KS、Brier、ECE、告警预算、固定 FPR 召回、
月度/模式/新账户/支付方式/地域/币种切片、漂移、SHAP、吞吐与内存。这套口径保留。

表达纪律：

1. 任何对外数字必须带三要素：**数据集为公开合成 SAML-D**、**run_id**、**切分协议**。
2. 不要单独说 ROC-AUC（1‰ 正例率下 ROC 易虚高）。主指标是 PR-AUC 与固定预算下的
   Precision/Recall。
3. 新账户切片若正例极少（如 `sender_new` 仅 1 个正例），须标注样本量不足，不得当作冷启动结论。
4. 引用示例（可直接用于简历）：
   > 在公开合成基准 SAML-D（约 950 万笔、正例率约 0.1%）上，PIT 特征 + CatBoost 在
   > 时间外测试期取得 PR-AUC 0.809；0.1% 告警预算下 Precision 0.858 / Recall 0.737。
   > 历史图、无标签推理的边分类中 GAT 最优（PR-AUC 0.948）；`catboost + GAT` 的 OOF 堆叠
   > 融合（验证期校准 + 锁定阈值）测试期 PR-AUC 0.918。相对规则基线在同召回点报告告警削减。
   > 说明：该融合优于任一表格模型与原 GraphSAGE 融合，但不优于单独 GAT，二者需同时披露。

## 四、优先级重排

原规划把「GAT/RGCN/PNA 全量对比」和「100 例人工 Golden」列为验收项。以当前边际收益判断，
这两项对目标岗位的性价比低于「规则基线 KPI + 端到端可复现」。

### P0（必须完成，构成项目主结论）
1. ~~修好规则基线阈值~~（已完成：v2026.2 三条规则 + `rule_*_hit` 已补写入 PIT）
   → 待 `table_oof` 结束后重跑 `table_baseline` → 拿到告警削减率
2. 完成在跑的 `table_oof` → `graph_oof` → `fusion` → `fusion_test`
3. 用 `fusion_test` 冻结分数跑 `aggregation.views`，产出账户风险/资金路径/案件视图
4. 回填 `MODEL_CARD.md` 与 `IMPLEMENTATION_STATUS.md`

### P1（显著提升可讲性）
5. 融合组合优先报告 `catboost,graphsage`，给出「融合是否真的优于单模型」的诚实结论
6. 一页 `docs/RESULTS.md`：主表 + 切片 + 复现命令
7. Golden 案例缩减为 **30 例**（覆盖主要 typology），够验证 SAR 草稿的事实约束

### P2（有余力再做）
8. ~~GAT / RGCN / PNA 对比（只在同协议下比 PR-AUC 与吞吐）~~ — **已完成**
   （测试 PR-AUC：GAT 0.9483 > RGCN 0.9031 > GraphSAGE 0.8777 > PNA 0.7049；见 RESULTS）
9. 阈值随时间漂移的重校准演练

### 暂缓 / 已修订
- 100 例人工 Golden：改为 30 例；**v1 已用户授权 agent 裁定**（见 `golden/adjudication_v1.json`）
- 多架构全量对比：由「验收项」改为「可选扩展」；**现已跑完并回填 RESULTS**

### LLM 部分单独成文

原规划的 LLM 内容散落在多份文档中。已补 [LLM_PLAN.md](LLM_PLAN.md)，要点：

- 有界 LangGraph、字段最小化、引用白名单 + 数值正则事实校验、失败降级路径已实现
- Golden 评测应补幻觉拦截率、无证据拒答率、端到端延迟 p50/p95
- 对抗测试（prompt 注入等）列为 P2

## 五、PIT 构建的性能问题与提速路径

全量 PIT 约需 8 小时（321 分区 / 9,504,852 行，单进程）。`features/pit.py` 的
`RollingHistory.summary()` 对每笔交易都重新遍历该账户过去 30 天的事件，
高频账户上成本偏高。

### 已落地：规则变更不再触发重建

阈值规则是已有列的纯函数。`scripts/backfill_rule_hits.py` 直接补写
`rule_*_hit` 与 `_rule_evidence`：

| 方式 | 耗时 |
|---|---|
| 全量 PIT 重建 | 约 8 小时 |
| 规则后处理补写（干跑） | **约 47 秒** |
| 规则后处理补写（写入） | **约 2 分钟** |

脚本在规则引用的特征缺失时会拒绝执行，避免掩盖真正需要重放的情况。

### 若将来确需重建（按性价比）

1. **增量式窗口聚合**：维护窗口 running 累加器，出窗减法；预期数量级提升且不改变语义。
2. **按账户哈希分桶并行**，再串行合流；图统计仍需注意因果顺序。
3. **DuckDB / Polars 窗口查询**，改动面较大，需重验同秒隔离语义。
4. **降低重建频率**：新特征若是已有列的函数，走后处理；只有新历史窗口才重放。

短期建议：不为提速而重构 PIT；增量聚合列为 P2。

## 六、文档与环境修正

1. 运行环境已迁移到 Linux + `/data1/yangjuhao/envs/risk` + RTX 3090；见
   [ENVIRONMENT.md](ENVIRONMENT.md)。
2. GraphSAGE 已支持多 GPU（上限 4 卡，`--max-gpus`）。
3. 规则基线与补写流程见 [RULE_BASELINE.md](RULE_BASELINE.md)。

## 七、当前进度对照

| 步骤 | 状态 |
|---|---|
| SAML-D 全量转换（9,504,852 行 / 321 天） | 完成 |
| 全量 PIT（`run_id=20260724T000855Z-e3d573661f`） | 完成；经规则补写后为 90 列 |
| 规则基线 v2026.2 + `rule_*_hit` 补写 | 完成（836,487 命中；test 告警率约 8.79% / 召回约 0.215） |
| `table_baseline` / `table_baseline_rules` | 完成；CatBoost 测试 PR-AUC 0.8092；见 [RESULTS.md](RESULTS.md) |
| `graphsage`（GPU） | 完成，test PR-AUC 0.8777 |
| `table_oof` / `graph_oof` | 完成 |
| `fusion` / `fusion_test`（含 graph_stats 三路对照） | 完成 |
| 双路融合 `catboost+graphsage` | 完成 → `artifacts/fusion_cb_gs` / `fusion_test_cb_gs`；PR-AUC 0.8973 |
| 调查视图 | 完成 → `artifacts/test_investigation_views` |
| RESULTS.md / MODEL_CARD / Golden 30 | 完成；Golden v1 已用户授权 agent 裁定（`golden/adjudication_v1.json`） |
| GAT/RGCN/PNA 全量对比 | **完成**；GAT 0.9483 / RGCN 0.9031 / GraphSAGE 0.8777 / PNA 0.7049（见 RESULTS） |
