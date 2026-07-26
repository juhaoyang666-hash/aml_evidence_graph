# 关联风险案件工作流（后评分聚合）

本项目的「团伙 / 关联」叙事来自**冻结交易分数之后的无标签聚合**，不是另训一套
大规模异构 GNN。对齐蚂蚁/京东「关联风险」面试话术时，先讲清这一边界。

## 流水线

```text
交易级分数（主线：catboost+GAT 校准概率）
    → 账户风险聚合（account_risk）
    → 严格时间递增资金路径（funds_paths）
    → 关联子图案件候选（case_views, status=investigation_candidate）
```

实现：`src/aml_evidence_graph/aggregation/views.py`  
命令见 [FULL_RUN_AFTER_PIT.md](FULL_RUN_AFTER_PIT.md) §6。

## 主线产物

| 项 | 值 |
|---|---|
| 目录 | `artifacts/test_investigation_views_gat` |
| run_id | `20260726T090826Z-b9ef382e9f` |
| as_of | `2023-08-23T23:59:59Z`（30 日窗） |
| 账户数 | 307,103 |
| 资金路径 | 135 |
| 案件候选 | 212 |

相对早期三路融合视图（266 路径 / 228 案件），GAT 融合校准后越过风险阈值的账户更少，
案件集合更紧——**同一账户宇宙、更尖的告警集**。

## 与规则的配合

规则命中（`rule_*_hit` / Evidence Package）提供可审计触发原因；图分数与路径提供
关联上下文。人工复核只写审计，不在线回写模型。

## 明确未做（可作扩展点）

- 全量异构多关系 GNN / RGCN 关系类型细分训练（RGCN 架构对比是同构边设定下的 bake-off）
- 社区检测 / Louvain 等子图算法专线
- 在线图谱特征服务

面试一句：**先用可审计的后评分聚合把关联案件工作流跑通；异构图与社区检测是增强项，不是当前主线声称。**
