# 受控 Agent 调查评估

> 评测日期：2026-07-28；数据均为公开结构上的虚构 Mock，不含真实账户或交易明细。

## 结论

受控单 Agent 的 60 案回归基线已通过。工具选择、参数合法、事实快照一致和故障恢复四项均为
`1.0`；本机端到端 p50/p95 约为 `18.4/111.5 ms`。该结果只证明当前确定性工作流在项目内
合成用例上的行为，不代表真实生产准确率、合规验收或服务 SLA。

| 类别 | 案例数 | 主要覆盖 |
|---|---:|---|
| 路由 | 12 | 特征、图、低证据、缺引用等输入组合 |
| 工具 | 24 | 正常边界、跨告警、超限参数、SQL/路径/URL 注入、缺字段 |
| 人工审核 | 12 | approve/edit/reject、非法动作、edit 缺备注 |
| 恢复 | 12 | 一次性/持续检索故障，内存与 SQLite checkpoint |

## 指标

| 指标 | 当前结果 | 口径 |
|---|---:|---|
| Case 通过率 | 1.000 | 实际结果符合逐案期望 |
| 工具选择准确率 | 1.000 | 路由出的工具序列与期望一致 |
| 参数合法率 | 1.000 | 合法输入通过、非法输入拒绝均算正确 |
| 事实一致率 | 1.000 | 报告 `fact_snapshot` 与输入 EvidencePackage 相同 |
| 恢复成功率 | 1.000 | 故障降级后仍暂停审核并按决策完成 |
| p50 / p95 | 约 18.4 / 111.5 ms | 当前 Windows 本机、确定性无 LLM 路径；含 SQLite 重开恢复案 |
| token/cost 覆盖 | 0.000 | 未调用外部 LLM，不虚构 token 或合同价格 |

持续检索失败时，工具审计记录 `TimeoutError`，报告继续以确定性模板生成，并在
`uncertainty_notes` 声明 Typology 检索不可用。审批前状态仍是
`draft_requires_human_review`，不会产生已提交或已处置结论。

每个工作流案例保存 `node_timeline`，包含节点、UTC 时间、耗时、状态和状态变化；工具事件
另外保存工具名、参数名、输出键、耗时、状态和错误码。完整逐案结果写入本地忽略的
`artifacts/agent_evaluation/metrics.json`，Bad Case 同文件输出。

## 独立审计存储

受控 API 已支持与 LangGraph checkpoint 分离的 SQLite 追加式审计库。配置
`AML_AGENT_AUDIT_PATH` 后，启动与人工复核恢复都会把当前可见事件投影到独立表；同一
`event_id` 使用 `INSERT OR IGNORE`，因此重放同一 checkpoint 不会重复写入。关闭并重新
创建存储对象后仍能按 `thread_id` 读取记录。

审计表只允许工具名、参数名、输出键、节点状态变化、错误码、耗时、复核动作、复核人引用和
“是否存在备注”。Schema 不提供证据正文、特征值、图边内容或备注正文列。该实现是本地持久化
原型，不等于生产 WORM 审计系统；正式部署仍需组织确定访问控制、集中备份、法务保留期限、
到期销毁与审计库自身监控。应用运行时不自动删除审计记录。

## 复现

```powershell
$env:PYTHONPATH = "src"
D:\Miniconda3\envs\aml-evidence\python.exe scripts/evaluate_agent_golden.py --overwrite
```

权威输入为 `golden/agent_cases_v2.json`。后续需要补充 API 层并发恢复幂等压测和生产审计
适配器，并在启用外部 LLM 后单独报告 provider token、配置价格下的成本和延迟，不能与本
基线混算。
