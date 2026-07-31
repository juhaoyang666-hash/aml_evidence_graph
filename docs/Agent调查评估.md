# 受控 Agent 调查评估

> 评测日期：2026-07-28；多 worker 补充验证：2026-07-31；租约续期与故障注入：2026-07-31。
> 数据均为公开结构上的虚构 Mock，不含真实账户或交易明细。

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

## 审批幂等与并发探针

`POST /v1/controlled-investigations/{thread_id}/review` 支持 `Idempotency-Key`。默认单进程
模式使用内存细粒度锁；配置 `AML_AGENT_COORDINATION_PATH` 后，多个 worker 通过独立 SQLite
租约表串行化同一 `thread_id` 的读取与变更。第一次请求执行恢复，随后相同键且
相同正文的请求返回完成状态并标记 `idempotent_replay=true`。同一键配不同审批正文返回
`409`；不带键的最终态重复审批仍返回 `409`。幂等键保存在 SQLite checkpoint 中，所以关闭
连接、重建应用后仍可识别回放。

2026-07-28 本机 HTTP 探针使用 50 个 Mock 案件线程、案件并发 5，每案并发提交 8 个相同
审批，共 400 个 review 请求。50 个周期全部成功、错误率 `0%`，350 个请求标记为回放；审计库
最终恰好有 50 条 `human_review_decision` 和 50 条 `finalize`。周期吞吐为
`8.99 cycle/s`，p50/p95/p99 为 `526.97/668.67/849.61 ms`。该延迟包含每案 8 个审批 HTTP
请求，不能与普通单审批周期直接比较。

2026-07-31 又在 Windows 本机启动两个真实 Uvicorn worker，并配置四个彼此分离的共享 SQLite：
Evidence Package、LangGraph checkpoint、审计事件和线程租约。跨连接证据读取 `40/40` 为
`200`；20 个调查线程各并发提交 8 次同键审批，共 160 个 review 请求，结果为 20 次实际执行、
140 次幂等回放、0 错误。共享审计库恰有 20 条 `human_review_decision` 和 20 条 `finalize`，
结束后活动租约为 0。七项自动校验全部通过。

这证明的是“本机两 worker + 共享 SQLite + 同键请求”的工程 exactly-once 行为，不是任意分布式
数据库、网络分区或生产 SLA。SQLite 适合求职 Demo 和单机多进程，不适合高并发多主部署。
Evidence Package 在本地数据库中保存完整 JSON，正式环境还需文件权限、磁盘加密、备份和
保留策略。

## 租约续期与故障注入

2026-07-31 补齐了此前记为“尚未验证”的两项：租约续期和故障注入。

此前 `SQLiteThreadLockRegistry.hold()` 只在进入时 claim 一次租约，临界区内既不续期也不检查
自己是否仍然持有。因此只要单次请求超过 `lease_seconds`，`_claim()` 中的
`DELETE ... WHERE expires_at <= ?` 就会把该租约回收，**第二个 worker 可以合法地拿到同一个
`thread_id` 并与第一个 worker 同时进入临界区**。原设计靠“租约默认 10 分钟必须高于单次工作流
最大运行时间”这一约定回避该问题，但该约定既没有强制，也没有回归覆盖；而
[服务性能基准.md](服务性能基准.md) 记录的完整融合评分 p95 已达 `34.85 s`，属于同一量级的
运行时间，不宜再依赖约定。

现在的实现：

- **续期**：`hold()` 启动守护心跳线程，每 `renew_interval_seconds`（默认 `lease_seconds/3`）
  执行一次带 `owner_token` 与 `expires_at > now` 双条件的 `UPDATE`。已过期的租约不会被复活，
  避免与正在接管的 worker 抢同一行。
- **Fencing**：续期失败或最终 `DELETE` 影响行数为 0，都判定为租约已丢失，`hold()` 抛出
  `LeaseLostError`。释放语句始终带 `owner_token` 条件，因此过期持有者不会误删接管者的租约。
- **API 语义**：`LeaseLostError` 映射为 `503`，表示该变更未被证明是独占的、可安全重试，
  而不是不透明的 `500`。若临界区自身已抛异常，则原异常优先，不被租约错误掩盖。

故障注入回归位于 `tests/test_thread_lease.py`（9 项）：

| 用例 | 注入的故障 | 期望 |
|---|---|---|
| 续期保活 | 持有时长 `0.9 s` = 3 倍租约周期 | 竞争者 `TimeoutError`，独占维持 |
| 崩溃接管 | 直接 claim 后不续期、不释放 | 租约到期后另一 registry 成功接管 |
| 租约被夺 | 临界区内改写 `owner_token` | 抛 `LeaseLostError`，且接管者租约存活 |
| 参数校验 | 续期间隔 ≥ 租约或非正数 | 构造期 `ValueError` |
| API 映射 | 持锁即丢失 | HTTP `503` 而非 `500` |

同一场景在旧实现下已实测复现出“两个 worker 同时在临界区内”，修复后竞争者按预期超时，
因此这些用例是真回归而不是同义反复。连接泄漏也一并修复：`_claim` / `_release` / 建表原先
只依赖 `with sqlite3.connect(...)`（提交但不关闭），在心跳循环下会持续累积连接。

边界不变：这仍是单机多进程结论。跨主机数据库、网络分区和真实进程 `SIGKILL` 未验证；
心跳线程与 API 进程同生共死，进程被杀后靠租约自然过期恢复，最坏停顿为一个 `lease_seconds`。

## 复现

```powershell
$env:PYTHONPATH = "src"
D:\Miniconda3\envs\aml-evidence\python.exe scripts/evaluate_agent_golden.py --overwrite
D:\Miniconda3\envs\aml-evidence\python.exe scripts/benchmark_controlled_agent.py `
  --requests 50 --concurrency 5 --duplicate-reviews 8 `
  --output artifacts/serving_benchmark_agent_idempotency
& scripts/run_multiworker_shared_probe.ps1
D:\Miniconda3\envs\aml-evidence\python.exe scripts/validate_multiworker_shared_probe.py
D:\Miniconda3\envs\aml-evidence\python.exe -m pytest tests/test_thread_lease.py -q
```

权威输入为 `golden/agent_cases_v2.json`。多 worker 聚合校验位于
`artifacts/multiworker_shared_v1/validation.json`。后续生产扩展仍需集中审计适配器，并在启用
外部 LLM 后单独报告 provider token、配置价格下的成本和延迟，不能与本基线混算。
