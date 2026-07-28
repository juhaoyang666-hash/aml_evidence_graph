# 2026 风控算法实习求职导向：AML Evidence Graph 项目改进建议

> 调研日期：2026-07-28
>
> 目标读者：项目作者、风控算法/反欺诈算法/金融 AI Agent 岗位的简历筛选者与技术面试官
>
> 结论边界：招聘样本是定向抽样，不代表全部市场；项目效果来自公开合成 SAML-D，不能外推到真实业务。

## 技术摘要

**本项目已经具备一条有竞争力的风控算法主线，不建议继续无目的堆模型。** 当前最强的
求职叙事是：在约 950 万笔公开合成交易上构建严格 Point-in-Time（PIT）特征和历史图，
以 CatBoost 与 GAT 完成交易级极不平衡风险排序，使用时间外测试、告警预算指标、漂移与
调查证据验证，并把 LLM 限制在证据约束的人工调查辅助环节。

对照 2026 年近期岗位，项目对传统风控算法岗和图风控岗的覆盖已经较强；主要差距集中在：

1. **大数据工程证据仍偏“演示级”**：有 Polars、DuckDB 和 ODPS SQL 模板，但没有可展示的
   Spark/Hive 特征作业及规模、耗时、资源对比。
2. **LLM 工作流仍偏线性**：虽然使用 LangGraph，但当前图固定执行检索、校验、注释、成稿，
   只有一次 BM25 检索，不足以证明岗位所说的动态工具调用、状态持久化和人工中断恢复。
3. **RAG 评测不足**：Typology 库只有小规模 YAML/BM25 路径，已有生成侧 Golden，但缺少
   检索侧 Recall@k、MRR、nDCG、hard-negative 和 BM25/向量/混合检索对照。
4. **工程结果缺少一页式可验证入口**：已有 run manifest、测试、FastAPI 和 Docker，但还需要
   一条命令生成“数据版本—特征版本—模型—指标—延迟—Prompt/知识库版本”的统一求职报告。
5. **FE v2 实验尚未闭环**：截至调研时，FE v2 CatBoost test PR-AUC 为 0.8754，GAT→OOF→
   融合→Bootstrap→调查视图仍在运行，不能提前写入简历。

建议优先完成 P0 的“实验闭环、检索评测、真实 HITL、性能报告”四项。完成后，这个项目既能投
传统风控算法实习，也能投图算法和大模型风控应用岗；QLoRA/SFT、多 Agent、Neo4j、Kafka
都不是当前必需项。

## 一、招聘样本显示的能力结构

### 1.1 调研口径

本次选取近期可访问的风控、反欺诈、通用算法和金融大模型实习页面，优先使用公司招聘页或
带完整职位描述的招聘页面。样本用于识别反复出现的技能组合，不用于推断岗位总量或薪资。

| 岗位样本 | 页面时效/状态 | 反复出现的要求 |
|---|---|---|
| [美图：风控算法暑期实习](https://cn.linkedin.com/jobs/view/%E9%A3%8E%E6%8E%A7%E7%AE%97%E6%B3%95%E5%AE%9E%E4%B9%A0%E7%94%9F-summer-intern-risk-control-algorithm-at-meitu-inc-4404539865) | 2026 暑期，页面显示已停止接收 | SQL、Python、特征工程、XGBoost/树模型、实时监控、拦截率与误判率、跨团队落地 |
| [Shopee：信贷风控算法实习](https://www.nowcoder.com/feed/main/detail/feb72d6842484affa365c8df755bad97) | 面向 2027 届，2026-06 发布 | Python/SQL、Hadoop/Spark、LightGBM/XGBoost、PyTorch、序列模型、GNN、多目标学习、Agentic AI |
| [快手：信贷风控算法实习](https://www.deizao.net/m/index/shixixq/jobid/101183) | 招聘期至 2026-12-30 | 标签定义、特征工程、模型训练评估部署、Hive SQL/Python/Spark、风险与模型监控、异动归因 |
| [百度：算法实习生 J80417](https://talent.baidu.com/jobs/detail/INTERN/90fde911-090d-4469-90b1-205f9a6d372f) | 2026-07-21 发布 | Python/Shell/MySQL、PyTorch 等框架、Hive/Spark、数据监控、半监督/多目标学习 |
| [携程：AI Agent 算法实习（风控）](https://www.shixiseng.com/intern/inn_oveszt5vsfer) | 2026-07 刷新，截止 2026-08-08 | 风控 SOP、Prompt/RAG/Function Calling、异构数据、Human-in-the-loop、可解释/可回溯/可撤回、Bad Case |
| [国金证券：AI Agent 研发暑期实习](https://cn.linkedin.com/jobs/view/%E5%85%AC%E5%8F%B8%E7%9B%B4%E5%B1%9E-%E7%A7%91%E6%8A%80%E7%A0%94%E5%8F%91%E9%83%A8-ai-agent%E7%A0%94%E5%8F%91%E6%9A%91%E6%9C%9F%E5%AE%9E%E4%B9%A0%E7%94%9F-j16600-at-%E5%9B%BD%E9%87%91%E8%AF%81%E5%88%B8-4433068442) | 2026 暑期页面 | Python、Git/Code Review/单测、PyTorch/Transformer、LangGraph、RAG/向量检索/Rerank、QLoRA/PEFT、Function Calling、评测集与部署 |
| [财通证券资管：大模型应用实习](https://cn.linkedin.com/jobs/view/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E5%AE%9E%E4%B9%A0%E7%94%9F-000853-at-%E8%B4%A2%E9%80%9A%E8%AF%81%E5%88%B8%E8%B5%84%E7%AE%A1-4399840978) | 2026-06/07 可访问页面 | RAG、Agent、Benchmark、Transformer、Python、金融知识、多模态探索 |
| [神舟软件：大模型算法实习](https://www.zhaopin.com/jobdetail/CC000432740J40705401316.htm) | 2026/2027 届，页面要求 6 个月 | 预训练/微调/对齐、RAG、Text2SQL、Agent、MCP、数据清洗、PyTorch、Python/Shell |

图算法岗位的上限参照可使用[蚂蚁反欺诈/风控图算法 Research Intern](https://www.nowcoder.com/jobs/detail/237841)：
页面当前显示职位已结束，但其要求能说明专业图风控团队看重社区检测、异常检测、GNN、
大规模图处理、扎实代码和研究成果。它应作为专精方向参照，而不是“当前仍在招聘”的证据。

### 1.2 能力分层

| 层级 | 招聘侧含义 | 典型技能 | 本项目匹配度 |
|---|---|---|---|
| 基础门槛 | 能独立做一次可信建模 | Python、SQL、数据清洗、特征工程、LR/树模型、分类指标 | **强** |
| 风控业务 | 知道模型如何变成告警和策略 | 标签、极不平衡、时间切分、拦截/误判、阈值、监控、解释 | **强** |
| 算法区分度 | 能处理行为、关系和复杂风险模式 | GNN、序列、多目标、异常/社区、hard negative | **图方向强；其余有消融但结果较弱** |
| 数据工程 | 能在大规模数据上稳定复现 | Hive/Spark/ODPS、分区、增量、性能、数据质量 | **部分具备** |
| 算法工程 | 能训练、部署、回归和监控 | PyTorch、API、Docker、测试、版本、实验追踪、性能压测 | **较强，但缺统一追踪与压测报告** |
| LLM 应用 | 能构建而非只调用聊天 API | Transformer、RAG、Rerank、Agent、工具调用、评测 | **基础具备，检索与真实 Agent 证据不足** |
| 金融合规 | 输出可审计、可拒答、有人审 | 证据绑定、事实校验、HITL、权限、回溯、成本/延迟 | **设计强；持久化 HITL 尚未落地** |

## 二、当前项目的求职价值与证据边界

### 2.1 已经值得保留的主线

| 能力 | 仓库证据 | 面试价值 |
|---|---|---|
| 大规模数据 | SAML-D 约 950 万笔交易、321 个日期分区 | 不是玩具 CSV；可讨论分区、内存和批处理 |
| 防泄漏 | 固定时间外切分；窗口与图邻居仅使用交易时点前历史 | 风控项目最重要的可信度证据之一 |
| 表格模型 | 规则、LR、CatBoost，含 SHAP、校准和告警预算指标 | 对齐大多数传统风控岗 |
| 图学习 | GraphSAGE/GAT/RGCN/PNA 同协议边分类；GAT 为最强单模型 | 对齐关系图谱、团伙欺诈和图算法岗位 |
| 稀有事件评估 | PR-AUC、固定告警预算 P/R、Bootstrap、规则同召回告警削减 | 比只写 Accuracy/ROC-AUC 更贴近业务 |
| 运行稳定性 | 月度漂移、阈值过期影响、资源与吞吐记录 | 可回答“上线后怎么监控” |
| 工程化 | FastAPI、Docker、Pydantic、pytest、配置与 run manifest | 证明不仅是 Notebook |
| 调查辅助 | RiskEvidencePackage、Typology、LangGraph、事实校验、Golden | 对齐金融 LLM 的安全与审计边界 |

### 2.2 必须诚实披露的边界

- 数据是**公开合成数据**，不是银行/支付机构真实客户数据。
- 主任务是**交易边风险排序**，账户、路径和案件视图是后处理聚合，不能写成账户监督模型。
- GAT test PR-AUC 0.9483 高于 `CatBoost + GAT` 融合 0.9175；不能写成“融合达到最优”。
- `GAT→CatBoost` 两阶段特征实验 0.9656 不是纯表格模型，也不是在线无依赖蒸馏。
- LangGraph 当前是受控线性工作流，不应写成“自主多 Agent 决策系统”。
- 34 案 Golden 是项目裁定回归集，不是独立合规专家标注或生产验收。
- 没有真实线上流量，因此不能写“实时生产部署”“支撑日均多少请求”或“降低真实损失”。

## 三、差距诊断：该补什么，不该补什么

### 3.1 P0：投递前最值得完成

#### P0-1 完成 FE v2 全链路并建立唯一结果入口

**问题。** FE v2 目前只有 CatBoost 结果，图模型、OOF、融合、置信区间和调查视图未闭环；
README、RESULTS、MODEL_CARD 多处维护数字，容易出现口径漂移。

**建议实现。** 保持当前后台串行实验，完成后新增 `scripts/build_resume_evidence.py`，只从
manifest 和 metrics 文件读取数据，生成 `artifacts/resume_evidence/report.json` 与
`docs/RESUME_EVIDENCE.md`。不要人工复制指标。

**验收标准。**

- FE v2 七步运行链全部 complete，所有指标能追溯到 run_id、配置、数据和特征版本。
- 自动比较 v1/v2 CatBoost、GAT、融合，并明确正结果或负结果。
- 自动检查测试时间范围、测试行数、正例数、score 列和 Bootstrap 次数一致。
- 任何 smoke 产物、缺 manifest 产物或正在运行产物不能进入简历报告。

#### P0-2 把“小型 BM25”升级为可评测的混合 RAG

**问题。** 招聘页面频繁要求 Embedding、Vector DB、Chunking 和 Rerank；本项目目前只有本地
BM25 与少量 Typology 文档，生成侧有 Golden，检索侧没有独立质量指标。

**建议实现。** 保留 BM25 作为基线，新增本地、可替换的 dense retriever 和可选 reranker：

```text
query builder
  ├─ BM25 top-20
  ├─ embedding cosine top-20
  └─ reciprocal-rank fusion → rerank → top-3 references
```

初版使用本地 FAISS 或 NumPy 索引即可，不需要为了简历强行部署 Milvus。构建至少 80 条
检索评测 query，包括同义表达、字段缺失、跨 Typology 混合、无相关文档和 hard negative。

**验收标准。**

- 报告 BM25、dense、hybrid、hybrid+rerank 的 Recall@1/3、MRR、nDCG@3、无答案误召回率。
- 文档版本、embedding 模型、chunk 规则、索引版本均进入 run manifest。
- 检索结果只作为调查线索，不改变交易风险分数和案件结论。
- 只有 hybrid/rerank 确有可重复增益才升为默认，否则保留 BM25 并报告负结果。

#### P0-3 将 LangGraph 变成真正可演示的“受控 Agent”

**问题。** 当前状态图固定串行执行，`tool_call_count` 实际主要记录一次 Typology 检索；没有
条件路由、持久化 checkpoint、暂停/恢复和审批决策。携程、Shopee、国金岗位尤其看重
Function Calling、流程拆解、HITL 与可回溯。

**建议实现。** 增加三个只读、参数受限的工具：

1. `get_feature_snapshot(alert_id)`：读取告警对应的已冻结 PIT 特征。
2. `get_bounded_subgraph(alert_id, hops<=2, max_edges<=N)`：读取受限历史子图。
3. `search_typologies(query, top_k<=5)`：返回版本化文档引用。

由确定性策略或结构化 LLM 输出选择下一步；所有工具参数使用 Pydantic Schema，禁止任意 SQL、
路径和外部 URL。生成草稿后用 LangGraph checkpoint + `interrupt()` 暂停，由审核人执行
approve/edit/reject，再恢复并写入审计事件。LangGraph 官方文档说明，持久化 checkpoint 是
人工中断、恢复和容错的基础：
[Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、
[Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)。

**验收标准。**

- 同一 thread_id 可在进程重启后恢复；审批前不产生“已提交/已处置”状态。
- 覆盖正确工具选择、错误工具拒绝、超限参数、工具超时、重复调用和恢复幂等性。
- 每案输出 node timeline、工具名、参数摘要、来源、耗时、状态变更和审核决定。
- Golden 至少扩展到 60 案，并分别报告工具选择准确率、参数合法率、事实一致率、拒答率、
  恢复成功率、p50/p95 延迟与 token/cost 覆盖率。

#### P0-4 补齐 API 与模型性能证据

**问题。** 有 FastAPI/Docker 只能证明“能启动”，不能证明模型服务可用。招聘中的部署、实时
监控和工程落地需要可量化证据。

**建议实现。** 提供 Mock API 和本地冻结产物 API 两套压测，记录冷启动、单笔、批量、调查
工作流的延迟和资源峰值。测试工具可选 Locust、k6 或简单异步 Python 客户端。

**验收标准。**

- 固定硬件、并发、请求数量、batch size 和数据边界。
- 报告吞吐、p50/p95/p99、错误率、CPU/RAM/GPU 峰值和模型加载时间。
- 加入无效 alert_id、超大 batch、并发恢复、超时和降级路径。
- README 只写测得的本机性能，不将其表述为生产 SLA。

### 3.2 P1：提升大厂风控算法匹配度

#### P1-1 做一条 Spark SQL 特征等价链，而不是再造全系统

招聘页面中 Hive/Spark 出现频繁。建议选择 5—8 个最有代表性的 PIT 特征，在本地 Spark 或
可用大数据环境实现分区读取、历史窗口、预聚合和增量写入，与官方 PIT 结果逐行对齐。

验收包括：match rate、扫描行数、shuffle、耗时、峰值内存及增量重跑。所有 ODPS/Spark SQL
保持显式等值连接和预聚合，**禁止普通笛卡尔积**。这项证据比在简历技能栏单写“Hive/Spark”
更可信。

#### P1-2 增加统一实验追踪，但保留现有 manifest

现有 run manifest 已经解决大部分可复现问题。可以增加 MLflow 适配层，将参数、代码版本、
数据/特征版本、指标和产物索引同步到本地 tracking store，不要重写训练器。MLflow 官方
[Tracking 文档](https://mlflow.org/docs/latest/ml/tracking/)将 run、参数、指标、代码版本与
产物作为核心对象，和当前项目结构匹配。

验收重点不是“有一个 UI”，而是：同协议 run 可比较、失败 run 可识别、候选模型满足门槛后
才能标为 candidate、测试集不参与模型选择。

#### P1-3 给图模型补工程型消融

模型架构对比已经足够，不建议再加 TransformerConv/TGAT 只为凑名词。更值得补的是：

- 邻居 fanout、历史窗口和 batch size 对 PR-AUC/延迟/显存的 Pareto 曲线。
- 冷启动账户、低度节点、高度节点、跨境/换汇关系切片。
- 删除金额、时间、关系或节点统计后的成组消融。
- 同一最优配置至少额外一个 seed；报告变化，不只报最好结果。

这些实验能回答“GAT 为什么有效、代价是什么、部署时如何取舍”，比继续比较更多 GNN 名称
更有面试价值。

### 3.3 P2：仅投大模型算法/Agent 岗时选做

#### P2-1 PEFT/QLoRA 小实验

岗位确实会要求 LoRA/QLoRA/PEFT，但本项目没有真实合规标注，不应把微调硬接到风险评分。
如目标岗位明显偏大模型算法，可另建隔离实验：使用公开/合成的“证据包→结构化调查问题”数据，
对小型开源模型做 QLoRA，并与 prompt-only 基线比较 JSON 合规、事实一致、拒答、成本和延迟。

只有在固定评测集上显示收益才保留；不得写成“微调提升 AML 识别率”。

#### P2-2 MCP 或 Multi-Agent

神舟软件等岗位出现 MCP，部分岗位提到 Multi-Agent，但对本项目不是默认优先级。现有三个只读
调查工具稳定后，可以把它们暴露为本地 MCP server，展示工具协议和权限边界。除非单 Agent
在可测任务上出现明确瓶颈，否则不拆成多个 Agent；多 Agent 会增加延迟、故障点和事实一致性
风险，却不自动增加简历含金量。

## 四、建议的实施顺序与工作量

| 优先级 | 工作项 | 预计个人工作量 | 依赖 | 可形成的简历证据 |
|---|---|---:|---|---|
| P0 | FE v2 全链路闭环 + 自动结果报告 | 1—2 天（不含训练等待） | 当前后台实验 | 特征升级、时间外对照、可追溯 run |
| P0 | 混合 RAG + 检索 Golden | 3—5 天 | 扩充 Typology/查询集 | Recall@k/MRR/nDCG、hard negative |
| P0 | LangGraph checkpoint + HITL + 三个只读工具 | 3—5 天 | 现有 Evidence Package/API | Function Calling、恢复、审计、失败降级 |
| P0 | API/Agent 压测与资源报告 | 1—2 天 | 冻结模型和调查链 | p95、吞吐、资源峰值、错误率 |
| P1 | Spark PIT 特征等价链 | 2—4 天 | Java/Spark 环境 | Hive/Spark、分区、shuffle、结果一致性 |
| P1 | MLflow 适配与 candidate gate | 1—2 天 | 现有 manifest | 实验追踪、模型生命周期 |
| P1 | GAT 工程消融与第二 seed | 2—4 天 + 训练 | GPU | 性能—资源 Pareto、稳定性 |
| P2 | QLoRA 调查结构化输出实验 | 3—6 天 + GPU | 合成训练集 | PEFT、基线对照、LLM 评测 |

推荐投递前先完成前四项。P1/P2 应根据目标 JD 选择，不需要全部完成。

### 4.1 代码框架落地状态（2026-07-28）

在 FE v2 后台实验运行期间，已完成不会争抢 GPU 的框架工作：

- P0-2：实现 BM25、TF-IDF 本地稠密基线、RRF、可选 rerank 和独立检索指标；15 条种子集
  只验证程序，hybrid/rerank 尚无稳定增益，不能写成正式 Golden 结果。
- P0-3：实现三个结构化只读工具、最多四次确定性路由、LangGraph checkpoint、强制
  approve/edit/reject 中断恢复，以及不记录正文/特征值的成功与失败审计事件。
- P0-4：实现有界并发 HTTP 压测与完整产物驱动的简历证据生成器；不会从运行中产物提取
  简历数字，本机观测也不表述为生产 SLA。
- P1-1：提供五个代表性 Spark PIT 窗口特征骨架，严格使用 `[t-window, t)`；当前 Windows
  未安装 Java/PySpark，因此等价性和资源报告仍未验收。代码不包含普通笛卡尔积。
- P1-2：提供 MLflow 可选适配层和仅使用验证指标的 candidate gate；MLflow 未在后台训练
  期间安装，现有 manifest 仍是权威记录。

环境采用 extras 分组，锁文件已更新；当前 GPU 版 PyTorch 与 SQLite checkpoint 可用，
SentenceTransformer、MLflow、PySpark 按需安装。详细命令和未完成项见
[`P0_FRAMEWORK_STATUS.md`](P0_FRAMEWORK_STATUS.md)。

## 五、目标仓库结构

建议在现有结构上增量扩展：

```text
configs/
  retrieval/hybrid_v1.yaml
  agents/investigation_v2.yaml
golden/
  retrieval_queries_v1.json
  agent_cases_v2.json
src/aml_evidence_graph/
  retrieval/{bm25,dense,hybrid,rerank}.py
  investigation/{tools,router,checkpoint,audit}.py
  tracking/mlflow_adapter.py
scripts/
  evaluate_retrieval.py
  benchmark_api.py
  build_resume_evidence.py
  replay_spark_features.py
docs/
  RETRIEVAL_EVALUATION.md
  AGENT_HITL_EVALUATION.md
  SERVING_BENCHMARK.md
  RESUME_EVIDENCE.md
```

不要把本地模型权重、完整交易或服务密钥提交到 Git。公开仓库只保留配置、Schema、Mock、
小型 Golden 和聚合指标。

## 六、简历表达建议

### 6.1 项目标题

推荐：

> **AML Evidence Graph：基于 PIT 特征与历史图学习的反洗钱风险排序及证据约束调查系统**

避免：

> 基于大模型的智能反洗钱识别平台

后者会误导面试官认为 LLM 直接打分，也无法体现项目最强的时间防泄漏与图模型能力。

### 6.2 当前即可使用的三条项目描述

以下数字来自当前主线，使用时必须同时保留“公开合成数据”限定：

1. 基于公开合成 SAML-D 约 950 万笔交易，构建按日分区的 Point-in-Time 特征与固定时间外
   评估流水线，确保每笔交易仅访问预测时点前的历史，覆盖规则、LR、CatBoost、校准、漂移
   与固定告警预算评估。
2. 实现历史交易图上的边分类，完成 GraphSAGE/GAT/RGCN/PNA 同协议对比；GAT 测试
   PR-AUC 0.948，在 0.1% 告警预算下 Precision 0.974、Recall 0.838，并用 200 次分层
   Bootstrap 报告不确定性。
3. 构建 RiskEvidencePackage 约束的 LangGraph 调查链，将 Typology 检索、事实校验、SAR
   草稿和人工复核与风险评分解耦；在项目裁定 Golden 上验证 Schema、事实快照、无证据拒答
   和注入拦截，并记录 Prompt 版本、延迟和 token 使用。

第三条应使用“调查链”而非“自主 Agent”。完成 P0-2/P0-3 后，才可升级为：

> 实现带持久化 checkpoint 的受控风控 Agent，通过结构化 Function Calling 查询 PIT 特征、
> 有界历史子图和 Typology；支持人工 approve/edit/reject 后恢复执行，并以检索与 Agent
> Golden 评估工具选择、事实一致、拒答、恢复成功率和 p95 延迟。

其中所有指标都必须替换为实际测得结果，不能使用计划值。

### 6.3 面试时的 90 秒主线

1. **业务对象**：对极低正例率的交易边做调查优先级排序，不自动判罪或申报。
2. **最大技术风险**：随机切分和全量聚合会泄漏未来，因此先做 PIT 数据与历史图快照。
3. **模型选择**：CatBoost 是强可解释基线，GAT 捕捉邻域关系；同协议测试表明 GAT 最强，
   融合反而低于 GAT，因此诚实保留负结论。
4. **业务评估**：主看 PR-AUC 与告警预算，而不是 Accuracy；同时看漂移、阈值、冷启动和资源。
5. **LLM 边界**：LLM 不打风险分，只在证据包和工具白名单内辅助调查，事实失败就拒绝并转人工。
6. **生产差距**：数据为公开合成，当前是可复现原型；真实落地仍需标签延迟、权限、审批、
   实时特征和长期回溯验证。

## 七、完成定义

当以下材料全部可由命令重新生成时，项目才算达到“简历投递版”：

- 一份唯一的 `RESUME_EVIDENCE.md`，包含数据、切分、模型、指标、run_id 与硬件。
- 一张主架构图，清楚分开风险评分、证据聚合、Agent 调查和人工审批。
- 一张模型对照表，包含规则、CatBoost、GAT、融合及负结果。
- 一张检索评测表和一张 Agent Golden 表。
- 一份 API/Agent 性能报告。
- 一条 Mock Demo 命令和一条完整测试命令。
- README 中所有公开数字与 manifest 自动报告一致。
- 简历中每个动词都能指向代码、测试、指标或演示，不写尚未完成的能力。

## 八、进一步问题

这些问题会影响 P1/P2 的取舍，建议投递前按目标公司回答：

1. 目标更偏信贷/反欺诈建模、支付图算法，还是金融大模型 Agent？
2. 是否有足够 GPU 与时间做第二 seed 和 QLoRA，而不影响 FE v2 主链？
3. 能否获得合规可公开的 Typology 文档来扩充检索评测？
4. 目标岗位是否明确要求 Spark/Hive；若没有，是否值得为其增加环境复杂度？
5. 简历篇幅只允许三条时，优先展示 GAT、防泄漏和受控 Agent，还是表格模型与监控？

默认建议是：**先把现有优势闭环，再按 JD 做一项针对性扩展。** 对传统风控岗补 Spark；
对图算法岗补 GAT 工程消融；对大模型风控岗补混合 RAG、真实工具路由与持久化 HITL。
