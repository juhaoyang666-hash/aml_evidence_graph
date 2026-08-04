# Golden 与回归集合

本目录只存放虚构或项目裁定的回归输入，不包含 SAML-D 原始交易、真实案件、客户标识或外部
服务密钥。生成、检索和 Agent 三类集合相互独立，不能把调查裁定回灌成交易评分标签。

## 当前集合

| 文件 | 性质 | 规模 | 用途 |
|---|---|---:|---|
| `mock_cases.json` | 完全虚构 | 6 | CI 与本地 Golden smoke |
| `cases_v1.json` | 项目裁定生成 Golden | 34 | Typology、低证据、对抗与事实校验 |
| `adjudication_v1.json` | `cases_v1` 裁定记录 | 34 | 期望结果、理由、备注与裁定时间 |
| `agent_cases_v2.json` | 项目构造 Agent 回归 | 60 | 路由、工具参数、人工审核和故障恢复 |
| `retrieval_queries_v1.json` | 早期检索回归 | 15 | 历史兼容，不作为当前主表 |
| `retrieval_queries_v2.json` | 检索校准/开发集 | 80 | 中英文、改写、多标签、hard negative、拒答 |
| `retrieval_queries_v3_additions.json` | 冻结增量诊断 | 50 | 在阈值冻结后评估覆盖与拒答泛化 |
| `retrieval_queries_v4_project_blind.json` | 项目作者盲法集合 | 50 | answerability gate 唯一一次项目评测 |
| `retrieval_adjudication_v4_project_blind.json` | v4 裁定 | 50 | 35 条可回答、15 条无答案及裁定声明 |

`v2+v3` 构成文档中的 130 条检索开发/诊断集合。v4 在参数和门禁冻结后构建并只评一次；后续
不得使用 v4 调参或再次宣称冻结测试。聚合指标见[检索评估](../docs/检索评估.md)。

## 裁定与独立性边界

- `cases_v1` 和 v4 检索集均由用户授权当前 Agent 完成项目裁定。
- v4 满足 `blind_to_model_outputs=true`，但
  `independent_from_system_development=false`。
- 这些集合可用于项目回归、Bad Case 和求职工程证据，不是独立第三方合规专家面板、银行生产
  验收或真实案件标签。
- `exclude`、`answerable`、`no_answer` 和生成侧期望状态只服务各自评测，不参与 CatBoost、GAT
  或融合器训练。

## 生成 Golden（34 案）

分层为 typology 18、low evidence 6、adversarial 10；其中 7 条带确定性
`injected_annotation` 探针，用于验证数字、实体和越界引用拦截，而不是依赖模型“碰巧幻觉”。

当前模板路径：Schema 合规、事实快照一致、幻觉拦截和无证据拒答均为 `1.0`。2026-08-04
已完成 34 案的 ECNU 外部调用：v1 首次冻结执行暴露 names-only 定性推断 Bad Case；v3 在同一
集合上的修复回归达到外部解析率 `0.7407`、解析后事实门 `1.0`，20 条安全输出的项目内人审
Grounding/Overall 为 `0.90`。v3 是开发回归而非独立盲测，且仍有 7 次非法 JSON 回退。
裁定文件为 `llm_adjudication_ecnu_max_v1.json` 与 `llm_adjudication_ecnu_max_v3.json`，详见
[大模型调查系统](../docs/大模型调查系统.md)。
可公开、无生成原文的聚合版本位于
`../reports/public/llm_ecnu_max_golden34_20260804.json`，并由发布门禁核对上述裁定文件。

## Agent 回归（60 案）

| 分组 | 数量 | 覆盖 |
|---|---:|---|
| 路由 | 12 | 特征、子图、低证据与缺引用组合 |
| 工具 | 24 | 合法边界、超限参数、SQL/路径/URL 注入、缺字段 |
| 人工审核 | 12 | approve/edit/reject、非法动作、缺备注 |
| 恢复 | 12 | 检索故障、checkpoint 与 SQLite 恢复 |

项目内确定性基线的 Case 通过率、工具选择准确率、参数合法率、事实一致率和恢复成功率均为
`1.0`。它验证受控工作流，不代表外部 LLM 质量或生产 SLA。

## 检索项目盲法结果

在 v4 上，冻结 hybrid 与 `hybrid + answerability gate` 的 Recall@3/MRR 都为
`0.757/0.619`；无答案误召回率由 `0.600` 降为 `0.133`。四项预注册项目 sidecar 门禁通过，
但 Recall@3 仍有改进空间，且裁定者参与系统开发，不能称为独立验证。

## 评测命令

```bash
export PYTHONPATH=src
PY=python

# 生成 Golden：确定性模板，不调用外部 LLM
$PY -m aml_evidence_graph.investigation.golden \
  --cases golden/cases_v1.json \
  --typologies knowledge/typologies \
  --output artifacts/golden_summary.json

# 受控 Agent 60 案
$PY scripts/operations/evaluate_agent_golden.py --overwrite

# 检索开发集聚合评测
$PY scripts/retrieval/evaluate_retrieval.py

# 求职发布版 Mock 验收
$PY scripts/operations/verify_resume_release.py

# 外部 LLM 34 案（显式启用后；值仍不进入外部 payload）
$PY -m aml_evidence_graph.investigation.golden \
  --cases golden/cases_v1.json \
  --typologies knowledge/typologies \
  --output artifacts/llm_ecnu_max/golden34.json \
  --use-llm

# 对哈希绑定的项目内人工裁定做覆盖与指标汇总
$PY scripts/reporting/summarize_llm_human_review.py \
  --summary artifacts/llm_ecnu_max/golden34.json \
  --adjudication golden/llm_adjudication_ecnu_max_v3.json \
  --output artifacts/llm_ecnu_max/human_review.json

# 从两个冻结本地摘要发布脱敏聚合（原始摘要仍留在 artifacts）
$PY scripts/reporting/publish_llm_evidence.py \
  --baseline-summary artifacts/llm_ecnu_max/golden34_v1.json \
  --baseline-adjudication golden/llm_adjudication_ecnu_max_v1.json \
  --development-summary artifacts/llm_ecnu_max/golden34_v3.json \
  --development-adjudication golden/llm_adjudication_ecnu_max_v3.json \
  --evaluation-id ecnu-max-golden34-v1-v3 \
  --evaluated-at 2026-08-04T03:45:00Z \
  --output reports/public/llm_ecnu_max_golden34_20260804.json
```

可选外部 LLM 评测必须显式启用并从当前进程读取 API key；不要将 key 写入 Golden、`.env`
示例、命令历史或仓库。LLM 不参与风险评分和案件结论。
