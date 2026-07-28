# Golden Cases

本目录含调查草稿 / Typology 检索 / SAR 草稿 / 事实校验的自动化回归用例。

| 文件 | 性质 | 规模 | 用途 |
|---|---|---|---|
| `mock_cases.json` | 虚构 smoke 种子 | 6 | CI / 本地快速回归 |
| `cases_v1.json` | **项目裁定** Golden | **34** | 分层 typology + 低证据 + 对抗（含 B5 扩容） |
| `adjudication_v1.json` | 逐案裁定记录 | **34** | 裁定决策 / 期望结果 / 备注 / 时间戳 |

## 裁定状态（2026-07-26）

`cases_v1.json` 已在用户明确授权下由 agent 完成正式逐案裁定
（「人工裁定 Golden由你来裁定」→ adjudicator=`agent-authorized-by-user`）。  
B5 对抗扩容（`agent-adv-07`…`10`）同样按该裁定约定写入 `adjudication_v1.json`。

| 约定 | 说明 |
|---|---|
| 数据性质 | 非 SAML-D 抽样、非真实案件；按 Evidence Package 结构构造 |
| 裁定性质 | **项目裁定的 Golden v1**（用户授权的 agent 裁定），**不是**独立第三方人工评审团 |
| 用途 | 调查起草、BM25 检索、无证据拒答、幻觉拦截（注入坏注释）、对抗提示回归 |
| 不可用途 | 对外宣称独立第三方人工面板准确率、替代合规审批后的生产 Golden、回灌为评分标签 |
| 分层（当前） | typology 18 · low_evidence 6 · adversarial **10**（含 `injected_annotation` 探针 **7**） |

详见 `adjudication_v1.json` 中每案的 `decision` / `expected_outcomes` / `notes`。

## 规划修订

正式集由 ~100 案下调为 **30** 案（分层 18 + 低证据 6 + 对抗 6），再于算力附录
**B5** 扩至 **34** 案（+4 对抗：FX 捏造、账户 exfil、过度自信百分比、角色越权）。  
三项可自动化指标：**幻觉拦截率**、**无证据拒答率**、**端到端延迟 p50/p95**。  
模板路径（34 案）幻觉拦截 / 无证据拒答均为 **1.0**（见 `docs/大模型调查系统.md`）。

## 评测命令

```bash
export PYTHONPATH=src
# 模板路径（默认 CI / 无 LLM）
/data1/yangjuhao/envs/risk/bin/python -m aml_evidence_graph.investigation.golden \
  --cases golden/cases_v1.json --typologies knowledge/typologies \
  --output artifacts/golden_summary.json

# 可选：ECNU LLM（需 .env：AML_LLM_ENABLED + API key；勿把密钥写入仓库）
/data1/yangjuhao/envs/risk/bin/python -m aml_evidence_graph.investigation.golden \
  --cases golden/cases_v1.json --typologies knowledge/typologies \
  --output artifacts/golden_summary_llm.json --use-llm
```

对抗子集中带 `injected_annotation` 的用例用于确定性验证 `validate_annotation`
拦截数字/越界引用；不依赖模型「碰巧幻觉」。LLM 路径额外覆盖 prompt 注入与诱导结论。
