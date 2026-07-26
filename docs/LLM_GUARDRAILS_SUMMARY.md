# LLM 护栏评测摘要（投字节/美团风控大模型向）

完整边界见 [LLM_PLAN.md](LLM_PLAN.md)；案例与裁定见 `golden/`。  
**一句话**：LLM 只写措辞与提问，**不**提高识别率。

## 已落地护栏

| 机制 | 行为 |
|---|---|
| 出站最小化 | 去掉交易/账户/告警 ID、时间戳、原始数值 |
| 返回校验 | 引用白名单 + 数值/实体正则；失败 → `rejected_facts` |
| 降级 | 网络/JSON 失败回退确定性模板；不影响评分 |
| Prompt 版本化 | `configs/prompts/ecnu-risk-evidence-v1.yaml` |

## Golden 30 关键（可引用）

| 路径 | 幻觉拦截 | 无证据拒答 | 备注 |
|---|---:|---:|---|
| 模板（无 LLM） | 1.0 | 1.0 | `artifacts/golden_summary.json` |
| ECNU `ecnu-max` | 1.0 | 1.0 | `artifacts/golden_summary_llm.json`；correct_rejection 0.90 |

Adjudicator：`agent-authorized-by-user`（非第三方面板）。

## 面试忌讳

- 「我们用大模型做反洗钱识别 / 提升了 PR-AUC」  
- 隐瞒合成数据或把 Mock Demo 指标当效果
