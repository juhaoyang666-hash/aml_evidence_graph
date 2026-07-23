# 数据字典

| Canonical 字段 | SAML-D 源字段 | 用途 | 安全边界 |
|---|---|---|---|
| transaction_id | 行序生成的内部 ID | 关联评分与证据 | 内部键；Demo 使用虚构 ID |
| event_ts | Date + Time | 时间切分与 PIT | 仅派生时间特征 |
| sender_account_id / receiver_account_id | Sender_account / Receiver_account | 图、窗口特征 | 合成账户标识；本地产物不入 Git |
| amount | Amount | 金额特征 | 遵守币种口径 |
| payment_currency / received_currency | Payment_currency / Received_currency | 币种特征 | 可入模 |
| sender_location / receiver_location | Sender_bank_location / Receiver_bank_location | 地域特征 | 可入模 |
| payment_type | Payment_type | 支付方式 | 可入模 |
| is_laundering | Is_laundering | 监督标签 | 仅训练/评估，禁止入特征 |
| laundering_type | Laundering_type | 模式切片 | 仅评估切片，禁止入特征 |

数据集来源为公开合成 **SAML-D**。结论须标注“公开合成基准”，不得包装成真实业务确认数据。

`aml-build-investigation-views` 仅消费模型概率、规则命中和上述交易字段：

| 产物 | 键 | 含义 |
|---|---|---|
| account_risk.parquet | 账户标识 × 明确 as-of 截点 | 窗口内最高/Top-N 交易风险、告警笔数和规则命中数；不是账户洗钱标签。 |
| funds_paths.json | 路径端点与时间 | 严格时间递增的资金路径摘要。 |
| case_views.json | 高风险弱连通子图 | `investigation_candidate`，不代表已确认案件、团伙或犯罪结论。 |
