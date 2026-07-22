# Canonical Transaction 数据字典

| Canonical 字段 | 私有源字段 | 用途 | 安全边界 |
|---|---|---|---|
| transaction_id | 源行号派生 | 内部、稳定的流水引用 | 不等同于源系统交易号 |
| event_ts | Date + Time | 时间切分、PIT 截点 | UTC 标准化 |
| sender_account_id / receiver_account_id | 收/付款账户 | 图、窗口特征 | 输入已脱敏；私有产物不公开 |
| amount | Amount | 当前交易数值特征 | 不进入公开 Demo |
| payment_currency / received_currency | 币种字段 | 表格和 PIT 特征 | 分类值受控保留 |
| sender_location / receiver_location | 银行地字段 | 跨境特征 | 分类值受控保留 |
| payment_type | 支付方式 | 表格特征 | 分类值受控保留 |
| is_laundering | Is_laundering | 唯一交易级监督标签 | 不作为线上特征 |
| laundering_type | Laundering_type | 评估切片 | 不作为线上特征 |
| source_row_number | CSV 行号 | 重放、稳定采样 | 不公开 |

任何 ODPS 特征或规则 SQL 只能通过明确业务键进行等值连接，禁止普通笛卡尔积。

## 后评分调查视图

`aml-build-investigation-views` 仅消费模型概率、规则命中和上述已脱敏交易字段：

| 产物 | 粒度 | 含义与边界 |
|---|---|---|
| account_risk.parquet | 脱敏账户标识 × 明确 as-of 截点 | 窗口内最高/Top-N 交易风险、告警笔数和规则命中数；不是账户洗钱标签。 |
| funds_paths.json | 高风险交易边路径 | 最多三跳、严格时间递增、无循环的资金路径候选；同一时间戳的边不串联。 |
| case_views.json | 高风险弱连通子图 | `investigation_candidate`，不代表已确认案件、团伙或犯罪结论。 |

构建器拒绝接收 `is_laundering` 和 `laundering_type`；这些标签只能留在训练/评估链路。
