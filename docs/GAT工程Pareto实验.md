# GAT 工程型 Pareto 实验

> 日期：2026-07-28—2026-07-29  
> 数据：公开合成 SAML-D；固定时间切分。  
> 选择边界：六个候选只读取 train/validation；test 只允许最终候选读取一次。

## 协议

固定架构、seed `20260722`、hidden dim 64、两层 GAT、最多 12 epochs 和 validation
early stopping。以 `batch=2048, fanout=(15,10), history=30d` 为基线，每次只改变一类
工程参数：

- batch size：`1024 / 4096`；
- neighbor fanout：`(10,5) / (25,15)`；
- 历史窗口：`14d / 60d`。

预注册选择规则：先保留距离最佳 validation PR-AUC 不超过 `0.002` 的候选，再依次选择
validation inference 更快、GPU 峰值更低、训练更快者。汇总脚本不复制或读取基线的
test metric。

## 验证集结果

验证集共 `1,814,844` 笔交易。

| 配置 | Validation PR-AUC | 训练耗时 | Validation 推理 | GPU peak |
|---|---:|---:|---:|---:|
| **window 60d** | **0.924227** | 681.1 s | 11.73 s | 1125.8 MiB |
| baseline：2048 / (15,10) / 30d | 0.917883 | 1357.4 s | 27.07 s | 1125.4 MiB |
| window 14d | 0.913967 | 520.3 s | **9.92 s** | 1125.5 MiB |
| fanout (25,15) | 0.913852 | 566.2 s | 13.13 s | 1125.7 MiB |
| batch 4096 | 0.906409 | **513.6 s** | 14.65 s | 1126.0 MiB |
| fanout (10,5) | 0.896107 | 673.1 s | 11.31 s | 1125.6 MiB |
| batch 1024 | 0.886937 | 891.7 s | 16.86 s | **1125.4 MiB** |

`window_60d` 比 30d 基线高 `0.006344` PR-AUC，是唯一落入最佳值 `0.002` 容差内的
候选，因此被冻结为一次性测试候选。减小 fanout 或 batch 没有形成精度—资源上的优势；
本实现的 GPU 峰值主要由节点 embedding 决定，各配置几乎相同。

机器上六个候选按两路并发、三波执行。训练计时由每个进程内部采集，但端到端 wall time
还包含重复读取约 950 万行数据和历史图构建；不同波次受到 page cache、CPU/IO 争用影响。
旧基线来自较早的单路运行，因此时延只能用于方向性 Pareto，不是严格的同负载微基准。

## 冻结测试

最终 `window_60d` checkpoint 的唯一一次冻结测试结果待本机回放完成后写入。回放直接加载
验证期保存的 checkpoint，不重新训练、不比较其他候选测试指标。

## 产物与复现

- 候选入口：`scripts/run_gat_validation_candidate.py`
- 两路调度：`scripts/run_gat_pareto.sh`
- 验证汇总：`scripts/summarize_gat_pareto.py`
- 冻结测试：`scripts/evaluate_frozen_graph_checkpoint.py`
- 私有聚合产物：`artifacts/gat_pareto_summary.json`

这些指标是本机离线实验，不是生产在线 SLA；模型未接触真实金融数据。
