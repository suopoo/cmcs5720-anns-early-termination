# 三个实验的最终交付摘要（全部为真实运行结果）

本目录包含 Experiment 1–3 的代码、共享数据缓存、中间轨迹、模型、逐 Query
结果、统计表、图表和报告。所有数字均来自 `data/`、`analysis/` 下的 CSV，
可由 `reproduce.md` 中的命令重新生成；未使用模拟数据。

## 0. 统一环境

| 项 | 值 |
|---|---|
| 数据集 | GIST1M，1,000,000 × 960，L2 |
| 索引基向量 | `train[0:992000]`（8,000 条 Query 留出） |
| HNSW | hnswlib M=16，efConstruction=200，seed 42 |
| 训练 / 验证 / 测试 | 5,000（exp2 为 2,000）/ 500 / 1,000，三者 Query ID 互斥 |
| 校验 | 解析图与 hnswlib 原生 top-10 重合率 0.988；精确 GT 对 992k 基向量重算 |
| 机器 | i7-12700H，16 GB RAM，仅 CPU |

共享轨迹保存在 `shared/traces/`，模型保存在各实验 `models/`，所有实验产物
都在本文件夹内（D 盘）。

## 1. Experiment 1 — Feature Transfer（k=10, R_t=0.90, f=1260）

| method | group | mean recall | hit rate | mean NDC | P95 NDC | native MAE | mult / calls |
|---|---|---:|---:|---:|---:|---:|---:|
| LAET | L0 | 0.9040 | 0.499 | 4,073 | 9,310 | 895.8 | 5.07 |
| LAET | L1 | 0.8908 | 0.465 | 3,798 | 10,247 | 870.2 | 4.57 |
| LAET | L2 | 0.8943 | 0.468 | 3,792 | 9,565 | 894.3 | 4.57 |
| DARTH | D0 | 0.9242 | 0.537 | 5,013 | 11,083 | — | 8.65 calls |
| DARTH | D1 | 0.9214 | 0.515 | 5,085 | 19,590 | — | 8.22 calls |
| DARTH | D2 | 0.9214 | 0.515 | 5,085 | 19,590 | — | 8.22 calls |
| DARTH | D0-original | 0.9250 | 0.546 | 4,632 | 10,381 | — | 8.52 calls |

配对 95% CI（1,000 条测试 Query）：

* LAET L1 vs L0：|log2 代价误差| −0.0715 [−0.1123, −0.0313]（预测变准），
  Recall −0.0132 [−0.0189, −0.0074]，NDC −275 [−418, −114]。
* LAET L2 vs L0：误差 −0.0467 [−0.0836, −0.0075]，Recall −0.0097，
  NDC −281。
* LAET L1 vs L2：误差 −0.0248 [−0.0539, +0.0066]（不显著）。
* DARTH D1 vs D0：Query 级 Recall 误差 −0.00007 [−0.0016, +0.0016]，
  Recall/NDC 均不显著；D1≡D2（DARTH 的 11 个特征是 ANNiE basic 的子集）。

实测延迟（30 条测试 Query）：L0 34 ms、L1 29 ms、L2 33 ms；D0 66 ms、
D1/D2 183 ms（每个 checkpoint 都提图特征，约 2.8×）。

**结论：** 迁移的 ANNiE 特征确实降低了 LAET 的代价预测误差（H1 支持），但
没有带来更好的搜索（Recall 下降、NDC 下降；H3 支持“预测提升不自动转化为
端到端提升”）。L1 vs L2 不显著（H1c 证据不足）。DARTH 迁移是控制良好的零
结果，且图特征提取使延迟增加 2.8 倍。f 敏感性曲线基本平坦。

## 2. Experiment 2 — Per-query Error Analysis（k=100, R_t=0.95, ef=2048）

| method | mean recall | hit rate | mean NDC | P95 NDC | relative extra NDC | oracle attainable |
|---|---:|---:|---:|---:|---:|---:|
| LAET | 0.9562 | 0.676 | 12,150 | 28,754 | 0.799 | 0.980 |
| DARTH | 0.9608 | 0.644 | 12,691 | 27,549 | 1.063 | 0.926 |
| ANNiE-S | 0.9736 | **0.838** | 15,996 | 34,560 | 0.935 | 0.980 |

Quality-failure 重合（同一 1,000 Query ID，按 Oracle NDC 分层置换 2,000 次）：

| pair | intersection | Jaccard | P(B\|A) | expected | p |
|---|---:|---:|---:|---:|---:|
| LAET∩DARTH | 225 | 0.495 | 0.694 | 115.3 | 0.0005 |
| LAET∩ANNiE | 122 | 0.335 | 0.377 | 52.5 | 0.0005 |
| DARTH∩ANNiE | 149 | 0.404 | 0.419 | 57.7 | 0.0005 |
| 三者共同 | 118 | — | — | — | — |

训练集规模（3 seeds 均值，hit rate）：ANNiE 0.769→0.838（500→2,000）；
DARTH 0.633→0.653；LAET 0.651→0.676（非单调，n=1,000 时 0.573）。
Recall Target 0.90/0.95/0.99 的 hit rate：LAET 0.604/0.676/0.687；
DARTH 0.703/0.644/0.584；ANNiE 0.854/0.838/0.703。
Hard-query-enriched 训练（n=1,000）使三者 hit rate 分别提高 2.8/7.0/6.0 pp，
但 NDC 增加 9–16%。

**结论：** 三种方法失败模式不同；失败 Query 有显著重合且分层后仍显著，说明
存在 Query 自身的难度成分，但条件概率不同说明机制并不相同。失败 Query 中，
77% 的 LAET 失败是预算低估（平均 -5,610 NDC），85% 的 DARTH 失败是停止时
Recall 高估（平均 +0.026），而 ANNiE 的失败中只有 49% 是成本低估（其保守
分位数在平均上过度花费，却仍在困难子集上未达标）。

## 3. Experiment 3 — Asymmetric Loss & Timing（k=10, R_t=0.90）

**LAET（A2：每个模型在验证集上重新校准 multiplier）**

| config | tau | hit rate | mean NDC |
|---|---:|---:|---:|
| L0 L2 | — | 0.499 | 4,073 |
| L1 | 0.50 | 0.473 | 3,622 |
| L2 | 0.75 | 0.495 | 4,105 |
| L3 | 0.90 | 0.484 | 3,741 |
| L4 | 0.95 | 0.504 | 3,985 |

L4 vs L0：Recall +0.0010 [−0.0046, +0.0063]，NDC −88 [−226, +58]，均不显著。

**LAET（A1：固定 L0 的 multiplier）**：tau 0.95 把 hit rate 从 0.499 提到
0.818，但 NDC 从 4,073 增到 12,530——收益来自更大的有效预算。

**DARTH**：tau 0.25/0.10/0.05 把 hit rate 从 0.546 提到 0.681/0.809/0.876，
但 NDC 从 4.6k 增到 7.0k/11.5k/18.1k，调用次数从 8.5 增到 14/24/41。
tau=0.50 出现病态：首次预测即错误停止（hit 0.261，1.0 次调用）。

**3B**：LAET 的四个 f 值下 quantile 仍不优于 L2；DARTH 的 P2（1.5× interval）
在两种 loss 下 hit rate 最高，P1（0.5×）只增加调用次数。

**结论：** 非对称损失只有在允许移动 operating point（固定 multiplier 或更
频繁预测）时改善 hit rate；经过常规验证集校准后，它并不优于原 L2。收益是
预算中介的，而不是更好的预测机制。

## 4. 主要文件

```
exp01_feature_transfer/
  data/{summary_metrics,per_query_results,prediction_results,
        darth_checkpoint_predictions,quality_cost_curves,
        checkpoint_sensitivity,training_summary}.csv
  analysis/{statistical_tests,latency_summary,importance_*}.csv
  models/*.joblib
  figures/fig1..fig8
  reports/{experiment_report.md, experiment_report_zh.md,
           source_audit.md, research_questions.md, reproduce.md}
exp02_error_analysis/
  data/{per_query,per_checkpoint,oracle_costs}.csv, failure_sets.json
  analysis/{error_summary,failure_overlap,prediction_error_summary,
            training_size_summary,distribution_summary,
            recall_target_summary}.csv
  figures/fig1..fig8
  audit/method_audit.md
  reports/{experiment_report.md, findings_summary.md, findings_summary_zh.md}
  reproducibility/{run_commands.md, environment.json, validation_results.md}
exp03_asymmetric_loss/
  data/{metrics,per_query_results,prediction_traces,
        offline_predictions_3A_darth,quality_cost_curves_3A}.csv
  analysis/statistical_tests_3A.csv
  models/*.joblib
  figures/fig1..fig5
  reports/{report.md, report_zh.md, findings.md,
           source_audit.md, reproduce.md}
shared/
  cache/  ground_truth/  splits/  traces/  logs/
common/   全部可复用代码与驱动
```

## 5. 明确的局限与偏差

1. Python 单线程搜索：NDC 与原论文同口径，墙钟延迟不能与 C++ 实现直接比较。
2. 索引规模 992k（留出 8k Query），不是完整 1M；图只建一次。
3. DARTH 的 ef 在 992k GIST 上最高测到 2048，验证集 mean recall@100=0.9855
   （不是论文的 0.99）；这是记录在案的资源限制，不是隐藏的调参。
4. DARTH 的统一 checkpoint 模式是受控变体，与原始 adaptive interval 分开
   报告；D1≡D2 是特征集合的结构事实。
5. ANNiE 局部子图使用 Def. 7 近似（s=200，h=4）；特征向量包含 §4.2 的
   960 维 Query 向量。
6. Exp2 的训练集规模为 2,000（500/1,000/2,000 扫描）；Spec 的 5,000 上限
   受计算资源限制未跑。
7. 所有轨迹在 recall 达到目标后停止（k=10 满召回 / k=100 达到 0.99），
   并对超出轨迹的预算做保守回放；相关标记在 CSV 中保留。
