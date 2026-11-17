# 第 6 周 —— 三个实验：实际运行了什么、发现了什么

数据集：**GIST1M**，在 **992,000** 个向量（留出 8,000 条 Query）上构建
一个 hnswlib 索引（M=16，efConstruction=200）；960 维平方 L2 距离。Query
划分互不重叠：train 7,000（exp1/3 使用 5,000，exp2 使用 2,000）、
validation 500、官方 test 1,000。下文所有数字均来自本文件夹中的 CSV/JSON
产物；没有任何模拟数据或手工填入的数据。

## Experiment 1 —— 特征迁移（k=10，R_t=0.90，f=1,260）

共六组，标签/损失/checkpoint/搜索完全一致；只有特征矩阵不同。

| group | mean recall | hit rate | mean NDC | native MAE (NDC) | MAPE |
|---|---|---|---|---|---|
| L0 (LAET original) | 0.9040 | 0.499 | 4,073 | 895.8 | 0.916 |
| L1 (LAET ∪ ANNiE) | 0.8908 | 0.465 | 3,798 | **870.2** | **0.848** |
| L2 (ANNiE only) | 0.8943 | 0.468 | 3,792 | 894.3 | 0.885 |
| D0 (DARTH original) | 0.9242 | 0.537 | 5,013 | — | — |
| D1 (DARTH ∪ ANNiE) | 0.9214 | 0.515 | 5,085 | — | — |
| D2 (ANNiE only) | 0.9214 | 0.515 | 5,085 | — | — |
| D0-original (adaptive ipi) | 0.9250 | 0.546 | 4,632 | — | — |

主要发现：

1. **H1（LAET）在预测准确度上得到支持。** L1 vs L0 的绝对 log2 代价
   误差：Δ = −0.0715，95% CI [−0.1123, −0.0313]；L2 vs L0：
   Δ = −0.0467 [−0.0836, −0.0075]。L1 vs L2 无法定论
   [−0.0539, +0.0066]。
2. **准确度提升没有转化为质量提升。** L1 vs L0 的搜索 recall 为
   −0.0132 [−0.0189, −0.0074]，同时 NDC 为 −275 [−418, −114]。迁移后
   的模型花得更少，因而以略低的频率达标；H3（端到端提升）不被支持。
3. **DARTH 不受迁移影响。** D1 vs D0 的 Query 级平均绝对 recall 误差
   Δ = −0.00007 [−0.0016, +0.0016]，recall Δ = −0.0028
   [−0.0076, +0.0022]，NDC Δ = +72 [−220, +366]。D1 与 D2 在构造上
   完全相同，因为 DARTH 的 11 个特征是 ANNiE 基本特征块的子集（所有配对
   差值恰好为 0）。
4. **特征提取代价是真实存在的。** 在 30 条测试 Query 上实测：DARTH D0
   65.7 ms/Query，D1/D2 182.7/183.5 ms/Query（其中约 105 ms 仅为图特征
   提取）。LAET L0/L1/L2：34.0/29.2/32.6 ms（每条 Query 一次提取）。
5. **Checkpoint 敏感性基本平坦。** L0 在 f=256/512/1260/2048 上重新训练：
   hit rate 0.500/0.479/0.499/0.496。D0：0.530/0.535/0.546/0.563。
   说明结论并不是选择 f=1,260 造成的假象。

## Experiment 2 —— 逐 Query 预测误差（k=100，R_t=0.95）

| method | mean recall | target hit rate | mean NDC | rel. extra NDC | oracle attainable |
|---|---|---|---|---|---|
| LAET | 0.9562 | 0.676 | 12,150 | 0.799 | 0.980 |
| DARTH | 0.9608 | 0.644 | 12,691 | 1.063 | 0.926 |
| ANNiE-S | **0.9736** | **0.838** | 15,996 | 0.935 | 0.980 |

1. **失败具有强烈的非随机性，并且经过代价分层后依然存在。**
   Quality-failure 重合（均为 p=0.0005，按 oracle 代价分层置换）：
   LAET∩DARTH 225 条 Query（Jaccard 0.495），LAET∩ANNiE 122 条（0.335），
   DARTH∩ANNiE 149 条（0.404）；**有 118 条 Query 在三种方法下全部失败**。
   Efficiency 失败同样存在重合（Jaccard 0.129–0.434，p≤0.0015）。
2. **各方法在质量与代价之间的权衡方式不同。** ANNiE vs LAET：
   recall +0.0175 [+0.0152, +0.0198]，代价为 +3,847 NDC
   [+3,414, +4,287]；LAET vs DARTH：recall −0.0046
   [−0.0070, −0.0024]，NDC −541 [−923, −179]，relative extra NDC
   −0.334 [−0.441, −0.230]。
3. **训练集规模单调地带来帮助，ANNiE 受益最大。** 500→2000 条 Query
   （3 个 seed 均值）：ANNiE hit 0.769→0.838，DARTH 0.633→0.653，LAET
   0.651→0.676。
4. **Hard-query-enriched 训练（1,000 条 Query）提高 hit rate，但以 NDC
   为代价。** D0→D2：LAET 0.596→0.624，DARTH 0.654→0.724，ANNiE
   0.848→0.908；NDC 相应从 10,881→11,850、12,894→14,324、
   15,652→18,119。没有免费的午餐。
5. **Recall 目标的敏感性与方法相关。** R_t = 0.90/0.95/0.99：
   LAET hit 0.604/0.676/0.687；DARTH 0.703/0.644/0.584（随目标升高而
   退化）；ANNiE 0.854/0.838/0.703。

## Experiment 3 —— 非对称损失与预测时机（k=10，R_t=0.90）

**3A-LAET（A2：每个模型的 multiplier 在验证集上重新校准）**

| loss | tau | hit rate | mean NDC |
|---|---|---|---|
| L0 L2 | — | 0.499 | 4,073 |
| L1 quantile | 0.50 | 0.473 | 3,622 |
| L2 | 0.75 | 0.495 | 4,105 |
| L3 | 0.90 | 0.484 | 3,741 |
| L4 | 0.95 | 0.504 | 3,985 |

重新校准之后**没有有意义的收益**（L4 vs L0 recall
+0.0010 [−0.0046, +0.0063]；NDC −88 [−226, +58]）。若固定 L0 的
multiplier（A1），L4 在 12,530 vs 4,073 NDC 下达到 hit 0.818 vs
0.499 —— 也就是说，表面上的“收益”来自更大的有效预算，而不是更好的代价
估计。这是 LAET 的核心发现。

**3A-DARTH**

| loss | tau | hit rate | mean NDC | calls |
|---|---|---|---|---|
| D0 L2 | — | 0.546 | 4,632 | 8.5 |
| D1 | 0.50 | 0.261 | 1,543 | 1.0 |
| D2 | 0.25 | 0.681 | 6,958 | 14.0 |
| D3 | 0.10 | 0.809 | 11,516 | 24.2 |
| D4 | 0.05 | **0.876** | 18,094 | 41.0 |

更低的 tau 单调地降低高估、提高 hit rate，但 NDC/调用代价陡增。
tau=0.50 属于病态情况（预测值永远无法越过目标，因此 adaptive interval
不断增长，最终只发生一次调用）。

**3B 时机** —— LAET：在任何一个 f（630/945/1260/1575）上都没有稳定的
quantile 收益；DARTH：quantile 模型（tau=0.25）在每一种 interval 配置下
都比 L2 多出约 +0.13 hit，而 P2（1.5×ipi/mpi）在两种损失下都是最佳
单元格（hit 0.692 quantile vs 0.566 L2）。因此损失带来的收益不能由
interval 调参解释掉，但它是以 NDC 为代价换来的。

## 诚实的局限

* 使用的是 992k 规模的 GIST1M，而非完整的 1M；只有一个图 seed（42）。
* DARTH 的受限搜索使用 ef=2048，即测试过的最高值，达到 mean recall@100
  0.9855 —— 而非论文中的约 0.99。对 92.6% 的测试 Query 其 oracle 是可达
  的；其余 Query 已相应标记。
* Exp2 在 2,000 条 Query 上训练（而非 5,000），以保持轨迹采集过程可行。
* Python 搜索：NDC 可比；墙钟时间不能与论文的 C++ 实现直接比较。
  无限束轨迹在满召回（k=10）或 recall@100=0.99 时停止并受扩展上限约束；
  超出轨迹末端的预算被标记为截断。
* DARTH D1≡D2 是特征集合的结构性事实，而非经验发现。
* 迁移/quantile 的结论基于一个数据集和一个 GIST 图。

## 各文件位置

```
experiments/
  SUMMARY.md                      本文件
  README.md                       目录结构 / 环境 / 命令
  common/                         全部代码（唯一来源）
  shared/cache/                   GIST 基础 .npy、hnswlib 索引、解析后的 CSR
  shared/traces/                  全部 11 条缓存轨迹
  shared/ground_truth/, splits/   精确 GT 与 Query 划分
  exp01_feature_transfer/
    data/ analysis/ models/ figures/ reports/ logs/ configs/
  exp02_error_analysis/
    data/ analysis/ figures/ reports/ audit/ reproducibility/ logs/ configs/
  exp03_asymmetric_loss/
    data/ analysis/ models/ figures/ reports/ logs/ configs/
```
