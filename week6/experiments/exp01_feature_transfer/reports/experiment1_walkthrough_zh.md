# 实验一方法说明：Feature Transfer（特征迁移消融）

> 本文档说明实验一**实际是如何执行的**：数据准备、索引构建、轨迹采集、特征构造、模型训练、停止规则、评估指标、统计检验、延迟测量与复现命令。  
> 文中数字来自 `data/`、`analysis/`、`shared/` 下的真实产物与 `common/` 中的实际代码。

---

## 0. 一句话概括

在 GIST1M 的同一张 HNSW 图上，固定 LAET/DARTH 的**预测目标、损失函数、训练数据、预测 checkpoint、搜索算法和停止规则**，只替换输入特征矩阵，比较六组：

- LAET：L0（原特征）、L1（原特征 ∪ ANNiE）、L2（仅 ANNiE）
- DARTH：D0（原 11 维）、D1（DARTH ∪ ANNiE）、D2（仅 ANNiE）

外加一个 `D0-original` 参考组（DARTH 原始自适应间隔）。目的是回答：

> ANNiE 的特征是否包含 LAET/DARTH 原特征没有的信息？预测变准后，能否转化为端到端搜索收益？

---

## 1. 实验设计

### 1.1 因子设计

| 方法 | 组 | 输入特征 | 特征维度 |
|---|---|---|---:|
| LAET | L0 | LAET 原特征 | 965 |
| LAET | L1 | LAET 原特征 ∪ ANNiE | 990 |
| LAET | L2 | 仅 ANNiE（完整向量） | 987 |
| DARTH | D0 | DARTH 原 11 维 | 11 |
| DARTH | D1 | DARTH ∪ ANNiE | 987 |
| DARTH | D2 | 仅 ANNiE（完整向量） | 987 |
| DARTH | D0-original | DARTH 原 11 维 + 原始自适应间隔 | 11 |

### 1.2 控制变量

在同一方法内部，以下全部固定，只改变特征矩阵：

- 同一张 HNSW 图、同一批 Query ID；
- 同一预测标签；
- 同一损失函数（LightGBM L2）；
- 同一模型结构与非损失超参数；
- 同一预测 checkpoint（LAET f=1,260；DARTH 统一网格）；
- 同一搜索轨迹与停止规则；
- 同一随机种子 42。

LAET 和 DARTH 之间不要求超参数相同，因为它们是两种不同方法；实验比较的是**方法内部的特征消融**。

### 1.3 为什么 D1 ≡ D2

DARTH 的 11 个特征：

```
step, ndc, inserts, d_first, d_closest, d_furthest,
avg, var, median, p25, p75
```

全部包含在 ANNiE basic 的 12 维特征中：

```
ndc, step, inserts, d_first, d_closest, d_furthest,
avg, var, median, p25, p75, recall_target
```

因此 `DARTH ∪ ANNiE` 去重后就是完整的 ANNiE 向量。**D1 与 D2 完全相同是特征集合的结构事实，不是实验得到的“平局”**。

---

## 2. 数据与索引准备

### 2.1 数据集

- 数据集：GIST1M；
- 规模：1,000,000 × 960；
- 距离：平方 L2；
- 文件：`D:\HNSW_Project\datasets\gist\gist.hdf5`。

### 2.2 索引基向量与 Query 划分

```
索引基向量：train[0 : 992000]          992,000 条
训练 Query 池：train[992000 : 999000]    7,000 条
验证 Query：train[999000 : 999500]         500 条
测试 Query：test[0 : 1000]               1,000 条
```

实验一实际使用：

```
训练：前 5,000 条（从 7,000 条池中取前 5,000）
验证：500 条
测试：1,000 条
```

三者 Query ID 互斥；训练和验证 Query 不在索引图中，避免泄漏。

### 2.3 Ground Truth

因为索引只用了 992,000 条基向量，而 ann-benchmarks 自带的 GT 是针对完整 1,000,000 条计算的，直接使用会包含被留出的 8,000 条向量，造成泄漏。因此：

- 对 992,000 条基向量重新计算精确 top-100 GT；
- 方法：分块 BLAS 矩阵乘法，`qblock=64`，避免一次性复制整个基向量矩阵；
- 输出：
  - `shared/ground_truth/gist_train_k100.npy`
  - `shared/ground_truth/gist_val_k100.npy`
  - `shared/ground_truth/gist_test_k100.npy`

实验一用 GT 的 top-10 计算 recall@10，top-1 用于 LAET 的 C1 标签。

### 2.4 HNSW 索引构建

使用 hnswlib 构建：

| 参数 | 值 |
|---|---:|
| 空间 | L2 |
| 基向量数 | 992,000 |
| 维度 | 960 |
| M | 16 |
| efConstruction | 200 |
| random_seed | 42 |
| 构建线程 | 14 |

构建后保存为：

```
D:/tmp/anns_wk6/gist_992000_M16_efc200.hnsw
```

然后把 hnswlib 的二进制索引解析成自己的 CSR 缓存：

- 读取 96 字节头部：`offset_level0, max_elements, n, size_elem, label_off, offset_data, maxlevel, entry, maxM, maxM0, M, mult, efc`；
- 读取 level-0 块：`uint32 count + maxM0 个 uint32 邻居`；
- 读取上层链接表；
- 输出：
  - `shared/cache/gist_idx992000_n992000_M16_efc200_graph.npz`
  - `shared/cache/gist_idx992000_n992000_M16_efc200_meta.json`

解析结果：

```
n = 992,000
M = 16
maxM0 = 32
max_level = 4
entry = 18,691
degree0 = 32
```

### 2.5 图解析校验

在 50 条官方测试 Query 上，用解析后的图和 hnswlib 自己的搜索做对比：

| 指标 | 值 |
|---|---:|
| efSearch | 100 |
| k | 10 |
| 与 hnswlib top-10 的平均 ID 重合率 | **0.9880** |
| 相对精确 GT 的平均 recall@10 | 0.8400 |
| 平均 NDC | 1,966.5 |

重合率接近 1，说明邻接表解析正确；剩余差异主要来自距离相同时的 tie-breaking。

---

## 3. 搜索轨迹采集

### 3.1 为什么要先采集轨迹

LAET、DARTH、ANNiE 都只决定**什么时候停止**，不改变 HNSW 的扩展顺序。因此可以先跑一次完整搜索，把每一步的 NDC、recall、特征状态存下来，之后所有模型、乘子、阈值、损失变体都可以在轨迹上**精确回放**，不需要重新跑图搜索。

这样做的优点：

- NDC/recall 与在线运行完全一致；
- 可以在同一批 Query 上比较几十个受控条件；
- 避免每次训练都重跑 992k 图搜索。

### 3.2 LAET 轨迹（无界束宽）

采集命令：

```bash
python -I collect_traces.py --stage exp13
```

LAET 使用 `ef=None` 的**无界束宽**搜索：

- 每个新访问的邻居都进入候选队列；
- 结果堆只保留 k=10 个；
- 只有学习到的预算/停止规则能结束搜索。

采集参数：

| 参数 | 值 |
|---|---:|
| k | 10 |
| max_ndc | 40,000 |
| fs | 256, 512, 630, 945, 1260, 1575, 2048 |
| 训练轨迹 max_steps | 20,000 |
| 验证/测试轨迹 max_steps | 6,000 |
| stop_floor | 1,260 |
| rt_grid | 0.90, 0.95, 0.99 |

对每条 Query 记录：

- `step_ndc`：每一步的 NDC；
- `step_recall`：每一步的 recall@10；
- `tail_f{f}`：在每个 f 首次达到时的 5 维 LAET 尾特征；
- `suffix`：在 f=1,260 时的 ANNiE basic+graph 后缀；
- `c1`：GT top-1 首次被距离计算时的 NDC；
- `cost_rt0.9/0.95/0.99`：recall@10 首次达到对应目标时的 NDC；
- `reason`：`natural` / `early` / `budget`。

训练轨迹的停止策略：

- 对训练 Query，搜索持续到 GT top-1 被访问且 NDC ≥ 2,048，保证 C1 可记录；
- 对验证/测试 Query，搜索在达到满 recall 后、且 NDC ≥ 1,260 时停止，保证轨迹足够长用于回放。

### 3.3 DARTH 轨迹（有界 efSearch）

DARTH 使用有界 efSearch：

- 候选队列容量为 ef；
- 只有比当前最差结果更近的节点才会被加入；
- 当最佳候选比结果集中最差的距离还远时，自然结束。

ef 在验证集上校准到目标 recall 0.99：

```json
"ef99_k10": {
  "ef": 2048,
  "val_mean_recall": 0.9862,
  "tuned_on": 200
}
```

即：在 992k GIST 上，ef 最大测到 2,048，验证集平均 recall@10 = 0.9862，**没有达到 0.99**。这是记录在案的资源限制，不是隐藏调参。

采集参数：

| 参数 | 值 |
|---|---:|
| k | 10 |
| ef | 2,048 |
| max_ndc | 40,000 |
| delta_obs | 25 |
| transfer_grid | 256, 512, 2048 + 1,260 到 12,000 的 24 个等距点 |
| need_transfer_suffix | True |
| h | 4 |
| s | 200 |
| rt_feat | 0.90 |

DARTH 轨迹记录三类数据：

1. **逐步轨迹**（验证/测试）：
   - `step_ndc`
   - `step_recall`
   - `step_feat`：每一步的 11 维 DARTH 特征

2. **训练观测**：
   - 每 25 NDC 记录一次
   - `obs_ndc`, `obs_recall`, `obs_feat`
   - 用于 `D0-original` 的训练（从 NDC 0 开始）

3. **统一 checkpoint 网格**：
   - `tr_ndc`
   - `tr_darth`：11 维 DARTH 特征
   - `tr_recall`：当前 recall@10
   - `tr_suffix`：ANNiE basic+graph 后缀（用于 D1/D2）

主实验的 DARTH 网格是：

```
256, 512, 2048
+ linspace(1260, 12000, 24)
```

也就是 1,260 到 12,000 之间 24 个等距点（间距约 467 NDC），加上 2,048，共 25 个主评估 checkpoint；256 和 512 供 checkpoint 敏感性实验使用。主评估通过 `min_ndc=1260` 跳过更浅的点。

> 注意：延迟测量脚本使用的是“每 200 NDC 预测一次”的在线循环，因此延迟实验中的 `calls` 与轨迹回放中的 `calls_mean` 不同。前者约 14.3 次，后者约 8.65 次。两者分别对应“在线实现”和“缓存网格评估”，不是同一个计数。

### 3.4 轨迹存储格式

所有轨迹以 `npz` 保存，核心约定：

```
ptr[i] : ptr[i+1]           第 i 条 Query 的逐步数组切片
tr_ptr[i] : tr_ptr[i+1]     第 i 条 Query 的 checkpoint 数组切片
obs_ptr[i] : obs_ptr[i+1]   第 i 条 Query 的训练观测切片
```

原始 Query 向量只存一份（在 split 中），训练时再拼接到特征矩阵前面，避免重复存储 960 维向量。

---

## 4. 特征工程

### 4.1 LAET 原特征（L0）

LAET 的特征来自 SIGMOD'20 Table 5：

| 特征 | 含义 |
|---|---|
| q | 960 维原始 Query 向量 |
| d_start | 入口/第一个 NN 的距离 |
| d_1st | 当前结果集中最近的距离 |
| d_10th | 当前结果集中第 10 近的距离 |
| d_1st/d_start | 比值 |
| d_10th/d_start | 比值 |

维度：

```
960 + 5 = 965
```

### 4.2 DARTH 原特征（D0）

DARTH 的 11 维特征来自 SIGMOD'25 Table 1：

| 特征 | 含义 |
|---|---|
| step | 当前扩展步数 |
| ndc | 当前距离计算次数 |
| inserts | 已插入结果集的节点数 |
| d_first | 第一个 NN 的距离 |
| d_closest | 当前 BSF k-set 中最近距离 |
| d_furthest | 当前 BSF k-set 中最远距离 |
| avg | BSF k-set 距离均值 |
| var | 方差 |
| median | 中位数 |
| p25 | 25 分位 |
| p75 | 75 分位 |

维度：

```
11
```

### 4.3 ANNiE basic 特征

ANNiE §4.2 的 basic 块：

| 特征 | 含义 |
|---|---|
| ndc | 当前 NDC |
| step | 当前步数 |
| inserts | 插入数 |
| d_first | 第一个 NN 距离 |
| d_closest | 最近距离 |
| d_furthest | 最远距离 |
| avg | 均值 |
| var | 方差 |
| median | 中位数 |
| p25 | 25 分位 |
| p75 | 75 分位 |
| recall_target | 目标 Recall（实验一为 0.90） |

维度：

```
12
```

### 4.4 ANNiE graph 特征

ANNiE §4.5 的图特征：

#### 局部子图构造（Def. 7 近似）

- 取当前已访问节点中距离最近的 `s=200` 个节点；
- 优先使用已弹出的节点（pops）；
- 如果不足 200 个，再从结果堆中补；
- 如果仍不足，再补出邻居；
- 在这个局部子图上计算图特征。

#### 图特征

| 特征 | 维度 | 含义 |
|---|---:|---|
| avg_clustering | 1 | 局部子图的平均聚类系数 |
| bsf_clustering | k=10 | 每个 BSF 节点的聚类系数 |
| h_hop_reachability | h=4 | 从 BSF 节点出发，每跳新可达节点数的平均 |

维度：

```
1 + 10 + 4 = 15
```

### 4.5 六组特征矩阵的精确组成

| 组 | 组成 | 维度 |
|---|---|---:|
| L0 | q(960) + LAET tail(5) | 965 |
| L1 | q(960) + LAET tail(5) + ANNiE basic 新增块(10) + graph(15) | 990 |
| L2 | q(960) + ANNiE basic(12) + graph(15) | 987 |
| D0 | DARTH 11 维 | 11 |
| D1 | q(960) + ANNiE basic(12) + graph(15)（DARTH 11 维是 basic 子集，去重后即完整 ANNiE） | 987 |
| D2 | 同 D1 | 987 |

L1 中“ANNiE basic 新增块”的 10 维是：

```
ndc, step, inserts, d_furthest, avg, var,
median, p25, p75, recall_target
```

它去掉了与 LAET tail 重复的 `d_first`、`d_closest`，但保留 `d_furthest` 和其他统计量。

### 4.6 特征泄漏检查

所有特征只来自**当前搜索状态**：

- 当前结果堆 / BSF k-set；
- 当前 NDC、step、inserts；
- 当前局部子图；
- Query 向量和 Recall 目标。

不使用：

- GT 标签；
- 未来节点；
- 最终停止代价；
- 任何事后统计量。

标签（C1、current recall）由同一轨迹计算，但只作为监督信号，不作为输入。

---

## 5. 模型训练

### 5.1 LAET 训练

#### 标签

```
y = log2(max(C1, 1))
```

其中 `C1` 是 GT top-1 首次被距离计算时的 NDC。

使用 `log2` 是因为 LAET 论文在 log 空间建模，预测后再用 `2^pred` 还原为 NDC。

#### 训练样本

- 每条训练 Query 在 f=1,260 处取**一行**；
- 5,000 条训练 Query，可用 4,880 行；
- 剩余 120 条因为 C1 在轨迹内未触及而被剔除。

#### LightGBM 参数

| 参数 | 值 |
|---|---:|
| objective | regression（L2） |
| n_estimators | 100 |
| learning_rate | 0.2 |
| num_leaves | 31 |
| min_child_samples | 20 |
| subsample | 1.0 |
| colsample_bytree | 1.0 |
| n_jobs | 4 |
| random_state | 42 |

#### `pred_max`

每组用自己的训练标签最大值计算：

```
pred_max = 2^(max y)
```

用于预算裁剪。

### 5.2 DARTH 训练（Mode A）

#### 标签

```
y = current recall@10 = hit / k
```

其中 `hit` 是当前结果集与 GT top-10 的交集大小。

#### 训练样本

- 来自 `tr_*` 统一 checkpoint 网格；
- 每条 Query 均匀抽取最多 20 行（`max_per_query=20`）；
- 只保留 `tr_ndc >= 1260`；
- 可用训练行数：64,783。

#### LightGBM 参数

| 参数 | 值 |
|---|---:|
| objective | regression（L2） |
| n_estimators | 100 |
| learning_rate | 0.1 |
| num_leaves | 31 |
| min_child_samples | min(1000, max(20, N/100)) |
| subsample | 1.0 |
| colsample_bytree | 1.0 |
| n_jobs | 4 |
| random_state | 42 |

#### Mode A 的含义

主实验的 D0/D1/D2 不使用 DARTH 原始的自适应间隔，而是：

- 在统一的 25 个 checkpoint 上**一次性批量预测**；
- 每条 Query 的预测与“是否停止”无关；
- 因此 D0/D1/D2 看到的是**完全相同的搜索状态**。

这消除了“模型改变预测 → 预测改变下一次调用时机 → 调用时机改变状态”的反馈回路，是特征消融所必需的控制。

### 5.3 D0-original 参考组

`D0-original` 不是六组受控实验的一部分，而是 DARTH 原始方法的参考：

- 训练观测从 NDC 0 开始；
- 每 25 NDC 记录一次；
- 训练行数 3,499,776；
- 推理时使用原始自适应间隔：

```
pi = mpi + (ipi - mpi) * (R_t - pred)
```

其中：

```
d_bar = 训练 Query 首次达到 R_t 的平均 NDC
ipi = d_bar / 2 = 1532.70
mpi = d_bar / 10 = 306.54
```

下一次调用条件：

```
NDC - last_NDC >= pi
```

停止条件：

```
pred >= R_t
```

该组单独报告，不与受控的 D0/D1/D2 混合。

---

## 6. 停止规则与评估

### 6.1 LAET 预算与停止

模型输出 `p = log2(cost)`，先还原：

```
pred_ndc = 2^max(p, 0)
```

在 500 条验证 Query 上校准乘子 `m`：

- 候选乘子：`geomspace(0.05, 200, 80)`，共 80 个；
- 选择“使验证集平均 recall 达到 0.90 的最小乘子”；
- 不使用测试集调参。

预算：

```
B = max(f, min(2^max(p,0) * m, pred_max))
```

其中 `f=1260`，`pred_max` 是该组训练标签的最大值还原后的 NDC。

停止：

- 在轨迹上找到第一个 `step_ndc >= B` 的位置；
- 返回该位置的 recall；
- 如果 B 超过轨迹终点：
  - 若轨迹已经达到满 recall，则按 `stop_NDC = B` 保守记账；
  - 否则标记 `truncated=True`。

**重要：** LAET 的乘子是按**平均 recall**校准到 0.90，不是按“90% 的 Query 达标”校准。因此 L0 的 mean recall = 0.904，但 hit rate 只有 0.499。这不是矛盾，而是两种指标的定义不同。

### 6.2 DARTH 停止（Mode A）

在每个 checkpoint：

```
pred = model.predict(features)
```

停止规则：

```
第一个 pred >= R_t (0.90) 的 checkpoint
```

如果没有任何 checkpoint 的预测达到 0.90，则搜索自然结束。

`calls` 的定义：

- 从第一个被评估的 checkpoint 到停止 checkpoint（含）之间的 checkpoint 数；
- D0 平均 8.65 次。

### 6.3 质量指标

| 指标 | 定义 |
|---|---|
| mean recall | 所有测试 Query 的 recall@10 平均值 |
| median recall | recall 中位数 |
| hit rate | 最终 `recall >= 0.90` 的 Query 比例 |
| failure rate | `1 - hit rate` |
| avg NDC | 平均停止 NDC |
| median NDC | 停止 NDC 中位数 |
| P95 / P99 NDC | 停止 NDC 的 95/99 分位，衡量尾部成本 |

### 6.4 成本指标

| 指标 | 定义 |
|---|---|
| oracle_NDC | 在**该方法自己的轨迹**上，recall@10 首次达到 0.90 的 NDC |
| oracle_attainable | oracle_NDC 是否有限；False 表示轨迹内不可达 |
| extra_NDC | `max(0, stop_NDC - oracle_NDC)` |
| relative_extra_NDC | `extra_NDC / oracle_NDC` |
| extra_ndc_mean_hit | 只在达标且 oracle 有限的 Query 上平均 |
| truncated | 预算超过轨迹终点时标记 |

注意：oracle 是**方法特定**的。LAET 的 oracle 在无界束宽轨迹上计算，DARTH 的 oracle 在 ef=2048 的有界轨迹上计算，两者不是同一个理论最优值。

### 6.5 预测误差指标

#### LAET

- `native_signed_mean`：`mean(2^pred - C1)`；
- `native_mae`：`mean(|2^pred - C1|)`；
- `native_mape`：`mean(|2^pred - C1| / C1)`；
- `native_qerror`：`mean(max(pred/true, true/pred))`；
- `native_over_rate` / `native_under_rate`：高估/低估比例；
- `abs_log2_cost_error`：`|log2(2^pred) - log2(C1)|`，用于配对统计。

#### DARTH

- `query_mean_abs_recall_error`：每条 Query 在所有 checkpoint 上 `|pred recall - actual recall|` 的平均；
- 每个 checkpoint 还记录：
  - `predicted_recall`
  - `actual_recall`
  - `recall_prediction_error`
  - `false_positive_stop`：预测 ≥0.90 但实际 <0.90，假停止；
  - `false_negative_continue`：预测 <0.90 但实际 ≥0.90，继续过久。

LAET 和 DARTH 的预测误差单位不同，**不能放在同一误差轴上比较**；实验只在方法内部比较。

### 6.6 配对统计

- 按 Query ID 配对；
- 对差值 `a_i - b_i` 做 2,000 次 Bootstrap；
- 取 2.5% 和 97.5% 分位作为 95% CI；
- 同时报告 Cliff's delta 作为非参数效应量。

判决规则：

- 95% CI 在预测方向上不跨 0 → supported；
- 跨 0 → inconclusive（不是“证明无差异”）；
- 显著但方向相反 → contradicted。

### 6.7 质量-成本曲线

用于画 fig2：

#### LAET

- 乘子扫描：`0.1` 到 `40`，40 个几何间隔点；
- 每个乘子重放预算；
- 预算上限裁剪到 40,000 NDC；
- 记录 mean recall、hit rate、mean NDC。

#### DARTH

- 停止阈值扫描：`0.50` 到 `0.995`，30 个线性点；
- 每个阈值重放固定网格；
- 记录 mean recall、hit rate、mean NDC。

---

## 7. 延迟测量

延迟不是在轨迹上估的，而是**真实在线循环**跑 30 条测试 Query：

```bash
python -I timing.py
```

### 7.1 LAET

- 在 NDC ≥ 1,260 时提取一次特征；
- 预测一次；
- 之后只检查 `NDC >= budget`；
- 记录：
  - total ms
  - feature ms
  - predict ms
  - NDC
  - recall

### 7.2 DARTH

- 从 NDC 1,260 开始；
- 每 200 NDC 提取一次特征并预测；
- 记录总调用次数、特征时间、预测时间、总时间。

### 7.3 实测结果

| 方法 | 组 | 平均总延迟 | 特征时间 | 预测时间 | 调用次数 | 平均 NDC |
|---|---|---:|---:|---:|---:|---:|
| LAET | L0 | 34.0 ms | 0.12 ms | 2.29 ms | 1.0 | 3,410 |
| LAET | L1 | 29.2 ms | 4.25 ms | 1.26 ms | 1.0 | 3,147 |
| LAET | L2 | 32.6 ms | 4.60 ms | 1.43 ms | 1.0 | 3,276 |
| DARTH | D0 | 65.7 ms | 9.41 ms | 19.04 ms | 14.3 | 4,064 |
| DARTH | D1 | 182.7 ms | 104.57 ms | 28.02 ms | 19.7 | 5,188 |
| DARTH | D2 | 183.5 ms | 104.92 ms | 28.21 ms | 19.7 | 5,188 |

关键观察：

- LAET 每条 Query 只提取一次特征，图特征总耗时约 4–5 ms；
- DARTH 每个 checkpoint 都提取特征，图特征总耗时约 105 ms/Query，是 D0 的约 11 倍；
- 总延迟 D1/D2 约为 D0 的 2.8 倍。

---

## 8. 完整执行流程

以下命令与实际 `reproduce.md` 一致：

```bash
PY="D:/大学/CUCS/CMCS5720/week5/.venv/Scripts/python.exe"
cd "D:/大学/CUCS/CMCS5720/week6/experiments/common"

# 1. 精确 GT + 划分
$PY -I prepare_gist.py --stage gt

# 2. 构建 HNSW（hnswlib 环境）
"D:/HNSW_Project/venv/Scripts/python.exe" build_hnswlib.py

# 3. 解析为 CSR 缓存
$PY -I parse_hnswlib.py

# 4. 校验解析图
$PY -I validate_graph.py

# 5. 采集 k=10 轨迹
$PY -I collect_traces.py --stage exp13

# 6. 运行实验一
$PY -I exp01_run.py --n-train 5000 --n-val 500 --n-test 1000 --seed 42 --sensitivity

# 7. 延迟测量
$PY -I timing.py

# 8. 图表与表格
$PY -I exp01_figures.py
$PY -I report_tables.py
```

---

## 9. 实际配置一览

| 项目 | 值 |
|---|---|
| 数据集 | GIST1M |
| 维度 | 960 |
| 距离 | 平方 L2 |
| 索引基向量 | 992,000 |
| 训练 / 验证 / 测试 | 5,000 / 500 / 1,000 |
| k | 10 |
| R_t | 0.90 |
| LAET checkpoint | f = 1,260 NDC |
| DARTH 主评估网格 | 1,260–12,000（24 个等距点）+ 2,048；256/512 仅用于敏感性 |
| HNSW M | 16 |
| efConstruction | 200 |
| DARTH ef | 2,048 |
| LAET LightGBM | 100 轮，lr=0.2，31 叶 |
| DARTH LightGBM | 100 轮，lr=0.1，31 叶 |
| 随机种子 | 42 |
| LAET 可用训练行 | 4,880 |
| DARTH 可用训练行 | 64,783 |
| D0-original 训练行 | 3,499,776 |

---

## 10. 控制变量与防泄漏

### 10.1 已控制

- 同一张图、同一批 Query ID；
- 同一搜索轨迹（模型只改停止点，不改扩展顺序）；
- 同一预测标签；
- 同一损失函数；
- 同一模型结构与超参数（方法内部）；
- 同一 checkpoint；
- 同一随机种子；
- 验证集只用于乘子校准，测试集不参与任何调参；
- 训练/验证/测试 Query ID 互斥；
- 训练和验证 Query 不在索引图中。

### 10.2 已检查

- 特征只使用当前状态，不使用 GT、未来节点或最终代价；
- 图解析与 hnswlib 原生搜索的 top-10 重合率 0.988；
- 轨迹回放与在线运行在 30 条 Query 上对齐；
- D1≡D2 是结构事实，已在报告中明确说明，不当作实证结果。

### 10.3 已知偏差

1. 索引规模 992k，不是完整 1M；
2. 图只建一次，seed=42；
3. Python 单线程搜索，NDC 可比，墙钟不可直接与 C++ 比；
4. DARTH ef 最高 2,048，验证集 mean recall@10 = 0.9862，未达到论文的约 0.99；
5. ANNiE 局部子图使用 Def. 7 近似（s=200，h=4）；
6. 只在 f=1,260 对 L1/L2、D1/D2 做完整比较；checkpoint 敏感性只重训 L0 和 D0，因为其他组的图特征没有在每个 f 上缓存；
7. 29 条测试 Query 和 120 条训练 Query 的 C1 在轨迹内不可得，不参与预测误差比较；
8. 训练集 5,000 条，不是完整 7,000 条。

---

## 11. 输出物

| 文件 | 内容 |
|---|---|
| `data/summary_metrics.csv` | 六组 + D0-original 的汇总指标 |
| `data/per_query_results.csv` | 每条 Query、每个组的 recall/NDC/oracle/extra |
| `data/prediction_results.csv` | LAET 的预测 log2 cost、pred_ndc、真实 C1、预算、乘子 |
| `data/darth_checkpoint_predictions.csv` | DARTH 每个 checkpoint 的预测/实际 recall、FP/FN 标记 |
| `data/quality_cost_curves.csv` | LAET 乘子扫描、DARTH 阈值扫描 |
| `data/checkpoint_sensitivity.csv` | f=256/512/1260/2048 的敏感性结果 |
| `analysis/statistical_tests.csv` | 配对 Bootstrap、95% CI、Cliff's delta |
| `analysis/importance_*.csv` | 每组 LightGBM gain importance |
| `analysis/latency_summary.csv` | 30 条 Query 的延迟分解 |
| `models/` | 每组训练好的 joblib 模型 |
| `figures/` | 图 1–8 |
| `reports/` | 中英文报告、研究问题、源码审计、复现说明 |

---

## 12. 读结果时最容易混淆的几点

1. **mean recall ≠ hit rate**  
   L0 的 mean recall=0.904，但 hit rate=0.499，因为乘子按平均 recall 校准，不是按达标率校准。

2. **NDC ≠ 墙钟延迟**  
   NDC 是距离计算次数，跨语言可比；延迟是 Python 实现的实际时间，不能直接与论文 C++ 比较。

3. **L0/L1/L2 是特征组，不是损失**  
   实验三里的 L0–L4 是损失配置；实验一的 L0–L2 是特征组。读表时先看实验编号。

4. **D0/D1/D2 是特征组，不是训练分布或损失**  
   实验二的 D0/D2 是训练分布；实验三的 D0–D4 是 DARTH 损失配置。

5. **D1 ≡ D2 不是“打平”**  
   是特征集合去重后完全相同，属于结构事实。

6. **native MAE 不能跨方法比较**  
   LAET 的 native MAE 是 NDC 空间；DARTH 的 native error 是 recall 空间，单位不同。

7. **oracle 不是全局最优**  
   oracle 是在该方法自己的轨迹上、禁用学习停止规则后，首次达到目标 recall 的 NDC；LAET 和 DARTH 的 oracle 轨迹不同。

---

## 13. 一句话总结

> 先用 hnswlib 建一张 992k 的 GIST HNSW 图，重算精确 top-100 GT；然后在同一批训练/验证/测试 Query 上采集 LAET 的无界束宽轨迹和 DARTH 的有界 ef=2048 轨迹；在固定的标签、损失、checkpoint 和停止规则下，只替换六组特征矩阵训练 LightGBM；用验证集校准 LAET 乘子，在测试集上做逐 Query 回放、配对 Bootstrap 和延迟测量，从而把“特征是否提升预测”和“预测是否提升端到端搜索”分开检验。

---

*文件位置：`D:/大学/CUCS/CMCS5720/week6/experiments/exp01_feature_transfer/reports/experiment1_walkthrough_zh.md`*
