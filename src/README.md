# src/ —— 实验框架

## 已完成并验证

| 模块 | 作用 | 状态 |
|---|---|---|
| `core/graph.py` | faiss 建 HNSW → 导出 CSR 邻接表 → 缓存到 `cache/` | ✅ |
| `core/search.py` | 可插桩搜索：统计 NDC、逐时刻特征、任意停止规则；支持 `ef=int`（标准/DARTH）与 `ef=None`（无限束宽/LAET/ANNiE-S） | ✅ |
| `core/gt.py` | 数据集加载、精确 GT、训练查询留出 | ✅ |
| `core/metrics.py` | recall / RQUT / P99 / Worst-1% / RDE / NRS / MAPE / Q-error / pMAPE / coverage | ✅ |
| `e0_validate.py` | 保真性验证（与 faiss 对比） | ✅ 已验证 |

## 运行

```powershell
cd "D:\大学\CUCS\CMCS5720\week5"
& ".\.venv\Scripts\python.exe" -I src\e0_validate.py 200000
```

当前结果（SIFT 20 万子集，M=16, efC=200）：

| ef | 自实现 recall@10 | faiss recall@10 | 平均 NDC | 速度 |
|---|---|---|---|---|
| 50 | 0.9750 | 0.9750 | 951 | 6.6 ms/q |
| 100 | 0.9955 | 0.9955 | 1589 | 11.3 ms/q |
| 200 | 0.9995 | 0.9995 | 2681 | 23.0 ms/q |

无限束宽模式（`ef=None` + `max_ndc`）冒烟测试通过：NDC 精确等于预算，停止原因 `budget`。

## 设计要点

- **距离**：`d² = ||x||² − 2·q·x + ||q||²`，用 `base[ids] @ q`（BLAS gemv）**整批**计算邻居距离，比逐点 `np.linalg.norm` 快 3–5×。
- **visited 集合**：`int32` stamp 数组复用，每查询 O(1) 重置（不用 python set）。
- **top-k 追踪**：额外维护 k 大小的堆，插入/淘汰时 O(1) 更新命中数，因此**任意时刻的 recall 都是 O(1) 可得**——这是生成 `C(q)` 标签的关键。
- **内存**：不复制 base；faiss 索引导出后立即释放；`stamp` 对 10M 节点仅 40MB。

## 待补（按计划 11-06 ~ 11-07）

- `core/parallel.py`：joblib 多进程驱动（8 进程 → 约 8× 加速）
- `methods/baselines.py`：fixed-ef 扫描、REM、静态预算、Oracle(C(q))
- `methods/laet.py` / `darth.py` / `annie.py`
- `exp/e1_hardness.py` … `e7_scale.py`

## 已知偏差（需在报告中写明）

- 邻居距离是**整批**计算的（每次扩展最多 2M 个），因此停止点相对预算最多**提前一个批次**（≤32 次距离计算）。三种方法走同一循环，偏差相同，不影响横向对比。
- Python 实现的绝对延迟不可与论文 C++ 比，主指标用 NDC。
