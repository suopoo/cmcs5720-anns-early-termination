# 第 6 周实验 —— 统一工作区

**请先阅读 [FINAL_SUMMARY.md](FINAL_SUMMARY.md)（汇总版）或
[SUMMARY.md](SUMMARY.md)**，以了解三个实验的实际结果、假设判定与局限。

三个实验的全部代码、数据、中间轨迹、模型、图表和报告都存放在本文件夹下
（仅在 D 盘；不向 C 盘写入任何内容）。

```
experiments/
├── README.md
├── common/                     共享实现（唯一的代码来源）
│   ├── anns_exp/
│   │   ├── core/               带插桩的 HNSW 搜索、图、GT、路径
│   │   ├── features.py         LAET / DARTH / ANNiE 特征集及其并集
│   │   ├── trace.py            轨迹采集（NDC、recall、特征）
│   │   ├── simulate.py         在轨迹上精确回放停止规则
│   │   ├── gbdt.py             LightGBM 训练辅助函数
│   │   ├── stats.py            配对 bootstrap、置换重合检验
│   │   ├── exp_common.py       共享 I/O / 指标辅助函数
│   │   ├── plotting.py         经过校验的浅色调色板（matplotlib）
│   │   └── experiments...      （下方为各实验驱动脚本）
│   ├── prepare_gist.py         GT + 划分准备
│   ├── build_hnswlib.py        多线程构建 GIST 索引
│   ├── parse_hnswlib.py        hnswlib -> CSR 图缓存
│   ├── collect_traces.py       共享搜索轨迹采集
│   ├── exp01_run.py            Experiment 1 驱动脚本
│   ├── exp02_run.py            Experiment 2 驱动脚本
│   ├── exp03_run.py            Experiment 3 驱动脚本
│   ├── exp01_figures.py / exp02_figures.py / exp03_figures.py
│   └── validate_graph.py       解析图 vs hnswlib 校验
├── shared/
│   ├── cache/                  GIST hnswlib 索引 + 解析后的 CSR 缓存
│   ├── ground_truth/           train/val/test 的精确 top-100 GT
│   ├── splits/                 Query 划分定义（JSON）
│   ├── traces/                 缓存的搜索轨迹（npz）
│   └── logs/                   准备 / 校验日志
├── exp01_feature_transfer/
│   ├── configs/ data/ models/ analysis/ figures/ reports/ logs/
├── exp02_error_analysis/
│   ├── configs/ audit/ data/ analysis/ figures/ reports/
│   ├── reproducibility/ logs/
└── exp03_asymmetric_loss/
    ├── configs/ data/ models/ analysis/ figures/ reports/ logs/
```

## 环境

* 数据集：GIST1M（`D:\HNSW_Project\datasets\gist\gist.hdf5`）
* Python：`D:\大学\CUCS\CMCS5720\week5\.venv\Scripts\python.exe`（3.14）
* 索引：hnswlib M=16，efConstruction=200，992,000 个向量（8,000 条留出
  Query）；解析为 `shared/cache/gist_idx992000_n992000_M16_efc200_graph.npz`
* 划分：train 7,000（使用前 5,000）、validation 500、test 1,000 条官方
  Query；三者互不重叠。

## 运行时说明

* 搜索循环为单线程 Python；NDC 是可移植的代价指标，墙钟时间与具体实现
  相关。
* 无限束（unlimited-beam）轨迹在 ground-truth top-k 集合被完全恢复（k=10）
  或 recall@100 达到 0.99（Experiment 2）时停止，硬上限为
  10,000–20,000 次扩展和 40,000–80,000 NDC。若预算超出轨迹末端，则在达到
  满召回后以 `stop_NDC = budget` 回放，并单独标记；这一点在各报告中均有
  说明。
* DARTH 迁移训练/评估使用统一的 checkpoint 网格 1,260 → 12,000 NDC
  （24 个点），使 D0/D1/D2 看到完全相同的状态。原始的 adaptive-interval
  参考实现仍使用每一次扩展。

## 状态

三个实验均已执行完毕；见 `FINAL_SUMMARY.md`。完成后 D 盘剩余空间约
32 GB；本文件夹约 7.5 GB（包括 `shared/cache/` 中 3.6 GB 的基础数组和
3.9 GB 的原始 hnswlib 索引）。

## 复现

```bash
PY="D:/大学/CUCS/CMCS5720/week5/.venv/Scripts/python.exe"
cd "D:/大学/CUCS/CMCS5720/week6/experiments/common"

$PY -I prepare_gist.py --stage gt          # ground truth + splits
$PY -I build_hnswlib.py                    # (run with the hnswlib venv)
$PY -I parse_hnswlib.py
$PY -I validate_graph.py

$PY -I collect_traces.py --stage exp13
$PY -I collect_traces.py --stage exp2
$PY -I exp01_run.py
$PY -I exp02_run.py --phase all --n-train 2000 --n-val 500 --n-test 1000
$PY -I exp03_run.py --phase all
$PY -I exp01_figures.py
$PY -I exp02_figures.py
$PY -I exp03_figures.py
```

各实验的 `reports/reproduce.md` 中记录了实际使用的精确命令与配置。
