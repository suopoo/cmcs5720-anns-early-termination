# Week 6 experiments — unified workspace

**Read [FINAL_SUMMARY.md](FINAL_SUMMARY.md) (consolidated) or
[SUMMARY.md](SUMMARY.md) first** for the three experiments' actual results,
hypothesis verdicts and limitations.

All code, data, intermediate traces, models, figures and reports for the
three experiments live under this folder (D: only; nothing is written to C:).

```
experiments/
├── README.md
├── common/                     shared implementation (one source of truth)
│   ├── anns_exp/
│   │   ├── core/               instrumented HNSW search, graph, GT, paths
│   │   ├── features.py         LAET / DARTH / ANNiE feature sets + unions
│   │   ├── trace.py            trace collection (NDC, recall, features)
│   │   ├── simulate.py         exact replay of stopping rules on traces
│   │   ├── gbdt.py             LightGBM training helpers
│   │   ├── stats.py            paired bootstrap, permutation overlap tests
│   │   ├── exp_common.py       shared I/O / metrics helpers
│   │   ├── plotting.py         validated light palette for matplotlib
│   │   └── experiments...      (drivers below)
│   ├── prepare_gist.py         GT + split preparation
│   ├── build_hnswlib.py        multi-threaded GIST index build
│   ├── parse_hnswlib.py        hnswlib -> CSR graph cache
│   ├── collect_traces.py       shared search-trace collection
│   ├── exp01_run.py            Experiment 1 driver
│   ├── exp02_run.py            Experiment 2 driver
│   ├── exp03_run.py            Experiment 3 driver
│   ├── exp01_figures.py / exp02_figures.py / exp03_figures.py
│   └── validate_graph.py       parsed-graph vs hnswlib validation
├── shared/
│   ├── cache/                  GIST hnswlib index + parsed CSR cache
│   ├── ground_truth/           exact top-100 GT for train/val/test
│   ├── splits/                 query split definition (JSON)
│   ├── traces/                 cached search trajectories (npz)
│   └── logs/                   preparation / validation logs
├── exp01_feature_transfer/
│   ├── configs/ data/ models/ analysis/ figures/ reports/ logs/
├── exp02_error_analysis/
│   ├── configs/ audit/ data/ analysis/ figures/ reports/
│   ├── reproducibility/ logs/
└── exp03_asymmetric_loss/
    ├── configs/ data/ models/ analysis/ figures/ reports/ logs/
```

## Environment

* Dataset: GIST1M (`D:\HNSW_Project\datasets\gist\gist.hdf5`)
* Python: `D:\大学\CUCS\CMCS5720\week5\.venv\Scripts\python.exe` (3.14)
* Index: hnswlib M=16, efConstruction=200, 992,000 vectors (8,000 held-out
  queries); parsed to `shared/cache/gist_idx992000_n992000_M16_efc200_graph.npz`
* Splits: train 7,000 (first 5,000 used), validation 500, test 1,000 official
  queries; all disjoint.

## Runtime notes

* The search loop is single-threaded Python; NDC is the portable cost and
  wall-clock is implementation-specific.
* Unlimited-beam traces are stopped when the ground-truth top-k set is fully
  recovered (k=10) or recall@100 reaches 0.99 (Experiment 2), with hard caps
  of 10,000-20,000 expansions and 40,000-80,000 NDC.  A budget beyond the
  trace end is replayed as `stop_NDC = budget` once full recall is reached
  and is flagged separately; this is documented in each report.
* DARTH transfer training/evaluation uses one common checkpoint grid of
  1,260 → 12,000 NDC (24 points) so D0/D1/D2 see identical states.  The
  original adaptive-interval reference still uses every expansion.

## Status

All three experiments executed; see `FINAL_SUMMARY.md`.  D: free space after
completion: ~32 GB; this folder is ~7.5 GB (including the 3.6 GB base array
and the 3.9 GB raw hnswlib index in `shared/cache/`).

## Reproduce

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

Each experiment's `reports/reproduce.md` holds the exact commands and
configuration actually used.
