# Reproducing Experiment 1

All paths are on D:.  Python is the week-5 virtual environment:
`D:\大学\CUCS\CMCS5720\week5\.venv\Scripts\python.exe`.

## 1. Shared environment (built once)

```bash
PY="D:/大学/CUCS/CMCS5720/week5/.venv/Scripts/python.exe"
cd "D:/大学/CUCS/CMCS5720/week6/experiments/common"

# exact top-100 ground truth for the 992k index base and the three splits
$PY -I prepare_gist.py --stage gt

# GIST index (M=16, efc=200) with hnswlib (ASCII output path needed by the
# C++ save routine), then parse into the instrumented CSR cache
"D:/HNSW_Project/venv/Scripts/python.exe" build_hnswlib.py
$PY -I parse_hnswlib.py
$PY -I validate_graph.py
```

Expected shared artefacts:

```
shared/cache/gist_base992000.npy
shared/cache/gist_992000_M16_efc200.hnsw          (D:/tmp/anns_wk6)
shared/cache/gist_idx992000_n992000_M16_efc200_graph.npz
shared/ground_truth/gist_{train,val,test}_k100.npy
shared/splits/gist_splits.json
shared/traces/gist_k10_{laet,darth}_{train,val,test}.npz
```

## 2. Traces

```bash
$PY -I collect_traces.py --stage exp13
```

## 3. Experiment

```bash
$PY -I exp01_run.py --n-train 5000 --n-val 500 --n-test 1000 --seed 42 \
    --sensitivity
$PY -I timing.py                 # measures latency on 30 test queries
$PY -I exp01_figures.py
$PY -I report_tables.py
```

## 4. Outputs

| File | Content |
|---|---|
| `data/summary_metrics.csv` | group-level quality / cost / error summary |
| `data/per_query_results.csv` | one row per (group, test query) |
| `data/prediction_results.csv` | LAET native predictions vs true C1 |
| `data/darth_checkpoint_predictions.csv` | every DARTH checkpoint |
| `data/quality_cost_curves.csv` | multiplier / threshold sweeps |
| `data/checkpoint_sensitivity.csv` | f ∈ {256,512,1260,2048} |
| `analysis/statistical_tests.csv` | paired bootstrap + Cliff's delta |
| `analysis/importance_*.csv` | LightGBM gain importance |
| `analysis/latency_summary.csv` | measured end-to-end latency |
| `models/` | joblib LightGBM models |
| `figures/` | figures 1-5, 8 |
| `reports/experiment_report.md`, `experiment_report_zh.md` | reports |

## Notes

* The test set was never used for tuning: multipliers are calibrated on the
  500 validation queries; the quantile tau in Experiment 3 is selected on
  validation native MAE.
* `D1 == D2` for DARTH by construction (its 11 features are a subset of
  ANNiE's basic block); this is stated in the report rather than presented
  as an empirical tie.
