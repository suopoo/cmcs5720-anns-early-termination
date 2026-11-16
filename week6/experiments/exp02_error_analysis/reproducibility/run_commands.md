# Reproducing Experiment 2

Environment: `D:\大学\CUCS\CMCS5720\week5\.venv\Scripts\python.exe`
(Python 3.14, numpy 2.5, LightGBM 4.7).  Shared GIST environment and graph
are the same as Experiment 1 (see `exp01_feature_transfer/reports/reproduce.md`).

```bash
PY="D:/大学/CUCS/CMCS5720/week5/.venv/Scripts/python.exe"
cd "D:/大学/CUCS/CMCS5720/week6/experiments/common"

# k=100, R_t=0.90/0.95/0.99 traces (unlimited LAET/ANNiE + bounded DARTH)
$PY -I collect_traces.py --stage exp2 --n-train 2000 --n-val 500 --n-test 1000

# Phase 2: per-query instrumentation + failure overlap (main results)
$PY -I exp02_run.py --phase main --n-train 2000 --n-val 500 --n-test 1000

# Phase 3: training-set size (nested subsets, 3 seeds)
$PY -I exp02_run.py --phase size

# Phase 4: hard-query-enriched training distribution (D2)
$PY -I exp02_run.py --phase dist

# Phase 5: recall-target sensitivity 0.90/0.95/0.99
$PY -I exp02_run.py --phase rt

$PY -I exp02_figures.py
$PY -I report_tables.py
```

Outputs (all under `exp02_error_analysis/`):

| File | Content |
|---|---|
| `data/per_query.csv` | one row per (method, test query): oracle, predictions, budget, stop NDC, recall, hit, extra NDC |
| `data/per_checkpoint.csv` | every DARTH prediction with actual recall, FP-stop and FN-continue flags |
| `data/oracle_costs.csv` | method-specific oracle and attainability status |
| `data/failure_sets.json` | quality/efficiency failure query IDs |
| `analysis/error_summary.csv` | method-level error and cost summary |
| `analysis/prediction_error_summary.csv` | native/effective error distributions |
| `analysis/failure_overlap.csv` | Jaccard, conditional probabilities, stratified permutation p-values |
| `analysis/training_size_summary.csv` | 4 sizes x 3 seeds x 3 methods |
| `analysis/distribution_summary.csv` | D0 vs hard-enriched D2 |
| `analysis/recall_target_summary.csv` | R_t sensitivity |
| `figures/` | error distributions, CDFs, overlap, Jaccard heatmap, extra NDC |
| `reports/experiment_report.md`, `findings_summary.md` | final report and findings |

Configuration is saved in `configs/experiment_config.json`; the raw harness
parameters (ef, homing, splits) are in `shared/traces/meta.json` and
`shared/splits/gist_splits.json`.
