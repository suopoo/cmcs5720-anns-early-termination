# Reproducing Experiment 3

Environment and shared GIST traces are the same as Experiments 1–2.

```bash
PY="D:/大学/CUCS/CMCS5720/week5/.venv/Scripts/python.exe"
cd "D:/大学/CUCS/CMCS5720/week6/experiments/common"

# 3A: five LAET losses and five DARTH losses, A1 fixed / A2 recalibrated
$PY -I exp03_run.py --phase 3a --n-train 5000 --n-val 500 --n-test 1000

# 3B: LAET f x loss, DARTH interval x loss
$PY -I exp03_run.py --phase 3b

$PY -I exp03_figures.py
$PY -I report_tables.py
```

Outputs (under `exp03_asymmetric_loss/`):

| File | Content |
|---|---|
| `data/metrics_3A.csv` | all 3A configurations, prediction and search metrics |
| `data/per_query_3A.csv` | per-query recall / stop NDC / oracle |
| `data/prediction_traces_3A.csv` | LAET native predictions vs true C1 |
| `data/offline_predictions_3A_darth.csv` | all DARTH models at the same checkpoints |
| `data/quality_cost_curves_3A.csv` | multiplier / stop-threshold sweeps |
| `data/metrics_3B.csv`, `per_query_3B.csv` | timing ablation |
| `configs/selection.json` | validation-MAE selection of the 3B quantile |
| `models/` | all trained LightGBM models |
| `figures/` | 5 required figures |
| `reports/report.md`, `findings.md` | full report and hypothesis verdicts |

All model selection (quantile tau for 3B) uses validation data only; the test
set is touched once, after the models are fixed.
