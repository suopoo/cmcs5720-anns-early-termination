"""Turn result CSVs into markdown tables (used by the final reports)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anns_exp.core import paths


def md(df, cols=None, floatfmt="{:.4f}"):
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    out = ["| " + " | ".join(df.columns) + " |",
           "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                vals.append("" if not np.isfinite(v) else floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                vals.append(str(v))
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def exp01():
    E = paths.exp_dir("exp01")
    parts = ["# Experiment 1 — tables"]
    s = pd.read_csv(os.path.join(E, "data", "summary_metrics.csv"))
    cols = ["method", "group", "n", "mean_recall", "hit_rate", "avg_ndc",
            "median_ndc", "p95_ndc", "failure_rate", "extra_ndc_mean_hit",
            "relative_extra_ndc", "native_mae", "native_mape",
            "native_qerror", "multiplier", "calls_mean"]
    parts += ["\n## Summary metrics\n", md(s, cols)]
    st = pd.read_csv(os.path.join(E, "analysis", "statistical_tests.csv"))
    parts += ["\n## Paired bootstrap comparisons (test queries)\n", md(st)]
    p = os.path.join(E, "data", "checkpoint_sensitivity.csv")
    if os.path.exists(p):
        parts += ["\n## Preliminary-search sensitivity\n", md(pd.read_csv(p))]
    p = os.path.join(E, "analysis", "latency_summary.csv")
    if os.path.exists(p):
        parts += ["\n## Measured latency (30 test queries)\n", md(pd.read_csv(p))]
    return "\n".join(parts)


def exp02():
    E = paths.exp_dir("exp02")
    parts = ["# Experiment 2 — tables"]
    p = os.path.join(E, "analysis", "error_summary.csv")
    if os.path.exists(p):
        parts += ["\n## Method summary\n", md(pd.read_csv(p))]
    p = os.path.join(E, "analysis", "failure_overlap.csv")
    if os.path.exists(p):
        parts += ["\n## Failure-set overlap\n", md(pd.read_csv(p))]
    p = os.path.join(E, "analysis", "training_size_summary.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        g = d.groupby(["method", "train_size"]).agg(
            hit_rate=("hit_rate", "mean"), hit_sd=("hit_rate", "std"),
            avg_ndc=("avg_ndc", "mean"), ndc_sd=("avg_ndc", "std"),
            mean_recall=("mean_recall", "mean")).reset_index()
        parts += ["\n## Training-set size (mean ± SD over 3 seeds)\n", md(g)]
    p = os.path.join(E, "analysis", "distribution_summary.csv")
    if os.path.exists(p):
        parts += ["\n## Training-distribution experiment\n", md(pd.read_csv(p))]
    p = os.path.join(E, "analysis", "recall_target_summary.csv")
    if os.path.exists(p):
        parts += ["\n## Recall-target sensitivity\n", md(pd.read_csv(p))]
    return "\n".join(parts)


def exp03():
    E = paths.exp_dir("exp03")
    parts = ["# Experiment 3 — tables"]
    p = os.path.join(E, "data", "metrics_3A.csv")
    if os.path.exists(p):
        parts += ["\n## 3A loss ablation\n", md(pd.read_csv(p))]
    p = os.path.join(E, "data", "metrics_3B.csv")
    if os.path.exists(p):
        parts += ["\n## 3B timing ablation\n", md(pd.read_csv(p))]
    p = os.path.join(E, "configs", "selection.json")
    if os.path.exists(p):
        parts += ["\n## Validation-based selection\n",
                  "```json\n" + open(p).read() + "\n```"]
    return "\n".join(parts)


if __name__ == "__main__":
    out = paths.EXP_ROOT
    open(os.path.join(paths.exp_dir("exp01"), "analysis", "report_tables.md"),
         "w", encoding="utf-8").write(exp01())
    open(os.path.join(paths.exp_dir("exp02"), "analysis", "report_tables.md"),
         "w", encoding="utf-8").write(exp02())
    open(os.path.join(paths.exp_dir("exp03"), "analysis", "report_tables.md"),
         "w", encoding="utf-8").write(exp03())
    print("tables written")
