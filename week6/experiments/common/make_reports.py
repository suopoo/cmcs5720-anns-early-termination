"""Generate the final markdown reports from the result CSVs.

The reports contain only numbers that exist in the CSV/JSON artefacts; the
hypothesis verdicts are derived from the paired-bootstrap CIs, not typed by
hand.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anns_exp.core import paths
from report_tables import md, exp01 as t01, exp02 as t02, exp03 as t03


def read(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def cfg(exp):
    p = os.path.join(paths.exp_dir(exp), "configs", "experiment_config.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def verdict(d, method, comparison, metric, lower_is_better=True):
    if d is None:
        return "inconclusive (no data)"
    r = d[(d.method == method) & (d.comparison == comparison) &
          (d.metric == metric)]
    if r.empty:
        return "inconclusive (not measured)"
    r = r.iloc[0]
    lo, hi, diff = r["lo"], r["hi"], r["diff"]
    if not np.isfinite(lo):
        return "inconclusive (no finite pairs)"
    if lower_is_better:
        if hi < 0:
            return f"supported (mean Δ={diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"
        if lo > 0:
            return f"contradicted (mean Δ={diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"
    else:
        if lo > 0:
            return f"supported (mean Δ={diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"
        if hi < 0:
            return f"contradicted (mean Δ={diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"
    return f"inconclusive (mean Δ={diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"


def write(exp, name, text):
    p = os.path.join(paths.exp_dir(exp), "reports", name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)
    print("wrote", p)


# --------------------------------------------------------------------------
def report_exp01():
    E = paths.exp_dir("exp01")
    s = read(os.path.join(E, "data", "summary_metrics.csv"))
    st = read(os.path.join(E, "analysis", "statistical_tests.csv"))
    if s is None:
        return
    c = cfg("exp01")
    laet = s[s.method == "LAET"].set_index("group")
    darth = s[s.method == "DARTH"].set_index("group")
    h1 = verdict(st, "LAET", "L1 vs L0", "abs_log2_cost_error")
    h1b = verdict(st, "LAET", "L2 vs L0", "abs_log2_cost_error")
    h1c = verdict(st, "LAET", "L1 vs L2", "abs_log2_cost_error")
    h1d = verdict(st, "DARTH", "D1 vs D0", "query_mean_abs_recall_error")
    h1e = verdict(st, "DARTH", "D1 vs D2", "query_mean_abs_recall_error")
    r_l1 = verdict(st, "LAET", "L1 vs L0", "recall", lower_is_better=False)
    n_l1 = verdict(st, "LAET", "L1 vs L0", "ndc")
    r_d1 = verdict(st, "DARTH", "D1 vs D0", "recall", lower_is_better=False)
    n_d1 = verdict(st, "DARTH", "D1 vs D0", "ndc")
    lat = read(os.path.join(E, "analysis", "latency_summary.csv"))
    lat_txt = ""
    if lat is not None and len(lat):
        lat_txt = ("Measured on 30 test queries (same process/hardware):\n\n"
                   + md(lat, ["method", "group", "mean_ms", "p50_ms", "p95_ms",
                              "feature_ms", "predict_ms", "calls", "mean_ndc"]))
    txt = f"""# Experiment 1 — Feature Transfer Ablation Study

## 1. Introduction

This experiment asks whether the input features used by ANNiE carry
information that LAET's and DARTH's original feature sets miss, while keeping
each method's prediction target, loss, checkpoint and stopping rule fixed.
Only the feature matrix changes across the six controlled groups.  The
dataset is GIST1M (992,000 index vectors, 960-d), k={c.get('k', 10)},
target recall {c.get('target_recall', 0.9)}, unified checkpoint f=1,260 NDC,
5,000 training / 500 validation / 1,000 official test queries (disjoint).

## 2. Research questions and hypotheses

RQ1: do ANNiE's features improve LAET/DARTH prediction accuracy?
RQ2: are the two feature sets complementary or substitutable?
RQ3: do prediction gains survive feature-extraction and inference cost?
RQ4: does the effect depend on preliminary-search depth?

The pre-registered hypotheses and decision rules are in
`research_questions.md`; they were fixed before the test run.

## 3. Methodology

* One hnswlib graph (M=16, efConstruction=200) parsed into the instrumented
  CSR search; NDC counts every base-layer squared-L2 evaluation.
* LAET: LightGBM L2 on `log2(C1)`, one prediction at NDC>=1,260, budget
  `max(f, min(2^pred * multiplier, pred_max))`; multiplier calibrated on the
  500 validation queries.
* DARTH: LightGBM L2 on current recall; Mode A uses the same 24-point fixed grid of
  checkpoints for D0/D1/D2 from NDC 1,260 to 12,000; see
  `experiment_config.json`.  The original adaptive-interval
  reference is reported separately as `D0-original`.
* ANNiE features: §4.2 basic (query vector, NDC, step, inserts, BSF distance
  statistics, recall target) and §4.5 graph (average/per-BSF clustering
  coefficient, 4-hop reachability, s=200 local subgraph).
* De-duplication makes **D1 == D2 for DARTH** by construction: DARTH's 11
  inputs are a strict subset of ANNiE's basic block.  This is a feature-set
  fact, not an experimental tie.
* Features are extracted only from the current search state; no future
  nodes, ground truth or final cost are used as inputs (see
  `source_audit.md`).

## 4. Results

### 4.1 Six-group summary

{md(s, ["method", "group", "n", "mean_recall", "hit_rate", "avg_ndc",
        "median_ndc", "p95_ndc", "failure_rate", "extra_ndc_mean_hit",
        "native_mae", "native_mape", "native_qerror", "multiplier",
        "calls_mean"])}

### 4.2 Paired query-level tests

{md(st)}

### 4.3 Latency

{lat_txt if lat_txt else 'Not measured in this run.'}

## 5. Per-query analysis

The paired tests use the same 1,000 Query IDs for every group.  The
`per_query_results.csv` file contains each query's stop NDC, actual recall,
oracle cost, extra NDC and prediction error, so every aggregate above can be
traced to individual queries.

## 6. Hypothesis evaluation

| Hypothesis | Evidence (paired 95% CI) | Verdict |
|---|---|---|
| H1 (LAET, L1 adds information) | {h1} | see evidence |
| H1b (LAET, L2 beats L0) | {h1b} | see evidence |
| H1c (LAET, originals add complementarity) | {h1c} | see evidence |
| H1 (DARTH, D1 adds information) | {h1d} | see evidence |
| H2 (DARTH, D2 replaces originals) | {h1e} (D1≡D2 structurally) | not testable as a tie |

Prediction accuracy does not automatically transfer to search quality:

* LAET L1 vs L0 on search recall: {r_l1}
* LAET L1 vs L0 on NDC: {n_l1}
* DARTH D1 vs D0 on search recall: {r_d1}
* DARTH D1 vs D0 on NDC: {n_d1}

## 7. Discussion

The direction of the evidence is read directly from the CIs.  A
non-significant difference is reported as inconclusive, not as proof that the
features are useless.  Two structural facts constrain the interpretation:

1. For DARTH, ANNiE's basic block contains every DARTH input, so D2 cannot be
   information-poorer than D0; only estimation variance can make it worse.
2. For LAET, the query vector (960 dims) dominates the input; the added
   distance statistics and graph features operate on a much smaller scale and
   LightGBM may under-use them at n=5,000.

## 8. Threats to validity

* Python search loop: NDC is portable, wall-clock is not comparable to the
  papers' C++.
* GIST1M at 992k (8k held-out queries), not the full 1M; the graph is built
  once with a fixed seed.
* The 1,260 checkpoint is LAET's published GIST value; for DARTH it is a
  controlled deviation from the adaptive first-call position.
* Graph features use the Def. 7 approximation; `s=200`, `h=4` follow the
  audited implementation.
* The training size is 5,000, below the papers' largest settings.

## 9. Conclusion

{hypothesis_conclusion(h1, h1b, h1c, h1d)}

"""
    write("exp01", "experiment_report.md", txt)
    zh = f"""# 实验一：特征迁移消融研究报告（中文）

## 实验设置

GIST1M（索引 992,000 条、960 维），k={c.get('k', 10)}，目标 Recall
{c.get('target_recall', 0.9)}，统一预测点 f=1,260 NDC；训练 5,000 /
验证 500 / 测试 1,000 条互不重叠的 Query。六个实验组只改变输入特征，
预测目标、损失函数、训练数据、预测时机和终止规则全部固定。DARTH 的
D1 与 D2 在去重后完全相同，因为 DARTH 的 11 个特征本来就是 ANNiE
basic 特征块的子集；这是结构事实，不是实验结论。

## 主要结果

{md(s, ["method", "group", "mean_recall", "hit_rate", "avg_ndc",
        "p95_ndc", "native_mae", "native_mape", "multiplier", "calls_mean"])}

## 配对统计与假设

| 假设 | 证据（配对 95% CI） |
|---|---|
| H1：L1 优于 L0 的预测误差 | {h1} |
| H1b：L2 优于 L0 | {h1b} |
| H1c：L1 明显优于 L2（原特征有互补信息） | {h1c} |
| H1：D1 优于 D0（DARTH） | {h1d} |
| 端到端：L1 vs L0 recall | {r_l1} |
| 端到端：L1 vs L0 NDC | {n_l1} |
| 端到端：D1 vs D0 recall | {r_d1} |
| 端到端：D1 vs D0 NDC | {n_d1} |

## 解释

预测精度的变化与端到端搜索收益必须分开判断。即使某个迁移组的平均
误差下降，若 NDC 或达标率没有同向变化，也不能声称端到端变好；反之，
图结构特征的提取开销也必须计入。所有结论均以 Query 级配对 Bootstrap
置信区间为准，不显著时写为“证据不足”。

## 局限

Python 实现、992k 索引规模、5,000 条训练 Query、ANNiE 局部子图的 Def.7
近似、以及只在 f=1,260 对 L1/L2、D1/D2 做完整比较，都会限制结论的外推。
"""
    write("exp01", "experiment_report_zh.md", zh)


def hypothesis_conclusion(h1, h1b, h1c, h1d):
    def label(x):
        return ("supported" if x.startswith("supported") else
                "contradicted" if x.startswith("contradicted") else
                "inconclusive")
    return (f"On GIST1M at the controlled checkpoint, H1 for LAET is "
            f"{label(h1)}, H1b is {label(h1b)}, H1c is {label(h1c)}, and H1 "
            f"for DARTH is {label(h1d)}.  These labels describe paired "
            f"query-level evidence, not a claim that one feature set is "
            f"universally better.")


# --------------------------------------------------------------------------
def report_exp02():
    E = paths.exp_dir("exp02")
    s = read(os.path.join(E, "analysis", "error_summary.csv"))
    ov = read(os.path.join(E, "analysis", "failure_overlap.csv"))
    size = read(os.path.join(E, "analysis", "training_size_summary.csv"))
    dist = read(os.path.join(E, "analysis", "distribution_summary.csv"))
    rt = read(os.path.join(E, "analysis", "recall_target_summary.csv"))
    if s is None:
        return
    c = cfg("exp02")
    ovq = ov[ov.failure_type == "quality_fail"] if ov is not None else None
    txt = f"""# Experiment 2 — Per-Query Error Analysis of Learned Early Termination

## 1. Introduction

Average NDC and average recall hide where the three methods fail.  This
experiment instruments every prediction on the same 1,000 official GIST1M
test queries (k=100, R_t=0.95, one graph) and asks four questions: what
errors each method makes, whether failures concentrate on the same queries,
how errors translate into stopping decisions, and whether training size /
distribution / target recall change the picture.

## 2. Research questions and hypotheses

RQ1/H1: the methods have different error distributions and tail risk.
RQ2/H2: severe failures may overlap across methods beyond query difficulty.
RQ3/H3: under- and over-estimation have asymmetric consequences.
RQ4/H4: training size, distribution and target recall affect the errors.

The null hypotheses and the stratified permutation design are in
`audit/method_audit.md` and the analysis code.

## 3. Methodology

* LAET: `log2(C1)` cost predictor, one call at f=1,260, multiplier
  calibrated on 500 validation queries; effective budget error is measured
  against the target-recall oracle.
* DARTH: current-recall predictor, adaptive interval
  `pi=mpi+(ipi-mpi)(R_t-pred)`; every checkpoint is logged.
* ANNiE-S: quantile (p=0.95) cost predictor at the homing transition;
  effective budget is `max(Chat, homing NDC)`.
* Oracles are computed on each method's own complete trajectory; `unknown`
  is kept distinct from `unreachable`.
* Training: 3,000 queries for the main fit; 500/1,000/2,000/3,000 nested
  subsets x 3 seeds for the size experiment.

## 4. Individual error analysis

{md(s)}

### Paired method comparison (same queries)

{md(read(os.path.join(E, "analysis", "method_comparison.csv"))) if os.path.exists(os.path.join(E, "analysis", "method_comparison.csv")) else 'No paired comparison table.'}

## 5. Cross-method query analysis

{md(ovq) if ovq is not None else 'No overlap table.'}

## 6. Training data analysis

### 6.1 Training-set size (mean over 3 seeds)

{md(size) if size is not None else 'Not run.'}

### 6.2 Hard-query-enriched training (D2)

{md(dist) if dist is not None else 'Not run.'}

## 7. Recall-target sensitivity

{md(rt) if rt is not None else 'Not run.'}

## 8. Hypothesis evaluation

* H1: compare the error distributions and tail metrics in §4.  Cost errors
  (LAET/ANNiE) and recall errors (DARTH) are reported on separate axes.
* H2: read the Jaccard, conditional failure probability and stratified
  permutation p-values in §5.  If the overlap disappears after conditioning
  on oracle-cost strata, the shared failure is query hardness, not a shared
  model defect.
* H3: compare false-positive stops / false-negative continues for DARTH and
  budget under-estimation for LAET/ANNiE with the target failure rates.  A
  positive signed error has opposite practical meaning for a cost predictor
  and a recall predictor.
* H4: compare the size, distribution and target-recall tables.  Three seeds
  are reported; a non-monotone or flat curve is evidence of no size effect,
  not proof of one.

Conclusions are limited to what the CIs and permutation tests support; a
non-significant overlap is reported as “no evidence of extra association”,
not as independence.

## 9. Threats to validity

Python implementation; 992k index; 3,000 training queries; one dataset;
DARTH observations sampled every 25 NDC; ANNiE graph features use the
Def. 7 approximation; the efficiency-failure threshold is the method's own
worst decile, so cross-method efficiency sets are relative, not absolute.

"""
    write("exp02", "experiment_report.md", txt)
    bullets = []
    for r in s.itertuples():
        bullets.append(
            f"- **{r.method}**: mean recall {r.mean_recall:.4f}, "
            f"target hit rate {r.hit_rate:.4f}, mean NDC {r.avg_ndc:.0f}, "
            f"P95 NDC {r.p95_ndc:.0f}; oracle-attainable "
            f"{r.oracle_attainable_rate:.4f}.")
    if ov is not None:
        q = ov[(ov.failure_type == "quality_fail") &
               ov.A.isin(["LAET", "DARTH", "ANNiE-S"]) &
               ov.B.isin(["LAET", "DARTH", "ANNiE-S"])]
        for r in q.itertuples():
            bullets.append(
                f"- Quality-failure overlap {r.A}∩{r.B}: "
                f"{int(r.intersection)} queries, Jaccard {r.jaccard:.3f}, "
                f"P({r.B}|{r.A})={r.P_B_given_A:.3f}, stratified "
                f"permutation p={r.p_value:.4f}.")
    if size is not None:
        g = size.groupby(["method", "train_size"]).hit_rate.mean().reset_index()
        for m in g.method.unique():
            d = g[g.method == m]
            bullets.append(
                f"- {m} target hit rate from n={int(d.train_size.min())} to "
                f"n={int(d.train_size.max())}: {d.hit_rate.iloc[0]:.4f} → "
                f"{d.hit_rate.iloc[-1]:.4f} (3-seed mean).")
    if rt is not None:
        for m in rt.method.unique():
            d = rt[rt.method == m]
            bullets.append(
                f"- {m} R_t sensitivity (0.90/0.95/0.99): hit rate "
                + " / ".join(f"{v:.3f}" for v in d.hit_rate) + ".")
    write("exp02", "findings_summary.md",
          "# Experiment 2 — findings (generated from the result tables)\n\n"
          + "\n".join(bullets) +
          "\n\nThese are observations, not causal claims; see "
          "`experiment_report.md` for the limitations.\n")


# --------------------------------------------------------------------------
def report_exp03():
    E = paths.exp_dir("exp03")
    a = read(os.path.join(E, "data", "metrics_3A.csv"))
    b = read(os.path.join(E, "data", "metrics_3B.csv"))
    sel = cfg("exp03") if False else None
    sp = os.path.join(E, "configs", "selection.json")
    sel = json.load(open(sp)) if os.path.exists(sp) else {}
    if a is None:
        return
    txt = f"""# Experiment 3 — Asymmetric Loss and Prediction Timing

## 1. Introduction and motivation

LAET and DARTH both train with squared error, which penalises an
under-estimate and an over-estimate equally even though their consequences
are asymmetric: an under-budget LAET stop loses recall, while an
over-estimated DARTH recall triggers a false stop.  ANNiE's quantile loss
suggests an alternative.  This experiment tests whether changing only the
training loss improves the recall–efficiency trade-off, and whether any
benefit survives prediction-timing changes and conventional multiplier /
interval calibration.

## 2. Research questions

RQ1: can asymmetric loss improve the trade-off?  RQ2: does prediction
timing change the benefit?

## 3. Hypotheses

H1-LAET: penalising cost under-estimation reduces budget shortfalls and
raises recall, possibly at the price of more NDC.  H1-DARTH: penalising
recall over-estimation reduces false stops and raises the target hit rate,
possibly at the price of more calls/NDC.  H2-LAET/H2-DARTH: the benefit
depends on f / (ipi, mpi).  All are stated before the test run.

## 4. Experimental setup and controlled variables

GIST1M, k=10, R_t=0.90, 5,000/500/1,000 split.  Within each method the
index, queries, features, label, LightGBM structure, seed, f (3A) and
ipi/mpi (3A) are fixed; only the loss changes.  LAET uses
`log2(C1)` and tau=0.50/0.75/0.90/0.95 (larger tau penalises low
predictions); DARTH uses current recall and tau=0.50/0.25/0.10/0.05 (lower
tau penalises high predictions).  A1 applies the L0 multiplier to every
model; A2 recalibrates each model on validation.  The 3B quantile is chosen
by validation native MAE: {sel.get('selected_laet')} for LAET,
{sel.get('selected_darth')} for DARTH.

## 5. Experiment 3A results

{md(a)}

### Paired tests vs the original loss

{md(read(os.path.join(E, "analysis", "statistical_tests_3A.csv"))) if os.path.exists(os.path.join(E, "analysis", "statistical_tests_3A.csv")) else 'No paired table.'}

## 6. Experiment 3B results

{md(b) if b is not None else 'Not run.'}

## 7. Statistical analysis

Query-level paired bootstrap is used throughout; DARTH checkpoint errors are
first averaged per query.  A difference is called meaningful only when the
95% CI excludes zero; the effect size and direction are reported in the
tables.  A recall gain accompanied by a much larger NDC increase is not
called an overall improvement.

## 8. Findings and interpretation

* LAET: compare A1 (fixed multiplier) with A2 (recalibrated).  If A1 improves
  but A2 does not, the gain was a larger effective budget rather than a
  better cost estimate.  The multiplier is the key confounder.
* DARTH: compare the false-stop rate and target hit rate across tau.  A lower
  tau should reduce positive recall error; whether that becomes a higher hit
  rate at an acceptable NDC/call cost is a separate question.
* 3B: read the f x loss and interval x loss tables.  A benefit that shrinks
  as f grows indicates that a longer preliminary search substitutes for the
  asymmetric penalty.

## 9. Threats to validity

Python implementation, 992k index, 5,000 training queries, one dataset,
LightGBM's quantile implementation, and the fixed 1,260 checkpoint.  The
original DARTH objective is the library default L2, verified in the audit;
the paper's text alone would not justify calling it quantile.

## 10. Conclusion

The experiment separates three effects that are often conflated: the model's
prediction change, the stopping-decision change, and the end-to-end cost.
The tables above are the evidence for each; no claim beyond their CIs is
made.

"""
    write("exp03", "report.md", txt)
    st = read(os.path.join(E, "analysis", "statistical_tests_3A.csv"))
    bul = []
    if a is not None:
        for r in a.itertuples():
            bul.append(f"- {getattr(r, 'experiment', '')} "
                       f"{getattr(r, 'config', getattr(r, 'setting', ''))}: "
                       f"hit rate {r.hit_rate:.4f}, mean NDC {r.avg_ndc:.0f}.")
    if st is not None:
        for r in st.itertuples():
            bul.append(f"- {r.experiment} {r.comparison} ({r.metric}): "
                       f"Δ={r.diff:+.4f}, 95% CI [{r.lo:+.4f}, {r.hi:+.4f}].")
    write("exp03", "findings.md",
          "# Experiment 3 — findings\n\n" + "\n".join(bul) +
          "\n\nVerdicts follow the CI signs; a benefit that exists only "
          "with a larger effective budget is labelled as such.\n")


if __name__ == "__main__":
    from anns_exp.features import registry_rows
    import pandas as _pd
    _pd.DataFrame(registry_rows(960, 10, 4)).to_csv(
        os.path.join(paths.exp_dir("exp01"), "data", "feature_registry.csv"),
        index=False, encoding="utf-8-sig")
    report_exp01()
    report_exp02()
    report_exp03()
    # config YAML copies
    try:
        import yaml
        for exp in ("exp01", "exp02", "exp03"):
            p = os.path.join(paths.exp_dir(exp), "configs", "experiment_config.json")
            if os.path.exists(p):
                d = json.load(open(p))
                y = os.path.join(paths.exp_dir(exp), "configs", "experiment_config.yaml")
                with open(y, "w", encoding="utf-8") as f:
                    yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print("yaml copy skipped", e)
    import platform, numpy, pandas, lightgbm
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": numpy.__version__, "pandas": pandas.__version__,
           "lightgbm": lightgbm.__version__,
           "dataset": "GIST1M gist.hdf5", "index_vectors": 992000,
           "index": "hnswlib M=16 efConstruction=200",
           "device": "CPU only"}
    p = os.path.join(paths.exp_dir("exp02"), "reproducibility", "environment.json")
    with open(p, "w") as f:
        json.dump(env, f, indent=2)
    print("reports done")
