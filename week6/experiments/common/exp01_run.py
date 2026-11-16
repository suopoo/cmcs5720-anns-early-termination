"""Experiment 1 - Feature Transfer Ablation Study (GIST1M).

Runs the six controlled groups L0/L1/L2 and D0/D1/D2 at the unified
checkpoint f=1260 NDC, k=10, target recall 0.90, plus the original-method
reference for DARTH.  All searches are replayed on the shared traces.
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core import paths
from anns_exp import features as F
from anns_exp import gbdt, simulate as S, stats
from anns_exp.exp_common import (load_trace, grid_query_index, summarize,
                                 error_stats, write_csv, save_json, log,
                                 safe_div)

EXP = paths.exp_dir("exp01")
DATA = os.path.join(EXP, "data")
MODELS = os.path.join(EXP, "models")
ANALYSIS = os.path.join(EXP, "analysis")
FIG = os.path.join(EXP, "figures")
REPORTS = os.path.join(EXP, "reports")
for d in (DATA, MODELS, ANALYSIS, FIG, REPORTS, os.path.join(EXP, "configs"),
          os.path.join(EXP, "logs")):
    os.makedirs(d, exist_ok=True)

K, H, S_SUB = 10, 4, 200
F_MAIN = 1260
RT = 0.90


def subset_trace(tr, n):
    if n is None or n >= len(tr["ptr"]) - 1:
        return tr
    out = {}
    out["ptr"] = tr["ptr"][:n + 1].copy()
    m = out["ptr"][-1]
    out["step_ndc"] = tr["step_ndc"][:m]
    out["step_recall"] = tr["step_recall"][:m]
    out["c1"] = tr["c1"][:n]
    out["reason"] = tr["reason"][:n]
    for key in list(tr.keys()):
        if key.startswith("tail_f"):
            out[key] = tr[key][:n]
        if key.startswith("cost_rt"):
            out[key] = tr[key][:n]
        if key in ("suffix", "hom_feat"):
            out[key] = tr[key][:n]
        if key == "hom_ndc":
            out[key] = tr[key][:n]
    if "obs_ptr" in tr:
        out["obs_ptr"] = tr["obs_ptr"][:n + 1]
        mo = out["obs_ptr"][-1]
        for key in ("obs_ndc", "obs_recall", "obs_feat"):
            out[key] = tr[key][:mo]
    if "tr_ptr" in tr:
        out["tr_ptr"] = tr["tr_ptr"][:n + 1]
        mt = out["tr_ptr"][-1]
        for key in ("tr_ndc", "tr_darth", "tr_recall", "tr_suffix"):
            out[key] = tr[key][:mt]
    return out


def laet_matrix(tr, q, group):
    tail = tr[f"tail_f{F_MAIN}"]
    suf = tr["suffix"] if group != "L0" else None
    return F.assemble(group, q, tail=tail, suffix=suf, k=K, h=H)


def grid_subsample(tr, max_per_query=20):
    """Evenly spaced grid rows per query, for training only.

    Identical row selection is used for D0/D1/D2, so the controlled
    comparison is unchanged; the evaluation still uses every checkpoint.
    """
    ptr = tr["tr_ptr"]
    qi = grid_query_index(ptr)
    cnt = np.diff(ptr)[qi]
    pos = np.arange(len(qi)) - ptr[qi]
    stride = np.maximum(1, np.ceil(cnt / float(max_per_query))).astype(int)
    return (pos % stride) == 0


def darth_grid_matrix(tr, q, group):
    qi = grid_query_index(tr["tr_ptr"])
    if group == "D0":
        return tr["tr_darth"]
    return F.assemble(group, q[qi], suffix=tr["tr_suffix"], k=K, h=H, rt=RT)


def metrics_row(name, group, method, res, oracle, c1=None, pred_ndc=None,
                rt=RT):
    rec = np.asarray(res["recall"], float)
    ndc = np.asarray(res["stop_ndc"], float)
    hit = rec >= rt - 1e-9
    row = {"method": method, "group": group, "name": name,
           "n": int(len(rec)), "mean_recall": float(np.nanmean(rec)),
           "median_recall": float(np.nanmedian(rec)),
           "hit_rate": float(np.nanmean(hit)),
           "avg_ndc": float(np.nanmean(ndc)),
           "median_ndc": float(np.nanmedian(ndc)),
           "p95_ndc": float(np.nanpercentile(ndc, 95)),
           "p99_ndc": float(np.nanpercentile(ndc, 99))}
    row.update(summarize(rec, "recall_"))
    row.update(summarize(ndc, "ndc_"))
    orc = np.asarray(oracle, float)
    extra = np.maximum(0.0, ndc - orc)
    extra_ok = hit & np.isfinite(orc)
    row["extra_ndc_mean_all"] = float(np.nanmean(np.where(hit, extra, np.nan)))
    row["extra_ndc_mean_hit"] = float(np.nanmean(extra[extra_ok])) if extra_ok.any() else float("nan")
    row["relative_extra_ndc"] = float(np.nanmean(
        np.where(extra_ok, extra / np.maximum(orc, 1.0), np.nan)))
    row["failure_rate"] = float(1 - np.nanmean(hit))
    if pred_ndc is not None and c1 is not None:
        es = error_stats(pred_ndc, c1, "native_")
        es["native_mae"] = es.get("native_abs_mean", float("nan"))
        row.update(es)
    return row


def evaluate_laet_group(tr_val, qval, tr_test, qtest, model, group,
                        pred_max):
    Xv = laet_matrix(tr_val, qval, group)
    Xt = laet_matrix(tr_test, qtest, group)
    pv = gbdt.predict(model, Xv)
    mult, score, mults, _ = S.calibrate_multiplier(
        pv, tr_val, F_MAIN, pred_max, target=RT)
    res = S.laet_from_trace(tr_test, Xt, model, F_MAIN, mult, pred_max)
    oracle = tr_test["cost_rt0.9"]
    p_ndc = np.power(2.0, np.maximum(res["pred"], 0.0))
    row = metrics_row(f"{group}", group, "LAET", res, oracle,
                      c1=tr_test["c1"], pred_ndc=p_ndc)
    row.update({"multiplier": mult, "f": F_MAIN, "pred_max": pred_max,
                "val_mean_recall": float(score[np.argmin(np.abs(mults - mult))]),
                "truncated": int(res["truncated"].sum())})
    return res, row, mult


def evaluate_darth_group(tr_val, qval, tr_test, qtest, model, group):
    Xv = darth_grid_matrix(tr_val, qval, group)
    Xt = darth_grid_matrix(tr_test, qtest, group)
    rv = S.darth_fixed_from_grid(tr_val, Xv, model, RT, min_ndc=F_MAIN)
    rt_ = S.darth_fixed_from_grid(tr_test, Xt, model, RT, min_ndc=F_MAIN)
    oracle = tr_test["cost_rt0.9"]
    row = metrics_row(f"{group}", group, "DARTH", rt_, oracle)
    row.update({"val_mean_recall": float(np.nanmean(rv["recall"])),
                "val_hit_rate": float(np.nanmean(rv["recall"] >= RT - 1e-9)),
                "calls_mean": float(np.mean(rt_["calls"]))})
    return rt_, row


def darth_per_checkpoint(tr, q, group, model, split="test"):
    qi = grid_query_index(tr["tr_ptr"])
    X = darth_grid_matrix(tr, q, group)
    pred = gbdt.predict(model, X)
    ndc = tr["tr_ndc"]
    actual = tr["tr_recall"]
    rows = []
    for j in range(len(pred)):
        rows.append({
            "split": split, "query_id": int(qi[j]), "group": group,
            "checkpoint_id": int(j), "ndc_at_checkpoint": int(ndc[j]),
            "predicted_recall": float(pred[j]),
            "actual_recall": float(actual[j]),
            "recall_prediction_error": float(pred[j] - actual[j]),
            "target_recall": RT,
            "false_positive_stop": bool(pred[j] >= RT and actual[j] < RT),
            "false_negative_continue": bool(pred[j] < RT and actual[j] >= RT),
        })
    return rows


def per_query_laet(tr, group, res, c1, oracle, method="LAET"):
    n = len(res["recall"])
    rows = []
    for i in range(n):
        hit = bool(res["recall"][i] >= RT - 1e-9)
        orc = oracle[i]
        extra = max(0.0, res["stop_ndc"][i] - orc) if (hit and np.isfinite(orc)) else np.nan
        rows.append({
            "dataset": "GIST1M", "split": "test", "query_id": i,
            "method": method, "group": group, "k": K, "target_recall": RT,
            "oracle_ndc": float(orc), "oracle_attainable": bool(np.isfinite(orc)),
            "predicted_cost_raw": float(np.power(2.0, max(res["pred"][i], 0))),
            "effective_budget": float(res["budget"][i]),
            "actual_recall": float(res["recall"][i]),
            "target_hit": hit, "stop_ndc": float(res["stop_ndc"][i]),
            "stop_reason": "prediction" if np.isfinite(res["stop_ndc"][i]) else "natural",
            "native_prediction_error": float(np.power(2.0, max(res["pred"][i], 0)) - c1[i]) if np.isfinite(c1[i]) else np.nan,
            "effective_budget_error": float(res["budget"][i] - orc) if np.isfinite(orc) else np.nan,
            "extra_ndc": extra,
            "relative_extra_ndc": float(extra / max(orc, 1.0)) if np.isfinite(extra) and np.isfinite(orc) else np.nan,
            "truncated": bool(res["truncated"][i]),
        })
    return rows


def per_query_darth(tr, group, res, oracle):
    n = len(res["recall"])
    rows = []
    for i in range(n):
        hit = bool(res["recall"][i] >= RT - 1e-9)
        orc = oracle[i]
        extra = max(0.0, res["stop_ndc"][i] - orc) if (hit and np.isfinite(orc)) else np.nan
        rows.append({
            "dataset": "GIST1M", "split": "test", "query_id": i,
            "method": "DARTH", "group": group, "k": K, "target_recall": RT,
            "oracle_ndc": float(orc), "oracle_attainable": bool(np.isfinite(orc)),
            "predicted_recall_at_stop": float(res["stop_pred"][i]),
            "actual_recall": float(res["recall"][i]),
            "target_hit": hit, "stop_ndc": float(res["stop_ndc"][i]),
            "stop_reason": "prediction" if np.isfinite(res["stop_pred"][i]) else "natural",
            "native_prediction_error": float(res["stop_pred"][i] - res["recall"][i]) if np.isfinite(res["stop_pred"][i]) else np.nan,
            "extra_ndc": extra,
            "relative_extra_ndc": float(extra / max(orc, 1.0)) if np.isfinite(extra) and np.isfinite(orc) else np.nan,
            "n_predictions": int(res["calls"][i]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args()

    log("load traces")
    ltr_tr = subset_trace(load_trace("gist_k10_laet_train" + a.tag), a.n_train)
    ltr_va = subset_trace(load_trace("gist_k10_laet_val" + a.tag), a.n_val)
    ltr_te = subset_trace(load_trace("gist_k10_laet_test" + a.tag), a.n_test)
    dtr_tr = subset_trace(load_trace("gist_k10_darth_train" + a.tag), a.n_train)
    dtr_va = subset_trace(load_trace("gist_k10_darth_val" + a.tag), a.n_val)
    dtr_te = subset_trace(load_trace("gist_k10_darth_test" + a.tag), a.n_test)

    from anns_exp.trace import load_split
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]

    # ---------------- LAET groups ----------------
    rows = []
    pred_rows = []
    perq = []
    train_rows = []
    laet_res = {}
    for group in ("L0", "L1", "L2"):
        log(f"train {group}")
        Xtr = laet_matrix(ltr_tr, qtr, group)
        y = np.log2(np.maximum(ltr_tr["c1"], 1))
        ok = np.isfinite(y) & np.isfinite(Xtr).all(axis=1)
        log(f"  usable train rows {ok.sum()}/{len(ok)} dim={Xtr.shape[1]}")
        model = gbdt.train_lgb(Xtr[ok], y[ok], objective="regression",
                               seed=a.seed, n_estimators=100,
                               learning_rate=0.2, num_leaves=31,
                               min_child_samples=20)
        train_rows.append({"method": "LAET", "group": group,
                           "features": int(Xtr.shape[1]),
                           "usable_rows": int(ok.sum()),
                           "label": "log2(C1_NDC)", "loss": "L2",
                           "n_estimators": 100, "learning_rate": 0.2,
                           "num_leaves": 31, "seed": a.seed})
        joblib.dump(model, os.path.join(MODELS, f"laet_{group}.joblib"))
        imp = gbdt.gain_importance(model)
        write_csv([{"group": group, "rank": r + 1, "feature_index": None,
                    "feature": nm, "gain": g}
                   for r, (nm, g) in enumerate(imp[:40])],
                  os.path.join(ANALYSIS, f"importance_laet_{group}.csv"))
        pred_max = float(np.power(2.0, y[ok].max()))
        res, row, mult = evaluate_laet_group(ltr_va, qva, ltr_te, qte,
                                             model, group, pred_max)
        res["name"] = group
        laet_res[group] = res
        rows.append(row)
        oracle = ltr_te["cost_rt0.9"]
        perq += per_query_laet(ltr_te, group, res, ltr_te["c1"], oracle)
        pred_rows += [{
            "split": "test", "query_id": i, "group": group,
            "pred_log2_cost": float(res["pred"][i]),
            "pred_ndc": float(np.power(2.0, max(res["pred"][i], 0))),
            "true_c1_ndc": float(ltr_te["c1"][i]),
            "oracle_ndc_rt": float(oracle[i]),
            "budget": float(res["budget"][i]),
            "multiplier": mult,
        } for i in range(len(res["pred"]))]
        log(f"  {row['name']}: hit={row['hit_rate']:.3f} "
            f"ndc={row['avg_ndc']:.0f} mult={mult:.2f}")

    # ---------------- DARTH groups ----------------
    d_bar = float(np.nanmean(dtr_tr["cost_rt0.9"]))
    ipi, mpi = max(d_bar / 2.0, 32.0), max(d_bar / 10.0, 8.0)
    darth_rows, darth_perq, ckpt_rows = [], [], []
    darth_res = {}
    keep_tr = (grid_subsample(dtr_tr, max_per_query=20)
               & (dtr_tr["tr_ndc"] >= F_MAIN))
    for group in ("D0", "D1", "D2"):
        log(f"train {group}")
        Xtr = darth_grid_matrix(dtr_tr, qtr, group)[keep_tr]
        y = dtr_tr["tr_recall"][keep_tr]
        ok = np.isfinite(y) & np.isfinite(Xtr).all(axis=1)
        log(f"  usable train rows {ok.sum()}/{len(ok)} dim={Xtr.shape[1]}")
        model = gbdt.train_lgb(Xtr[ok], y[ok], objective="regression",
                               seed=a.seed, n_estimators=100,
                               learning_rate=0.1, num_leaves=31,
                               min_child_samples=min(1000, max(20, ok.sum() // 100)))
        train_rows.append({"method": "DARTH", "group": group,
                           "features": int(Xtr.shape[1]),
                           "usable_rows": int(ok.sum()),
                           "label": "current_recall", "loss": "L2",
                           "n_estimators": 100, "learning_rate": 0.1,
                           "num_leaves": 31, "seed": a.seed})
        joblib.dump(model, os.path.join(MODELS, f"darth_{group}.joblib"))
        imp = gbdt.gain_importance(model)
        write_csv([{"group": group, "rank": r + 1, "feature_index": None,
                    "feature": nm, "gain": g}
                   for r, (nm, g) in enumerate(imp[:40])],
                  os.path.join(ANALYSIS, f"importance_darth_{group}.csv"))
        res, row = evaluate_darth_group(dtr_va, qva, dtr_te, qte, model, group)
        darth_res[group] = res
        darth_rows.append(row)
        darth_perq += per_query_darth(dtr_te, group, res, dtr_te["cost_rt0.9"])
        ckpt_rows += darth_per_checkpoint(dtr_te, qte, group, model)
        log(f"  {row['name']}: hit={row['hit_rate']:.3f} ndc={row['avg_ndc']:.0f} "
            f"calls={row['calls_mean']:.1f}")

    # DARTH original reference: trained on observations from NDC 0, adaptive ipi
    log("train D0-original (observations from 0, adaptive interval)")
    Xo = dtr_tr["obs_feat"]; yo = dtr_tr["obs_recall"]
    ok = np.isfinite(yo) & np.isfinite(Xo).all(axis=1)
    model_o = gbdt.train_lgb(Xo[ok], yo[ok], objective="regression",
                             seed=a.seed, n_estimators=100,
                             learning_rate=0.1, num_leaves=31,
                             min_child_samples=min(1000, max(20, ok.sum() // 100)))
    joblib.dump(model_o, os.path.join(MODELS, "darth_D0_original.joblib"))
    res_o = S.darth_adaptive_from_steps(dtr_te, model_o, RT, ipi, mpi)
    row_o = metrics_row("D0-original", "D0-original", "DARTH", res_o,
                        dtr_te["cost_rt0.9"])
    row_o.update({"ipi": ipi, "mpi": mpi, "calls_mean": float(np.mean(res_o["calls"]))})
    darth_rows.append(row_o)
    darth_perq += per_query_darth(dtr_te, "D0-original", res_o,
                                  dtr_te["cost_rt0.9"])
    log(f"  D0-original: hit={row_o['hit_rate']:.3f} ndc={row_o['avg_ndc']:.0f} "
        f"calls={row_o['calls_mean']:.1f}")

    # ---------------- save raw results ----------------
    train_rows.append({"method": "DARTH", "group": "D0-original",
                       "features": 11, "usable_rows": int(ok.sum()),
                       "label": "current_recall", "loss": "L2",
                       "n_estimators": 100, "learning_rate": 0.1,
                       "num_leaves": 31, "seed": a.seed})
    write_csv(train_rows, os.path.join(DATA, "training_summary.csv"))
    write_csv(rows + darth_rows, os.path.join(DATA, "summary_metrics.csv"))
    write_csv(perq + darth_perq, os.path.join(DATA, "per_query_results.csv"))
    write_csv(pred_rows, os.path.join(DATA, "prediction_results.csv"))
    write_csv(ckpt_rows, os.path.join(DATA, "darth_checkpoint_predictions.csv"))

    # ---------------- quality-cost curves ----------------
    curve_rows = []
    mults = np.geomspace(0.1, 40.0, 40)
    for group in ("L0", "L1", "L2"):
        model = joblib.load(os.path.join(MODELS, f"laet_{group}.joblib"))
        Xt = laet_matrix(ltr_te, qte, group)
        p = gbdt.predict(model, Xt)
        pm = 2.0 ** np.maximum(p, 0.0)
        for m in mults:
            B = np.maximum(F_MAIN, np.minimum(pm * m, 40000.0))
            rr = S.laet_with_budget(ltr_te, B)
            curve_rows.append({"method": "LAET", "group": group,
                               "knob": "multiplier", "value": float(m),
                               "mean_recall": float(np.nanmean(rr["recall"])),
                               "hit_rate": float(np.nanmean(rr["recall"] >= RT - 1e-9)),
                               "mean_ndc": float(np.nanmean(rr["stop_ndc"]))})
    for group in ("D0", "D1", "D2"):
        model = joblib.load(os.path.join(MODELS, f"darth_{group}.joblib"))
        Xt = darth_grid_matrix(dtr_te, qte, group)
        for t in np.linspace(0.50, 0.995, 30):
            rr = S.darth_fixed_from_grid(dtr_te, Xt, model, float(t),
                                         min_ndc=F_MAIN)
            curve_rows.append({"method": "DARTH", "group": group,
                               "knob": "stop_threshold", "value": float(t),
                               "mean_recall": float(np.nanmean(rr["recall"])),
                               "hit_rate": float(np.nanmean(rr["recall"] >= RT - 1e-9)),
                               "mean_ndc": float(np.nanmean(rr["stop_ndc"]))})
    write_csv(curve_rows, os.path.join(DATA, "quality_cost_curves.csv"))

    # ---------------- paired statistics ----------------
    stat_rows = []
    # per-query absolute prediction error for the accuracy hypotheses
    assert_err = {}
    for g in ("L0", "L1", "L2"):
        p = np.power(2.0, np.maximum(laet_res[g]["pred"], 0.0))
        c1 = ltr_te["c1"]
        assert_err[g] = np.abs(np.log2(np.maximum(p, 1)) -
                               np.log2(np.maximum(c1, 1)))
    test_pairs = [("L1", "L0"), ("L2", "L0"), ("L1", "L2")]
    for x, y in test_pairs:
        for metric, arr in (("recall", "recall"), ("ndc", "stop_ndc")):
            d = stats.paired_diff(laet_res[x][arr], laet_res[y][arr])
            stat_rows.append({"method": "LAET", "comparison": f"{x} vs {y}",
                              "metric": metric, **d,
                              "cliffs_delta": stats.cliffs_delta(
                                  laet_res[x][arr], laet_res[y][arr])})
        d = stats.paired_diff(assert_err[x], assert_err[y])
        stat_rows.append({"method": "LAET", "comparison": f"{x} vs {y}",
                          "metric": "abs_log2_cost_error", **d,
                          "cliffs_delta": stats.cliffs_delta(
                              assert_err[x], assert_err[y])})
    ck = pd.DataFrame(ckpt_rows)
    ck["abs_err"] = ck["recall_prediction_error"].abs()
    qerr = ck.groupby(["query_id", "group"])["abs_err"].mean().unstack()
    for x, y in (("D1", "D0"), ("D2", "D0"), ("D1", "D2")):
        for metric, arr in (("recall", "recall"), ("ndc", "stop_ndc")):
            d = stats.paired_diff(darth_res[x][arr], darth_res[y][arr])
            stat_rows.append({"method": "DARTH", "comparison": f"{x} vs {y}",
                              "metric": metric, **d,
                              "cliffs_delta": stats.cliffs_delta(
                                  darth_res[x][arr], darth_res[y][arr])})
        d = stats.paired_diff(qerr[x].values, qerr[y].values)
        stat_rows.append({"method": "DARTH", "comparison": f"{x} vs {y}",
                          "metric": "query_mean_abs_recall_error", **d,
                          "cliffs_delta": stats.cliffs_delta(
                              qerr[x].values, qerr[y].values)})
    write_csv(stat_rows, os.path.join(ANALYSIS, "statistical_tests.csv"))

    # ---------------- config ----------------
    save_json({
        "experiment": "exp01_feature_transfer", "dataset": "GIST1M",
        "index": {"n": 992000, "M": 16, "efConstruction": 200,
                  "cache": "gist_idx992000_n992000_M16_efc200_graph.npz"},
        "k": K, "target_recall": RT, "checkpoint_ndc": F_MAIN,
        "train_queries": a.n_train, "val_queries": a.n_val,
        "test_queries": a.n_test, "seed": a.seed,
        "laet": {"loss": "L2 on log2(C1)", "n_estimators": 100,
                 "learning_rate": 0.2, "num_leaves": 31},
        "darth": {"loss": "L2 on current recall", "n_estimators": 100,
                  "learning_rate": 0.1, "num_leaves": 31,
                  "mode_A_delta": 200, "ipi": ipi, "mpi": mpi},
        "annie_graph": {"h": H, "s": S_SUB},
        "note": "D1 == D2 for DARTH because DARTH's 11 features are a subset "
                "of ANNiE's basic block after de-duplication.",
    }, os.path.join(EXP, "configs", "experiment_config.json"))

    if a.sensitivity:
        phase_sensitivity(a)
    log("done; summary saved")




def phase_sensitivity(a):
    """Preliminary-search-depth sensitivity for the original-feature models.

    L0 and D0 are retrained at each f; L1/L2/D1/D2 at f!=1260 would need
    ANNiE graph features at those states, which were not cached (documented
    limitation in the report).
    """
    ltr_tr = subset_trace(load_trace("gist_k10_laet_train" + a.tag), a.n_train)
    ltr_va = subset_trace(load_trace("gist_k10_laet_val" + a.tag), a.n_val)
    ltr_te = subset_trace(load_trace("gist_k10_laet_test" + a.tag), a.n_test)
    dtr_tr = subset_trace(load_trace("gist_k10_darth_train" + a.tag), a.n_train)
    dtr_te = subset_trace(load_trace("gist_k10_darth_test" + a.tag), a.n_test)
    from anns_exp.trace import load_split
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]
    rows = []
    train_rows = []
    for f in (256, 512, 1260, 2048):
        log(f"sensitivity LAET L0 f={f}")
        Xtr = np.concatenate([qtr, ltr_tr[f"tail_f{f}"]], axis=1)
        y = np.log2(np.maximum(ltr_tr["c1"], 1))
        ok = np.isfinite(y) & np.isfinite(Xtr).all(axis=1)
        model = gbdt.train_lgb(Xtr[ok], y[ok], objective="regression",
                               seed=a.seed, n_estimators=100,
                               learning_rate=0.2, num_leaves=31,
                               min_child_samples=20)
        pmax = float(np.power(2.0, y[ok].max()))
        Xv = np.concatenate([qva, ltr_va[f"tail_f{f}"]], axis=1)
        pv = gbdt.predict(model, Xv)
        mult, *_ = S.calibrate_multiplier(pv, ltr_va, f, pmax, target=RT)
        Xt = np.concatenate([qte, ltr_te[f"tail_f{f}"]], axis=1)
        res = S.laet_from_trace(ltr_te, Xt, model, f, mult, pmax)
        r = metrics_row(f"L0@f{f}", "L0", "LAET", res, ltr_te["cost_rt0.9"],
                        c1=ltr_te["c1"],
                        pred_ndc=np.power(2.0, np.maximum(res["pred"], 0)))
        r.update({"experiment": "checkpoint_sensitivity", "f": f,
                  "multiplier": mult})
        rows.append(r)
        log(f"  hit={r['hit_rate']:.3f} ndc={r['avg_ndc']:.0f}")
    # DARTH D0 at each f: train on grid states with ndc >= f
    for f in (256, 512, 1260, 2048):
        log(f"sensitivity DARTH D0 f={f}")
        grid = dtr_tr["tr_ndc"] >= f
        Xtr = dtr_tr["tr_darth"][grid]
        y = dtr_tr["tr_recall"][grid]
        ok = np.isfinite(y) & np.isfinite(Xtr).all(axis=1)
        model = gbdt.train_lgb(Xtr[ok], y[ok], objective="regression",
                               seed=a.seed, n_estimators=100,
                               learning_rate=0.1, num_leaves=31,
                               min_child_samples=min(1000, max(20, ok.sum() // 100)))
        Xt = dtr_te["tr_darth"][dtr_te["tr_ndc"] >= f]
        res = _darth_replay_subset(dtr_te, Xt, model, f)
        r = metrics_row(f"D0@f{f}", "D0", "DARTH", res, dtr_te["cost_rt0.9"])
        r.update({"experiment": "checkpoint_sensitivity", "f": f})
        rows.append(r)
        log(f"  hit={r['hit_rate']:.3f} ndc={r['avg_ndc']:.0f}")
    write_csv(rows, os.path.join(DATA, "checkpoint_sensitivity.csv"))


def _darth_replay_subset(tr, Xsub, model, f):
    """Fixed schedule using only grid rows with NDC >= f."""
    n = len(tr["ptr"]) - 1
    pred = gbdt.predict(model, Xsub)
    # map each prediction back to its query
    keep = np.flatnonzero(tr["tr_ndc"] >= f)
    qi = grid_query_index(tr["tr_ptr"])[keep]
    stop_ndc = np.full(n, np.nan); recall = np.full(n, np.nan)
    calls = np.zeros(n, dtype=np.int32); stop_pred = np.full(n, np.nan)
    for i in range(n):
        idx = np.flatnonzero(qi == i)
        if idx.size == 0:
            segn = tr["step_ndc"][tr["ptr"][i]:tr["ptr"][i+1]]
            segr = tr["step_recall"][tr["ptr"][i]:tr["ptr"][i+1]]
            if len(segn):
                stop_ndc[i] = segn[-1]; recall[i] = segr[-1]
            continue
        pp = pred[idx]
        hit = np.flatnonzero(pp >= RT - 1e-12)
        segn = tr["step_ndc"][tr["ptr"][i]:tr["ptr"][i+1]]
        segr = tr["step_recall"][tr["ptr"][i]:tr["ptr"][i+1]]
        if hit.size:
            j = int(hit[0])
            nd = tr["tr_ndc"][keep[idx[j]]]
            k = int(np.searchsorted(segn, nd, side="left"))
            k = min(k, len(segn) - 1)
            stop_ndc[i] = nd; recall[i] = segr[k]
            calls[i] = j + 1; stop_pred[i] = pp[j]
        elif len(segn):
            stop_ndc[i] = segn[-1]; recall[i] = segr[-1]
    return {"stop_ndc": stop_ndc, "recall": recall, "calls": calls,
            "stop_pred": stop_pred}


if __name__ == "__main__":
    main()
