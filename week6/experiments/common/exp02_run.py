"""Experiment 2 - Per-query prediction-error analysis (GIST1M, k=100).

Methods: LAET (cost of top-1), DARTH (current recall), ANNiE-S (cost to
reach recall@100 >= 0.95).  All three run on the same 1000 official test
queries and the same index.

Phases:
  main  : per-query + per-checkpoint instrumentation, failure overlap
  size  : training-set-size sweep (nested subsets, 3 seeds)
  dist  : hard-query-enriched training (D2)
  rt    : recall-target sensitivity 0.90 / 0.95 / 0.99
"""
from __future__ import annotations

import argparse
import json
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
                                 error_stats, write_csv, save_json, log)
from anns_exp.trace import load_split

EXP = paths.exp_dir("exp02")
DATA = os.path.join(EXP, "data")
ANA = os.path.join(EXP, "analysis")
MODELS = os.path.join(EXP, "models")
REP = os.path.join(EXP, "reports")
for d in (DATA, ANA, MODELS, REP, os.path.join(EXP, "configs"),
          os.path.join(EXP, "logs"), os.path.join(EXP, "figures"),
          os.path.join(EXP, "audit"), os.path.join(EXP, "reproducibility")):
    os.makedirs(d, exist_ok=True)

K, H, S_SUB, F_MAIN = 100, 4, 200, 1260
RT = 0.95
N_TRAIN = 2000


def win(tr, n):
    if n is None or n >= len(tr["ptr"]) - 1:
        return tr
    out = {"ptr": tr["ptr"][:n + 1].copy()}
    m = out["ptr"][-1]
    for key in ("step_ndc", "step_recall"):
        out[key] = tr[key][:m]
    for key in list(tr.keys()):
        if key.startswith("tail_f") or key.startswith("cost_rt") or key in (
                "c1", "hom_feat", "hom_ndc", "reason", "suffix"):
            out[key] = tr[key][:n]
    if "obs_ptr" in tr:
        out["obs_ptr"] = tr["obs_ptr"][:n + 1]
        for key in ("obs_ndc", "obs_recall", "obs_feat"):
            out[key] = tr[key][:out["obs_ptr"][-1]]
    if "tr_ptr" in tr:
        out["tr_ptr"] = tr["tr_ptr"][:n + 1]
        for key in ("tr_ndc", "tr_darth", "tr_recall", "tr_suffix"):
            out[key] = tr[key][:out["tr_ptr"][-1]]
    return out


# --------------------------------------------------------------------------
# training helpers
# --------------------------------------------------------------------------
def laet_X(tr, q, f=F_MAIN, idx=None):
    tail = tr[f"tail_f{f}"] if idx is None else tr[f"tail_f{f}"][idx]
    return np.concatenate([q, tail], axis=1)


def annie_X(tr, q, rt=RT, idx=None):
    # hom_feat stores only basic+graph (12 + 1 + k + h); the 960-d query
    # vector is stored once per query and re-attached here.
    hf = tr["hom_feat"] if idx is None else tr["hom_feat"][idx]
    X = np.concatenate([q, hf], axis=1)
    X[:, 960 + 11] = rt
    return X


def train_laet(ltr, qtr, idx, seed=42):
    c1 = ltr["c1"][idx]
    ok = np.isfinite(c1)
    X = np.concatenate([qtr[idx], ltr["tail_f1260"][idx]], axis=1)
    ok &= np.isfinite(X).all(axis=1)
    y = np.log2(np.maximum(c1[ok], 1))
    model = gbdt.train_lgb(X[ok], y, objective="regression", seed=seed,
                           n_estimators=100, learning_rate=0.2,
                           num_leaves=31, min_child_samples=20)
    return model, float(2.0 ** y.max()), int(ok.sum())


def train_darth(dtr, qtr, idx, seed=42):
    qi = grid_query_index(dtr["obs_ptr"])
    sel = np.isin(qi, idx)
    X = dtr["obs_feat"][sel]
    y = dtr["obs_recall"][sel]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    model = gbdt.train_lgb(X[ok], y, objective="regression", seed=seed,
                           n_estimators=100, learning_rate=0.1,
                           num_leaves=31,
                           min_child_samples=min(1000, max(20, ok.sum() // 100)))
    dbar = float(np.nanmean(dtr["cost_rt0.95"][idx]))
    return model, max(dbar / 2.0, 32.0), max(dbar / 10.0, 8.0), int(ok.sum())


def train_annie(utr, qtr, idx, rt=RT, seed=42, p=0.95):
    X = np.concatenate([qtr[idx], utr["hom_feat"][idx]], axis=1)
    X[:, 960 + 11] = rt
    y = utr[f"cost_rt{rt}"][idx]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    model = gbdt.train_lgb(X[ok], y[ok], objective="quantile", alpha=p, seed=seed,
                           n_estimators=400, learning_rate=0.05,
                           num_leaves=6,
                           min_child_samples=min(1000, max(20, ok.sum() // 50)))
    return model, int(ok.sum())


# --------------------------------------------------------------------------
# evaluation helpers (all replay on the shared test traces)
# --------------------------------------------------------------------------
def eval_laet(utr_va, qva, utr_te, qte, model, pred_max, rt=RT):
    Xv = laet_X(utr_va, qva)
    mult, *_ = S.calibrate_multiplier(gbdt.predict(model, Xv), utr_va,
                                      F_MAIN, pred_max, target=rt)
    Xt = laet_X(utr_te, qte)
    res = S.laet_from_trace(utr_te, Xt, model, F_MAIN, mult, pred_max)
    res["multiplier"] = mult
    return res


def eval_annie(utr_te, qte, model, rt=RT):
    Xt = annie_X(utr_te, qte, rt)
    return S.annie_from_trace(utr_te, Xt, model,
                              utr_te["hom_ndc"])


def eval_darth(dtr_te, model, ipi, mpi, rt=RT):
    return S.darth_adaptive_from_steps(dtr_te, model, rt, ipi, mpi)


def oracle_columns(tr, rt, method):
    cost = tr.get(f"cost_rt{rt}", np.full(len(tr["ptr"]) - 1, np.nan))
    reason = tr["reason"]
    status = np.where(np.isfinite(cost), "reached",
                      np.where(reason == "natural", "unreachable", "unknown"))
    return cost, status


def summary_row(method, res, oracle, rt=RT):
    rec = np.asarray(res["recall"], float)
    ndc = np.asarray(res["stop_ndc"], float)
    hit = rec >= rt - 1e-9
    extra = np.maximum(0.0, ndc - np.asarray(oracle, float))
    ok = hit & np.isfinite(oracle)
    return {
        "method": method, "n": int(len(rec)),
        "mean_recall": float(np.nanmean(rec)),
        "median_recall": float(np.nanmedian(rec)),
        "hit_rate": float(np.nanmean(hit)),
        "failure_rate": float(1 - np.nanmean(hit)),
        "avg_ndc": float(np.nanmean(ndc)),
        "median_ndc": float(np.nanmedian(ndc)),
        "p95_ndc": float(np.nanpercentile(ndc, 95)),
        "p99_ndc": float(np.nanpercentile(ndc, 99)),
        "extra_ndc_mean": float(np.nanmean(extra[ok])) if ok.any() else float("nan"),
        "relative_extra_ndc": float(np.nanmean(extra[ok] / np.maximum(
            np.asarray(oracle, float)[ok], 1.0))) if ok.any() else float("nan"),
        "oracle_attainable_rate": float(np.nanmean(np.isfinite(oracle))),
    }


# --------------------------------------------------------------------------
# main phase
# --------------------------------------------------------------------------
def phase_main(a):
    log("load k=100 traces")
    utr_tr = win(load_trace("gist_k100_unlim_train" + a.tag), a.n_train)
    utr_va = win(load_trace("gist_k100_unlim_val" + a.tag), a.n_val)
    utr_te = win(load_trace("gist_k100_unlim_test" + a.tag), a.n_test)
    dtr_tr = win(load_trace("gist_k100_darth_train" + a.tag), a.n_train)
    dtr_va = win(load_trace("gist_k100_darth_val" + a.tag), a.n_val)
    dtr_te = win(load_trace("gist_k100_darth_test" + a.tag), a.n_test)
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]

    idx = np.arange(a.n_train)
    log("train LAET")
    lm, lpm, ln = train_laet(utr_tr, qtr, idx, a.seed)
    log(f"  rows={ln}")
    lres = eval_laet(utr_va, qva, utr_te, qte, lm, lpm)
    log(f"  mult={lres['multiplier']:.3f} hit={np.nanmean(lres['recall']>=RT-1e-9):.3f}")

    log("train DARTH")
    dm, dipi, dmpi, dn = train_darth(dtr_tr, qtr, idx, a.seed)
    log(f"  rows={dn} ipi={dipi:.0f} mpi={dmpi:.0f}")
    dres = eval_darth(dtr_te, dm, dipi, dmpi)
    log(f"  hit={np.nanmean(dres['recall']>=RT-1e-9):.3f} calls={dres['calls'].mean():.1f}")

    log("train ANNiE-S")
    am, an = train_annie(utr_tr, qtr, idx, RT, a.seed)
    log(f"  rows={an}")
    ares = eval_annie(utr_te, qte, am)
    log(f"  hit={np.nanmean(ares['recall']>=RT-1e-9):.3f}")

    joblib.dump({"model": lm, "pred_max": lpm, "mult": lres["multiplier"]},
                os.path.join(MODELS, "laet_main.joblib"))
    joblib.dump({"model": dm, "ipi": dipi, "mpi": dmpi},
                os.path.join(MODELS, "darth_main.joblib"))
    joblib.dump({"model": am}, os.path.join(MODELS, "annie_main.joblib"))

    # ---- oracle + per-query rows ----
    lor, lstat = oracle_columns(utr_te, RT, "LAET")
    dor, dstat = oracle_columns(dtr_te, RT, "DARTH")
    aor, astat = oracle_columns(utr_te, RT, "ANNiE")
    n = a.n_test
    perq, orows = [], []
    for i in range(n):
        orows += [
            {"method": "LAET", "query_id": i, "oracle_ndc": float(lor[i]),
             "oracle_attainable": bool(np.isfinite(lor[i])),
             "oracle_status": lstat[i], "oracle_type": "unlimited_beam"},
            {"method": "DARTH", "query_id": i, "oracle_ndc": float(dor[i]),
             "oracle_attainable": bool(np.isfinite(dor[i])),
             "oracle_status": dstat[i], "oracle_type": "ef995_bounded"},
            {"method": "ANNiE", "query_id": i, "oracle_ndc": float(aor[i]),
             "oracle_attainable": bool(np.isfinite(aor[i])),
             "oracle_status": astat[i], "oracle_type": "unlimited_beam"},
        ]
        rec, ndc = lres["recall"][i], lres["stop_ndc"][i]
        hit = rec >= RT - 1e-9
        pred = float(np.power(2.0, max(lres["pred"][i], 0)))
        extra = max(0.0, ndc - lor[i]) if hit and np.isfinite(lor[i]) else np.nan
        perq.append({
            "dataset": "GIST1M", "query_id": i, "split": "test",
            "index_id": "gist_idx992000_M16_efc200", "method": "LAET",
            "model_version": "L2-log2C1", "training_condition": "n=2000,f=1260",
            "k": K, "target_recall": RT, "rtg_probability": np.nan,
            "oracle_ndc": float(lor[i]),
            "oracle_attainable": bool(np.isfinite(lor[i])),
            "oracle_status": lstat[i],
            "predicted_cost_raw": pred,
            "effective_budget": float(lres["budget"][i]),
            "predicted_recall_at_stop": np.nan,
            "actual_recall": float(rec), "target_hit": bool(hit),
            "stop_ndc": float(ndc),
            "stop_reason": "prediction" if np.isfinite(lres["stop_ndc"][i]) else "natural",
            "native_prediction_error": float(pred - utr_te["c1"][i]) if np.isfinite(utr_te["c1"][i]) else np.nan,
            "effective_budget_error": float(lres["budget"][i] - lor[i]) if np.isfinite(lor[i]) else np.nan,
            "extra_ndc": extra,
            "relative_extra_ndc": float(extra / max(lor[i], 1.0)) if np.isfinite(extra) and np.isfinite(lor[i]) else np.nan,
            "multiplier": float(lres["multiplier"]), "f": F_MAIN,
            "n_predictions": 1,
        })
        rec, ndc = ares["recall"][i], ares["stop_ndc"][i]
        hit = rec >= RT - 1e-9
        pred = float(ares["pred"][i])
        extra = max(0.0, ndc - aor[i]) if hit and np.isfinite(aor[i]) else np.nan
        perq.append({
            "dataset": "GIST1M", "query_id": i, "split": "test",
            "index_id": "gist_idx992000_M16_efc200", "method": "ANNiE-S",
            "model_version": "quantile-p0.95", "training_condition": "n=2000,homing",
            "k": K, "target_recall": RT, "rtg_probability": 0.95,
            "oracle_ndc": float(aor[i]),
            "oracle_attainable": bool(np.isfinite(aor[i])),
            "oracle_status": astat[i],
            "predicted_cost_raw": pred,
            "effective_budget": float(max(pred, utr_te["hom_ndc"][i])) if np.isfinite(utr_te["hom_ndc"][i]) else pred,
            "predicted_recall_at_stop": np.nan,
            "actual_recall": float(rec), "target_hit": bool(hit),
            "stop_ndc": float(ndc),
            "stop_reason": "prediction" if np.isfinite(ndc) else "natural",
            "native_prediction_error": float(pred - aor[i]) if np.isfinite(aor[i]) else np.nan,
            "effective_budget_error": float(pred - aor[i]) if np.isfinite(aor[i]) else np.nan,
            "extra_ndc": extra,
            "relative_extra_ndc": float(extra / max(aor[i], 1.0)) if np.isfinite(extra) and np.isfinite(aor[i]) else np.nan,
            "homing_ndc": float(utr_te["hom_ndc"][i]),
            "n_predictions": 1,
        })
        rec, ndc = dres["recall"][i], dres["stop_ndc"][i]
        hit = rec >= RT - 1e-9
        sp = float(dres["stop_pred"][i])
        extra = max(0.0, ndc - dor[i]) if hit and np.isfinite(dor[i]) else np.nan
        perq.append({
            "dataset": "GIST1M", "query_id": i, "split": "test",
            "index_id": "gist_idx992000_M16_efc200", "method": "DARTH",
            "model_version": "L2-recall", "training_condition": "n=2000,adaptive-ipi",
            "k": K, "target_recall": RT, "rtg_probability": np.nan,
            "oracle_ndc": float(dor[i]),
            "oracle_attainable": bool(np.isfinite(dor[i])),
            "oracle_status": dstat[i],
            "predicted_cost_raw": np.nan,
            "effective_budget": np.nan,
            "predicted_recall_at_stop": sp,
            "actual_recall": float(rec), "target_hit": bool(hit),
            "stop_ndc": float(ndc),
            "stop_reason": "prediction" if np.isfinite(sp) else "natural",
            "native_prediction_error": float(sp - rec) if np.isfinite(sp) else np.nan,
            "effective_budget_error": np.nan,
            "extra_ndc": extra,
            "relative_extra_ndc": float(extra / max(dor[i], 1.0)) if np.isfinite(extra) and np.isfinite(dor[i]) else np.nan,
            "ipi": dipi, "mpi": dmpi, "n_predictions": int(dres["calls"][i]),
        })

    # per-checkpoint DARTH
    ck = []
    for i, preds in enumerate(dres["preds"]):
        s, e = dtr_te["ptr"][i], dtr_te["ptr"][i + 1]
        segn = dtr_te["step_ndc"][s:e]
        segr = dtr_te["step_recall"][s:e]
        for cid, (nd, p) in enumerate(preds):
            j = int(np.searchsorted(segn, nd, side="left"))
            j = min(j, len(segn) - 1)
            actual = float(segr[j])
            ck.append({
                "query_id": i, "checkpoint_id": cid,
                "ndc_at_checkpoint": int(nd),
                "predicted_recall": float(p),
                "actual_recall": actual,
                "recall_prediction_error": float(p - actual),
                "target_recall": RT,
                "false_positive_stop": bool(p >= RT and actual < RT),
                "false_negative_continue": bool(p < RT and actual >= RT),
            })

    write_csv(perq, os.path.join(DATA, "per_query.csv"))
    write_csv(orows, os.path.join(DATA, "oracle_costs.csv"))
    write_csv(ck, os.path.join(DATA, "per_checkpoint.csv"))

    summ = [summary_row("LAET", lres, lor),
            summary_row("DARTH", dres, dor),
            summary_row("ANNiE-S", ares, aor)]
    write_csv(summ, os.path.join(ANA, "error_summary.csv"))

    # error diagnostics
    erows = []
    erows.append({"method": "LAET", **error_stats(
        np.power(2.0, np.maximum(lres["pred"], 0)), utr_te["c1"], "native_")})
    erows.append({"method": "LAET", **error_stats(
        lres["budget"], lor, "budget_")})
    erows.append({"method": "ANNiE-S", **error_stats(ares["pred"], aor, "cost_")})
    ckdf = pd.DataFrame(ck)
    erows.append({"method": "DARTH", **error_stats(
        ckdf["predicted_recall"], ckdf["actual_recall"], "checkpoint_")})
    qmean = ckdf.groupby("query_id")["recall_prediction_error"].mean()
    erows.append({"method": "DARTH", **error_stats(
        qmean.values, np.zeros(len(qmean)), "query_abs_")})
    write_csv(erows, os.path.join(ANA, "prediction_error_summary.csv"))

    # failure overlap
    overlap_rows = failure_overlap(pd.DataFrame(perq), utr_te)
    write_csv(overlap_rows, os.path.join(ANA, "failure_overlap.csv"))

    save_json({
        "experiment": "exp02_error_analysis", "dataset": "GIST1M",
        "k": K, "target_recall": RT, "rtg_p": 0.95,
        "n_train": a.n_train, "n_val": a.n_val, "n_test": a.n_test,
        "laet": {"f": F_MAIN, "mult": float(lres["multiplier"]),
                 "pred_max": lpm},
        "darth": {"ipi": dipi, "mpi": dmpi},
        "annie": {"homing": "calibrated", "p": 0.95},
        "oracle": {"LAET": "unlimited beam", "ANNiE": "unlimited beam",
                   "DARTH": "ef995 bounded"},
    }, os.path.join(EXP, "configs", "experiment_config.json"))
    log("main phase done")


def failure_overlap(df, utr):
    methods = ["LAET", "DARTH", "ANNiE-S"]
    sets = {}
    for m in methods:
        sub = df[df.method == m].sort_values("query_id")
        q = sub["query_id"].values
        fail = q[~sub["target_hit"].values.astype(bool)]
        succ = q[sub["target_hit"].values.astype(bool)]
        rel = sub[sub["target_hit"].values]["relative_extra_ndc"].values
        if len(rel) >= 10:
            thr = np.nanpercentile(rel, 90)
            eff = succ[np.nan_to_num(rel, nan=-1) >= thr]
        else:
            eff = np.array([], dtype=int)
        sets[m] = {"quality_fail": set(fail.tolist()),
                   "efficiency_fail": set(eff.tolist()),
                   "success": set(succ.tolist())}
    universe = np.arange(len(df) // 3)
    oracle = utr["cost_rt0.95"]
    ok = np.isfinite(oracle)
    strata = np.full(len(oracle), -1)
    if ok.sum() >= 30:
        qs = np.quantile(oracle[ok], [1 / 3, 2 / 3])
        strata[ok] = np.digitize(oracle[ok], qs)
    rows = []
    for ftype in ("quality_fail", "efficiency_fail"):
        for i, x in enumerate(methods):
            for y in methods[i + 1:]:
                A = np.array(sorted(sets[x][ftype])); B = np.array(sorted(sets[y][ftype]))
                r = stats.permutation_overlap_test(
                    A, B, universe, strata=strata,
                    n_perm=2000, seed=42)
                rows.append({"failure_type": ftype, "A": x, "B": y, **r})
        A = np.array(sorted(sets["LAET"][ftype]))
        B = np.array(sorted(sets["DARTH"][ftype]))
        C = np.array(sorted(sets["ANNiE-S"][ftype]))
        tri = len(set(A) & set(B) & set(C))
        rows.append({"failure_type": ftype, "A": "LAET∩DARTH∩ANNiE-S",
                     "B": "", "intersection": tri,
                     "n_A": tri, "n_B": 0, "jaccard": np.nan,
                     "P_B_given_A": np.nan, "p_value": np.nan})
    with open(os.path.join(DATA, "failure_sets.json"), "w") as f:
        json.dump({m: {k: sorted(v) for k, v in d.items()}
                   for m, d in sets.items()}, f, indent=2)
    return rows


# --------------------------------------------------------------------------
# size phase
# --------------------------------------------------------------------------
def phase_size(a):
    log("load traces for size sweep")
    utr_tr = win(load_trace("gist_k100_unlim_train" + a.tag), a.n_train)
    utr_va = win(load_trace("gist_k100_unlim_val" + a.tag), a.n_val)
    utr_te = win(load_trace("gist_k100_unlim_test" + a.tag), a.n_test)
    dtr_tr = win(load_trace("gist_k100_darth_train" + a.tag), a.n_train)
    dtr_te = win(load_trace("gist_k100_darth_test" + a.tag), a.n_test)
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]
    perm = np.random.default_rng(0).permutation(a.n_train)
    rows = []
    for size in (500, 1000, 2000):
        if size > a.n_train:
            continue
        for seed in (42, 43, 44):
            idx = perm[:size]
            log(f"size={size} seed={seed}")
            lm, lpm, _ = train_laet(utr_tr, qtr, idx, seed)
            lr = eval_laet(utr_va, qva, utr_te, qte, lm, lpm)
            dm, ipi, mpi, _ = train_darth(dtr_tr, qtr, idx, seed)
            dr = eval_darth(dtr_te, dm, ipi, mpi)
            am, _ = train_annie(utr_tr, qtr, idx, RT, seed)
            ar = eval_annie(utr_te, qte, am)
            lor = utr_te["cost_rt0.95"]; dor = dtr_te["cost_rt0.95"]
            for name, res, orc in (("LAET", lr, lor), ("DARTH", dr, dor),
                                   ("ANNiE-S", ar, lor)):
                r = summary_row(name, res, orc)
                rows.append({"train_size": size, "seed": seed, **r,
                             "multiplier": float(lr.get("multiplier", np.nan)) if name == "LAET" else np.nan,
                             "ipi": ipi if name == "DARTH" else np.nan,
                             "mpi": mpi if name == "DARTH" else np.nan})
    write_csv(rows, os.path.join(ANA, "training_size_summary.csv"))


# --------------------------------------------------------------------------
# distribution phase (D2: hard-query-enriched)
# --------------------------------------------------------------------------
def phase_dist(a):
    log("distribution experiment (hard-query enriched)")
    utr_tr = win(load_trace("gist_k100_unlim_train" + a.tag), a.n_train)
    utr_va = win(load_trace("gist_k100_unlim_val" + a.tag), a.n_val)
    utr_te = win(load_trace("gist_k100_unlim_test" + a.tag), a.n_test)
    dtr_tr = win(load_trace("gist_k100_darth_train" + a.tag), a.n_train)
    dtr_te = win(load_trace("gist_k100_darth_test" + a.tag), a.n_test)
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]
    cost = utr_tr["cost_rt0.95"]
    ok = np.isfinite(cost)
    size = 1000
    rng = np.random.default_rng(42)
    rows = []
    # D0 baseline: random 2000
    rand_idx = np.flatnonzero(ok)[rng.permutation(ok.sum())[:size]]
    # D2: 1000 hardest + 1000 random from the rest
    hard = np.argsort(np.where(ok, cost, -np.inf))[::-1][:size // 2]
    rest = np.array([i for i in np.flatnonzero(ok) if i not in set(hard.tolist())])
    easy = rng.permutation(rest)[:size // 2]
    hard_idx = np.concatenate([hard, easy])
    for cond, idx in (("D0_in_distribution", rand_idx),
                      ("D2_hard_enriched", hard_idx)):
        log(f"  {cond}")
        lm, lpm, _ = train_laet(utr_tr, qtr, idx, 42)
        lr = eval_laet(utr_va, qva, utr_te, qte, lm, lpm)
        dm, ipi, mpi, _ = train_darth(dtr_tr, qtr, idx, 42)
        dr = eval_darth(dtr_te, dm, ipi, mpi)
        am, _ = train_annie(utr_tr, qtr, idx, RT, 42)
        ar = eval_annie(utr_te, qte, am)
        lor = utr_te["cost_rt0.95"]; dor = dtr_te["cost_rt0.95"]
        for name, res, orc in (("LAET", lr, lor), ("DARTH", dr, dor),
                               ("ANNiE-S", ar, lor)):
            rows.append({"condition": cond, **summary_row(name, res, orc)})
    write_csv(rows, os.path.join(ANA, "distribution_summary.csv"))


# --------------------------------------------------------------------------
# recall-target phase
# --------------------------------------------------------------------------
def phase_rt(a):
    log("recall-target sensitivity")
    utr_tr = win(load_trace("gist_k100_unlim_train" + a.tag), a.n_train)
    utr_va = win(load_trace("gist_k100_unlim_val" + a.tag), a.n_val)
    utr_te = win(load_trace("gist_k100_unlim_test" + a.tag), a.n_test)
    dtr_tr = win(load_trace("gist_k100_darth_train" + a.tag), a.n_train)
    dtr_te = win(load_trace("gist_k100_darth_test" + a.tag), a.n_test)
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]
    idx = np.arange(a.n_train)
    lm, lpm, _ = train_laet(utr_tr, qtr, idx, 42)
    dm, _, _, _ = train_darth(dtr_tr, qtr, idx, 42)
    rows = []
    for rt in (0.90, 0.95, 0.99):
        log(f"  rt={rt}")
        lr = eval_laet(utr_va, qva, utr_te, qte, lm, lpm, rt=rt)
        dbar = float(np.nanmean(dtr_tr[f"cost_rt{rt}"]))
        ipi, mpi = max(dbar / 2.0, 32.0), max(dbar / 10.0, 8.0)
        dr = eval_darth(dtr_te, dm, ipi, mpi, rt=rt)
        am, _ = train_annie(utr_tr, qtr, idx, rt, 42)
        ar = eval_annie(utr_te, qte, am, rt=rt)
        lor, _ = oracle_columns(utr_te, rt, "LAET")
        dor, _ = oracle_columns(dtr_te, rt, "DARTH")
        for name, res, orc, extra in (
                ("LAET", lr, lor, {"multiplier": lr["multiplier"]}),
                ("DARTH", dr, dor, {"ipi": ipi, "mpi": mpi}),
                ("ANNiE-S", ar, lor, {"p": 0.95})):
            rows.append({"target_recall": rt, **summary_row(name, res, orc, rt),
                         **extra})
    write_csv(rows, os.path.join(ANA, "recall_target_summary.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["main", "size", "dist", "rt", "all"],
                    default="main")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.phase in ("main", "all"):
        phase_main(a)
    if a.phase in ("size", "all"):
        phase_size(a)
    if a.phase in ("dist", "all"):
        phase_dist(a)
    if a.phase in ("rt", "all"):
        phase_rt(a)
