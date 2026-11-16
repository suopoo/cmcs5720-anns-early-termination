"""Experiment 3 - Asymmetric loss and prediction timing (GIST1M, k=10).

3A: LAET L2 vs quantile tau={0.50,0.75,0.90,0.95}; DARTH L2 vs
    tau={0.50,0.25,0.10,0.05}.  LAET is evaluated with (A1) a fixed
    multiplier and (A2) a per-model recalibrated multiplier.
3B: LAET preliminary-search length f in {630,945,1260,1575} x {L2, best
    quantile}; DARTH interval configs P0-P3 x {original, quantile}.
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
from anns_exp import gbdt, simulate as S, stats
from anns_exp.exp_common import (load_trace, grid_query_index, summarize,
                                 error_stats, write_csv, save_json, log)
from anns_exp.trace import load_split

EXP = paths.exp_dir("exp03")
DATA = os.path.join(EXP, "data")
ANA = os.path.join(EXP, "analysis")
MODELS = os.path.join(EXP, "models")
REP = os.path.join(EXP, "reports")
for d in (DATA, ANA, MODELS, REP, os.path.join(EXP, "configs"),
          os.path.join(EXP, "logs"), os.path.join(EXP, "figures")):
    os.makedirs(d, exist_ok=True)

K, RT = 10, 0.90
F0 = 1260
LAET_TAUS = [("L0", None), ("L1", 0.50), ("L2", 0.75), ("L3", 0.90),
             ("L4", 0.95)]
DARTH_TAUS = [("D0", None), ("D1", 0.50), ("D2", 0.25), ("D3", 0.10),
              ("D4", 0.05)]


def win(tr, n):
    if n is None or n >= len(tr["ptr"]) - 1:
        return tr
    out = {"ptr": tr["ptr"][:n + 1].copy()}
    m = out["ptr"][-1]
    for key in ("step_ndc", "step_recall"):
        out[key] = tr[key][:m]
    for key in list(tr.keys()):
        if key.startswith("tail_f") or key.startswith("cost_rt") or key in (
                "c1", "reason", "suffix"):
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


def laet_X(tr, q, f):
    return np.concatenate([q, tr[f"tail_f{f}"]], axis=1)


def fit_laet(tr, q, f, tau, seed=42):
    y = np.log2(np.maximum(tr["c1"], 1))
    X = laet_X(tr, q, f)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    kw = dict(seed=seed, n_estimators=100, learning_rate=0.2,
              num_leaves=31, min_child_samples=20)
    model = gbdt.train_lgb(X[ok], y[ok],
                           objective="regression" if tau is None else "quantile",
                           alpha=tau, **kw)
    return model, float(2.0 ** y[ok].max()), int(ok.sum()), X[ok], y[ok]


def fit_darth(dtr, tau, seed=42):
    X = dtr["obs_feat"]; y = dtr["obs_recall"]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    model = gbdt.train_lgb(X[ok], y[ok],
                           objective="regression" if tau is None else "quantile",
                           alpha=tau, seed=seed, n_estimators=100,
                           learning_rate=0.1, num_leaves=31,
                           min_child_samples=min(1000, max(20, ok.sum() // 100)))
    return model, int(ok.sum())


def laet_eval_rows(name, tau, res, tr, oracle, pred):
    rec = res["recall"]; ndc = res["stop_ndc"]
    hit = rec >= RT - 1e-9
    extra = np.maximum(0.0, ndc - oracle)
    return {
        "config": name, "quantile_tau": tau,
        "n": int(len(rec)), "mean_recall": float(np.nanmean(rec)),
        "hit_rate": float(np.nanmean(hit)),
        "avg_ndc": float(np.nanmean(ndc)),
        "median_ndc": float(np.nanmedian(ndc)),
        "p95_ndc": float(np.nanpercentile(ndc, 95)),
        "extra_ndc_mean": float(np.nanmean(extra[hit & np.isfinite(oracle)])),
        "truncated": int(res.get("truncated", np.zeros(len(rec), bool)).sum()),
        **error_stats(np.power(2.0, np.maximum(pred, 0)), tr["c1"], "pred_"),
    }


def main_3a(a):
    ltr_tr = win(load_trace("gist_k10_laet_train" + a.tag), a.n_train)
    ltr_va = win(load_trace("gist_k10_laet_val" + a.tag), a.n_val)
    ltr_te = win(load_trace("gist_k10_laet_test" + a.tag), a.n_test)
    dtr_tr = win(load_trace("gist_k10_darth_train" + a.tag), a.n_train)
    dtr_va = win(load_trace("gist_k10_darth_val" + a.tag), a.n_val)
    dtr_te = win(load_trace("gist_k10_darth_test" + a.tag), a.n_test)
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]

    rows, perq, predrows = [], [], []
    models, preds_val, preds_test = {}, {}, {}
    stat_rows = []
    laet_res, darth_res = {}, {}
    oracle_l = ltr_te["cost_rt0.9"]
    for name, tau in LAET_TAUS:
        log(f"LAET {name} tau={tau}")
        model, pmax, n, _, _ = fit_laet(ltr_tr, qtr, F0, tau, a.seed)
        models[name] = (model, pmax)
        joblib.dump(model, os.path.join(MODELS, f"laet_{name}.joblib"))
        pv = gbdt.predict(model, laet_X(ltr_va, qva, F0))
        pt = gbdt.predict(model, laet_X(ltr_te, qte, F0))
        preds_val[name], preds_test[name] = pv, pt
        # A2: per-model recalibration
        mult2, *_ = S.calibrate_multiplier(pv, ltr_va, F0, pmax, target=RT)
        res2 = S.laet_from_trace(ltr_te, laet_X(ltr_te, qte, F0), model,
                                 F0, mult2, pmax)
        laet_res[name] = res2
        rows.append({"experiment": "3A-LAET", "setting": "A2_recalibrated",
                     "multiplier": mult2, "pred_max": pmax,
                     **laet_eval_rows(name, tau, res2, ltr_te, oracle_l, pt)})
        for i in range(len(res2["recall"])):
            perq.append({
                "experiment": "3A-LAET", "setting": "A2_recalibrated",
                "config": name, "quantile_tau": tau, "query_id": i,
                "target_recall": RT, "actual_recall": float(res2["recall"][i]),
                "stop_ndc": float(res2["stop_ndc"][i]),
                "oracle_ndc": float(oracle_l[i]),
                "predicted_ndc": float(np.power(2.0, max(pt[i], 0))),
                "true_c1_ndc": float(ltr_te["c1"][i]),
                "multiplier": mult2,
            })
        predrows += [{"experiment": "3A-LAET", "config": name,
                      "split": "test", "query_id": i,
                      "pred_log2": float(pt[i]),
                      "pred_ndc": float(np.power(2.0, max(pt[i], 0))),
                      "true_c1_ndc": float(ltr_te["c1"][i])}
                     for i in range(len(pt))]

    # A1 fixed multiplier: original L2 calibration, applied to every model
    base_model, base_pmax = models["L0"]
    pv0 = preds_val["L0"]
    m_fixed, *_ = S.calibrate_multiplier(pv0, ltr_va, F0, base_pmax,
                                         target=RT)
    log(f"fixed multiplier (from L0) = {m_fixed:.3f}")
    for name, tau in LAET_TAUS:
        model, pmax = models[name]
        res = S.laet_from_trace(ltr_te, laet_X(ltr_te, qte, F0), model,
                                F0, m_fixed, pmax)
        rows.append({"experiment": "3A-LAET", "setting": "A1_fixed",
                     "multiplier": m_fixed, "pred_max": pmax,
                     **laet_eval_rows(name, tau, res, ltr_te, oracle_l,
                                      preds_test[name])})

    # ---------------- DARTH 3A ----------------
    dbar = float(np.nanmean(dtr_tr["cost_rt0.9"]))
    ipi, mpi = max(dbar / 2.0, 32.0), max(dbar / 10.0, 8.0)
    log(f"DARTH ipi={ipi:.0f} mpi={mpi:.0f}")
    oracle_d = dtr_te["cost_rt0.9"]
    dmodels, dpv, dpt = {}, {}, {}
    for name, tau in DARTH_TAUS:
        log(f"DARTH {name} tau={tau}")
        model, n = fit_darth(dtr_tr, tau, a.seed)
        dmodels[name] = model
        joblib.dump(model, os.path.join(MODELS, f"darth_{name}.joblib"))
        dpv[name] = gbdt.predict(model, dtr_va["tr_darth"])
        dpt[name] = gbdt.predict(model, dtr_te["tr_darth"])
        res = S.darth_adaptive_from_steps(dtr_te, model, RT, ipi, mpi)
        darth_res[name] = res
        rec, ndc = res["recall"], res["stop_ndc"]
        hit = rec >= RT - 1e-9
        row = {"experiment": "3A-DARTH", "config": name, "quantile_tau": tau,
               "n": len(rec), "mean_recall": float(np.nanmean(rec)),
               "hit_rate": float(np.nanmean(hit)),
               "avg_ndc": float(np.nanmean(ndc)),
               "median_ndc": float(np.nanmedian(ndc)),
               "p95_ndc": float(np.nanpercentile(ndc, 95)),
               "calls_mean": float(np.mean(res["calls"])),
               "ipi": ipi, "mpi": mpi}
        row.update(error_stats(dpt[name], dtr_te["tr_recall"], "checkpoint_"))
        rows.append(row)
        for i in range(len(rec)):
            perq.append({
                "experiment": "3A-DARTH", "setting": "adaptive",
                "config": name, "quantile_tau": tau, "query_id": i,
                "target_recall": RT, "actual_recall": float(rec[i]),
                "stop_ndc": float(ndc[i]), "oracle_ndc": float(oracle_d[i]),
                "predicted_recall_at_stop": float(res["stop_pred"][i]),
                "n_predictions": int(res["calls"][i]),
            })
        for j in range(len(dpt[name])):
            predrows.append({"experiment": "3A-DARTH", "config": name,
                             "split": "test",
                             "checkpoint_id": j,
                             "ndc_at_checkpoint": int(dtr_te["tr_ndc"][j]),
                             "predicted_recall": float(dpt[name][j]),
                             "actual_recall": float(dtr_te["tr_recall"][j])})

    # offline comparison at identical checkpoints
    off = []
    qi = grid_query_index(dtr_te["tr_ptr"])
    for name, tau in DARTH_TAUS:
        err = dpt[name] - dtr_te["tr_recall"]
        for j in range(len(err)):
            off.append({"config": name, "quantile_tau": tau,
                        "query_id": int(qi[j]),
                        "ndc_at_checkpoint": int(dtr_te["tr_ndc"][j]),
                        "predicted_recall": float(dpt[name][j]),
                        "actual_recall": float(dtr_te["tr_recall"][j]),
                        "error": float(err[j]),
                        "false_positive_risk": bool(dpt[name][j] >= RT and
                                                    dtr_te["tr_recall"][j] < RT)})
    write_csv(off, os.path.join(DATA, "offline_predictions_3A_darth.csv"))

    # paired statistics vs the original-loss baseline
    for name in laet_res:
        if name == "L0":
            continue
        for metric in ("recall", "stop_ndc"):
            d = stats.paired_diff(laet_res[name][metric],
                                  laet_res["L0"][metric])
            stat_rows.append({"experiment": "3A-LAET", "setting": "A2",
                              "comparison": f"{name} vs L0", "metric": metric,
                              **d,
                              "cliffs_delta": stats.cliffs_delta(
                                  laet_res[name][metric],
                                  laet_res["L0"][metric])})
    for name in darth_res:
        if name == "D0":
            continue
        for metric in ("recall", "stop_ndc"):
            d = stats.paired_diff(darth_res[name][metric],
                                  darth_res["D0"][metric])
            stat_rows.append({"experiment": "3A-DARTH",
                              "comparison": f"{name} vs D0", "metric": metric,
                              **d,
                              "cliffs_delta": stats.cliffs_delta(
                                  darth_res[name][metric],
                                  darth_res["D0"][metric])})
    write_csv(stat_rows, os.path.join(ANA, "statistical_tests_3A.csv"))

    # quality-cost curves for the figure
    crows = []
    mset = np.geomspace(0.1, 40.0, 40)
    for name, tau in LAET_TAUS:
        model, pmax = models[name]
        p = preds_test[name]
        pm = 2.0 ** np.maximum(p, 0)
        for m in mset:
            B = np.maximum(F0, np.minimum(pm * m, pmax))
            rr = S.laet_with_budget(ltr_te, B)
            crows.append({"method": "LAET", "config": name, "knob": "multiplier",
                          "value": float(m),
                          "mean_recall": float(np.nanmean(rr["recall"])),
                          "mean_ndc": float(np.nanmean(rr["stop_ndc"]))})
    for name, tau in DARTH_TAUS:
        model = dmodels[name]
        for t in np.linspace(0.5, 0.99, 30):
            rr = S.darth_adaptive_from_steps(dtr_te, model, float(t), ipi, mpi)
            crows.append({"method": "DARTH", "config": name,
                          "knob": "stop_threshold", "value": float(t),
                          "mean_recall": float(np.nanmean(rr["recall"])),
                          "mean_ndc": float(np.nanmean(rr["stop_ndc"]))})
    write_csv(crows, os.path.join(DATA, "quality_cost_curves_3A.csv"))

    # best tau chosen on validation (native MAE, no test leakage)
    laet_val_mae = {}
    for name, tau in LAET_TAUS:
        pv = preds_val[name]
        yv = np.log2(np.maximum(ltr_va["c1"], 1))
        ok = np.isfinite(yv)
        laet_val_mae[name] = float(np.mean(np.abs(pv[ok] - yv[ok])))
    darth_val_mae = {}
    for name, tau in DARTH_TAUS:
        ok = np.isfinite(dtr_va["tr_recall"])
        darth_val_mae[name] = float(np.mean(np.abs(
            dpv[name][ok] - dtr_va["tr_recall"][ok])))
    best_l = min((k for k in laet_val_mae if k != "L0"),
                 key=laet_val_mae.get)
    best_d = min((k for k in darth_val_mae if k != "D0"),
                 key=darth_val_mae.get)
    save_json({"laet_val_mae": laet_val_mae, "darth_val_mae": darth_val_mae,
               "selected_laet": best_l, "selected_darth": best_d,
               "fixed_multiplier": m_fixed, "ipi": ipi, "mpi": mpi},
              os.path.join(EXP, "configs", "selection.json"))
    save_json({
        "experiment": "exp03_asymmetric_loss", "dataset": "GIST1M",
        "k": K, "target_recall": RT, "checkpoint": F0,
        "train_queries": a.n_train, "val_queries": a.n_val,
        "test_queries": a.n_test, "seed": a.seed,
        "laet_losses": [{"group": g, "tau": t} for g, t in LAET_TAUS],
        "darth_losses": [{"group": g, "tau": t} for g, t in DARTH_TAUS],
        "ipi": ipi, "mpi": mpi, "fixed_multiplier": m_fixed,
        "selected_laet": best_l, "selected_darth": best_d,
    }, os.path.join(EXP, "configs", "experiment_config.json"))
    write_csv(rows, os.path.join(DATA, "metrics_3A.csv"))
    write_csv(perq, os.path.join(DATA, "per_query_3A.csv"))
    write_csv(predrows, os.path.join(DATA, "prediction_traces_3A.csv"))
    return best_l, best_d, ipi, mpi, models, dmodels


def main_3b(a, best_l, best_d, ipi, mpi):
    ltr_tr = win(load_trace("gist_k10_laet_train" + a.tag), a.n_train)
    ltr_va = win(load_trace("gist_k10_laet_val" + a.tag), a.n_val)
    ltr_te = win(load_trace("gist_k10_laet_test" + a.tag), a.n_test)
    dtr_tr = win(load_trace("gist_k10_darth_train" + a.tag), a.n_train)
    dtr_te = win(load_trace("gist_k10_darth_test" + a.tag), a.n_test)
    qtr, _ = load_split("train"); qtr = qtr[:a.n_train]
    qva, _ = load_split("val"); qva = qva[:a.n_val]
    qte, _ = load_split("test"); qte = qte[:a.n_test]
    oracle_l = ltr_te["cost_rt0.9"]
    oracle_d = dtr_te["cost_rt0.9"]
    rows, perq = [], []
    # ---- LAET: f x {L2, best quantile} ----
    taus = [("L2", None), (best_l, dict(LAET_TAUS)[best_l])]
    for f in (630, 945, 1260, 1575):
        for name, tau in taus:
            log(f"3B LAET f={f} {name}")
            model, pmax, n, X, y = fit_laet(ltr_tr, qtr, f, tau, a.seed)
            joblib.dump(model, os.path.join(MODELS, f"laet_3b_f{f}_{name}.joblib"))
            pv = gbdt.predict(model, laet_X(ltr_va, qva, f))
            mult, *_ = S.calibrate_multiplier(pv, ltr_va, f, pmax, target=RT)
            pt = gbdt.predict(model, laet_X(ltr_te, qte, f))
            res = S.laet_from_trace(ltr_te, laet_X(ltr_te, qte, f), model,
                                    f, mult, pmax)
            r = laet_eval_rows(f"f{f}_{name}", tau, res, ltr_te, oracle_l, pt)
            r.update({"experiment": "3B-LAET", "f": f, "loss": name,
                      "multiplier": mult})
            rows.append(r)
            for i in range(len(res["recall"])):
                perq.append({"experiment": "3B-LAET", "f": f, "loss": name,
                             "query_id": i, "actual_recall": float(res["recall"][i]),
                             "stop_ndc": float(res["stop_ndc"][i]),
                             "oracle_ndc": float(oracle_l[i])})
    # ---- DARTH: P0-P3 x {original, best quantile} ----
    configs = [("P0", ipi, mpi), ("P1", 0.5 * ipi, 0.5 * mpi),
               ("P2", 1.5 * ipi, 1.5 * mpi), ("P3", ipi, 0.5 * mpi)]
    for pname, pi_, mp_ in configs:
        pi_, mp_ = max(1.0, min(pi_, 1e9)), max(1.0, mp_)
        if mp_ > pi_:
            mp_ = pi_
        for name, tau in [("D0", None), (best_d, dict(DARTH_TAUS)[best_d])]:
            log(f"3B DARTH {pname} {name}")
            model, _ = fit_darth(dtr_tr, tau, a.seed)
            res = S.darth_adaptive_from_steps(dtr_te, model, RT, pi_, mp_)
            rec, ndc = res["recall"], res["stop_ndc"]
            hit = rec >= RT - 1e-9
            r = {"experiment": "3B-DARTH", "interval": pname,
                 "loss": name, "ipi": pi_, "mpi": mp_,
                 "n": len(rec), "mean_recall": float(np.nanmean(rec)),
                 "hit_rate": float(np.nanmean(hit)),
                 "avg_ndc": float(np.nanmean(ndc)),
                 "median_ndc": float(np.nanmedian(ndc)),
                 "p95_ndc": float(np.nanpercentile(ndc, 95)),
                 "calls_mean": float(np.mean(res["calls"]))}
            r.update(error_stats(gbdt.predict(model, dtr_te["tr_darth"]),
                                 dtr_te["tr_recall"], "checkpoint_"))
            rows.append(r)
            for i in range(len(rec)):
                perq.append({"experiment": "3B-DARTH", "interval": pname,
                             "loss": name, "query_id": i,
                             "actual_recall": float(rec[i]),
                             "stop_ndc": float(ndc[i]),
                             "oracle_ndc": float(oracle_d[i]),
                             "n_predictions": int(res["calls"][i])})
    write_csv(rows, os.path.join(DATA, "metrics_3B.csv"))
    write_csv(perq, os.path.join(DATA, "per_query_3B.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["3a", "3b", "all"], default="all")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    best_l, best_d, ipi, mpi = "L3", "D3", 100.0, 10.0
    if a.phase in ("3a", "all"):
        best_l, best_d, ipi, mpi, _, _ = main_3a(a)
    if a.phase in ("3b", "all"):
        main_3b(a, best_l, best_d, ipi, mpi)
