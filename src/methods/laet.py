"""LAET (SIGMOD'20) -- learned adaptive early termination, HNSW variant.

Faithful to the paper's HNSW case (§4.3):
  * target  : log2(minimum NDC to find the ground-truth 1-NN)
  * features: F0 query vector, F1 d_start, F2 d_1st, F3 d_10th,
              F4 d_1st/d_start, F5 d_10th/d_start   (Table 5)
  * online  : search the first f distance calcs, predict once, then continue
              until  NDC >= max(f, max_thresh, multiplier * 2^y_hat)
  * model   : LightGBM, L2 loss, 100 rounds, lr=0.2  (§4.1.3)

The `multiplier` is LAET's *implicit* recall knob: it is tuned per target
recall on a validation set, exactly as DARTH §4.2.5 describes when it
compares against LAET.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from core.search import run_query


def _features(ctx, use_query):
    d = ctx.dists()
    if d.size == 0:
        d1 = d10 = ctx.pop_dist
    else:
        d1 = float(d[0])
        d10 = float(d[min(9, d.size - 1)])
    dstart = float(ctx.first_nn)
    tail = [dstart, d1, d10,
            d1 / max(dstart, 1e-9), d10 / max(dstart, 1e-9)]
    if use_query:
        return np.concatenate([ctx.q.astype(np.float32),
                               np.asarray(tail, dtype=np.float32)])
    return np.asarray(tail, dtype=np.float32)


def estimate_top1_cost(g, queries, gt, k, max_ndc):
    """C1(q) = NDC at which the GT 1-NN is first touched (nan if never)."""
    out = np.full(len(queries), np.nan, dtype=np.float64)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        o = run_query(g, q, k=k, ef=None, gt_set={int(gts[0])},
                      max_ndc=max_ndc, stop=lambda c: c.hit >= 1)
        if o["recall"] and o["recall"] > 0:
            out[i] = o["ndc"]
    return out


def choose_f(costs, pct=20.0, floor=32):
    """Observation point f: the pct-th percentile of C1 (paper grid-searched it;
    the design doc re-calibrates it as a percentile of the empirical CDF)."""
    c = costs[np.isfinite(costs)]
    if c.size == 0:
        return floor
    return int(max(floor, np.percentile(c, pct)))


def collect_training(g, queries, gt, k, f, max_ndc, use_query=True):
    """One row per training query: features snapshot at NDC>=f, label log2(C1).

    Queries that find their 1-NN before reaching f cannot provide the
    intermediate-result features; they are re-run once with a stop at f
    (cheap, they are the easy ones).
    """
    X, y = [], []
    for q, gts in zip(queries, gt):
        gid = {int(gts[0])}
        st = {"feat": None}
        o = run_query(g, q, k=k, ef=None, gt_set=gid, max_ndc=max_ndc,
                      stop=lambda c, st=st: (
                          st.__setitem__("feat", _features(c, use_query))
                          if (st["feat"] is None and c.ndc >= f) else None,
                          c.hit >= 1)[1])
        if o["recall"] is None or o["recall"] <= 0:
            continue
        if st["feat"] is None:          # found before f -> collect features at f
            def stop_at_f(c, st=st):
                st["feat"] = _features(c, use_query)
                return c.ndc >= f
            run_query(g, q, k=k, ef=None, max_ndc=f + 4096, stop=stop_at_f)
        if st["feat"] is None:
            continue
        X.append(st["feat"])
        y.append(np.log2(max(o["ndc"], 1.0)))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class LAET:
    """Budget rule taken verbatim from HNSW.cpp:search_from_candidate_unbounded_pred

        term_cond = min(pred_max, ceil(2^max(pred,0) * efSearch / 100))

    with pred_max = max training ground truth (a *ceiling*) and efSearch/100
    playing the role of the tunable multiplier.  The searched prefix f is the
    lower bound.
    """

    def __init__(self, model, f, pred_max, use_query=True):
        self.model = model
        self.f = int(f)
        self.pred_max = float(pred_max)
        self.use_query = use_query

    def budget(self, ctx, mult):
        p = float(self.model.predict(_features(ctx, self.use_query)[None, :])[0])
        b = min((2.0 ** max(p, 0.0)) * mult, self.pred_max)
        return max(b, float(self.f)), p


def train(g, queries, gt, k, max_ndc, use_query=True, pct=20.0,
          n_estimators=100, learning_rate=0.2, seed=0):
    costs = estimate_top1_cost(g, queries, gt, k, max_ndc)
    f = choose_f(costs, pct=pct)
    X, y = collect_training(g, queries, gt, k, f, max_ndc, use_query)
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=n_estimators,
        learning_rate=learning_rate, num_leaves=31, min_child_samples=20,
        subsample=1.0, colsample_bytree=1.0, random_state=seed,
        n_jobs=4, verbose=-1)
    model.fit(X, y)
    return LAET(model, f, float(2.0 ** y.max()), use_query)


def run_laet(g, queries, gt, k, method, mult, max_ndc):
    """Online LAET: one prediction after f distance calcs."""
    rec = np.empty(len(queries), dtype=np.float32)
    ndc = np.empty(len(queries), dtype=np.float64)
    preds = np.empty(len(queries), dtype=np.float64)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        st = {"b": None, "p": np.nan}
        def stop(c, st=st):
            if st["b"] is None:
                if c.ndc >= method.f:
                    st["b"], st["p"] = method.budget(c, mult)
                return False
            return c.ndc >= st["b"]
        o = run_query(g, q, k=k, ef=None, gt_set=set(gts[:k].tolist()),
                      max_ndc=max_ndc, stop=stop)
        rec[i] = o["recall"]
        ndc[i] = o["ndc"]
        preds[i] = st["p"]
    return {"recall": rec, "ndc": ndc, "pred_log2": preds}


def tune_multiplier(g, queries, gt, k, method, max_ndc, target,
                    mults=None):
    """Sweep the multiplier on validation queries with a single traced pass.

    For each query we run the unlimited-beam search once and record
    (NDC, recall) points; the recall at any budget is then a lookup, so the
    whole multiplier sweep costs one search per query instead of one per
    (query, multiplier).
    """
    if mults is None:
        mults = np.geomspace(0.1, 60.0, 32)
    rec_at = np.zeros((len(queries), len(mults)), dtype=np.float32)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        st = {"feat": None}
        def stop(c, st=st):
            if st["feat"] is None and c.ndc >= method.f:
                st["feat"] = _features(c, method.use_query)
            return False
        o = run_query(g, q, k=k, ef=None, gt_set=set(gts[:k].tolist()),
                      max_ndc=max_ndc, stop=stop, trace_every=1)
        if st["feat"] is None or not o["trace"]:
            rec_at[i] = o["recall"]
            continue
        p = float(method.model.predict(st["feat"][None, :])[0])
        tr = np.asarray([(a, c) for a, _, c in o["trace"]])   # (ndc, recall)
        for j, m in enumerate(mults):
            b = max(float(method.f),
                    min((2.0 ** max(p, 0.0)) * m, method.pred_max))
            idx = np.searchsorted(tr[:, 0], b, side="right") - 1
            rec_at[i, j] = tr[idx, 1] if idx >= 0 else tr[0, 1]
    mean_rec = rec_at.mean(axis=0)
    # smallest multiplier whose mean recall reaches the target (else the best)
    ok = np.flatnonzero(mean_rec >= target)
    j = int(ok[0]) if ok.size else int(np.argmax(mean_rec))
    return float(mults[j]), mean_rec, mults
