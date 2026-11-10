"""DARTH (SIGMOD'25) -- declarative target recall via early termination.

Faithful to §3:
  * predictor target : the *current* recall (not the remaining cost)
  * features         : the 11 of Table 1
  * online           : call the predictor every p_i distance calcs,
                       p_i = m_pi + (i_pi - m_pi) * (R_t - R_hat)
                       stop as soon as R_hat >= R_t
  * hyper-parameters : i_pi = d_bar(R_t)/2, m_pi = d_bar(R_t)/10  (§3.2.2)
    -- the heuristic choice, i.e. DARTH runs untuned, as the paper claims.

The base search is plain HNSW with the efSearch the index was configured
with (high enough to reach >= 0.99 recall), exactly like DARTH's default
speedup baseline.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from core.search import run_query


def features(ctx):
    """DARTH Table 1: 3 index + 3 NN-distance + 5 NN-stat features."""
    d = ctx.knn_dists()
    if d.size:
        first = ctx.first_nn
        closest = float(d[0])
        furthest = float(d[-1])
        avg = float(d.mean())
        var = float(d.var())
        med = float(np.median(d))
        p25 = float(np.percentile(d, 25))
        p75 = float(np.percentile(d, 75))
    else:
        first = closest = furthest = ctx.pop_dist
        avg = var = med = p25 = p75 = ctx.pop_dist
    return np.asarray([ctx.step, ctx.ndc, ctx.inserts,
                       first, closest, furthest,
                       avg, var, med, p25, p75], dtype=np.float32)


def collect_training(g, queries, gt, k, ef, rt, max_ndc,
                     delta=5, max_obs=250):
    """Observations (features, current recall) every `delta` NDC.

    Also returns d_bar(R_t): the mean NDC at which the target recall is
    first reached, used for DARTH's heuristic prediction intervals.
    """
    X, y = [], []
    dists_rt = np.empty(len(queries), dtype=np.float64)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        target = set(gts[:k].tolist())
        st = {"last": -delta, "hit_t": np.nan, "nobs": 0}
        def stop(c, st=st):
            if np.isnan(st["hit_t"]) and c.hit >= rt * k - 1e-9:
                st["hit_t"] = c.ndc
            if c.ndc - st["last"] >= delta and st["nobs"] < max_obs:
                st["last"] = c.ndc
                st["nobs"] += 1
                X.append(features(c))
                y.append(c.hit / k)
            return False
        o = run_query(g, q, k=k, ef=ef, gt_set=target, max_ndc=max_ndc,
                      stop=stop)
        dists_rt[i] = st["hit_t"] if not np.isnan(st["hit_t"]) else o["ndc"]
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), dists_rt


class DARTH:
    def __init__(self, model, rt, ipi, mpi):
        self.model = model
        self.rt = float(rt)
        self.ipi = float(ipi)
        self.mpi = float(mpi)

    @classmethod
    def train(cls, X, y, dists_rt, rt, seed=0):
        model = lgb.LGBMRegressor(
            objective="regression", n_estimators=100, learning_rate=0.1,
            num_leaves=31, min_child_samples=min(1000, max(20, len(X) // 100)),
            random_state=seed, n_jobs=4, verbose=-1)
        model.fit(X, y)
        d_bar = float(np.mean(dists_rt))
        return cls(model, rt, max(d_bar / 2.0, 32.0), max(d_bar / 10.0, 8.0))


def run_darth(g, queries, gt, k, ef, method, max_ndc):
    """Online DARTH: adaptive prediction intervals, untuned."""
    n = len(queries)
    rec = np.empty(n, dtype=np.float32)
    ndc = np.empty(n, dtype=np.float64)
    calls = np.empty(n, dtype=np.int32)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        st = {"last": 0.0, "pi": method.ipi, "calls": 0}
        def stop(c, st=st):
            if c.ndc - st["last"] < st["pi"]:
                return False
            st["calls"] += 1
            st["last"] = c.ndc
            p = float(method.model.predict(features(c)[None, :])[0])
            if p >= method.rt:
                return True
            st["pi"] = method.mpi + (method.ipi - method.mpi) * (method.rt - p)
            return False
        o = run_query(g, q, k=k, ef=ef, gt_set=set(gts[:k].tolist()),
                      max_ndc=max_ndc, stop=stop)
        rec[i] = o["recall"]
        ndc[i] = o["ndc"]
        calls[i] = st["calls"]
    return {"recall": rec, "ndc": ndc, "calls": calls}
