"""ANNiE (VLDB'26) -- learned query cost estimator + ANNiE-S.

Implements §4 of the paper:
  * homing phase (Alg. 2): sliding window w of popped distances vs. the
    top of a k'-sized BSF heap; the search switches to the refining phase
    when  min(window) > dists[k'-1].
  * approximate local subgraph (Def. 7 approximation): the s closest nodes
    accessed during homing, plus out-neighbours if the set is short.
  * graph features: clustering coefficient (density) of the BSF kNN nodes
    and of the whole subgraph, plus h-hop reachability from each BSF node,
    computed with Python ints as bitsets (the POPCNT/BFS of Alg. 4).
  * model: GBDT with *quantile* loss at tau = p  ->  Pr[Chat >= C] >= p.
  * ANNiE-S: predict once at the phase transition, stop at ndc >= Chat.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import lightgbm as lgb

from core.search import run_query


# --------------------------------------------------------------------------
# approximate local subgraph + graph features
# --------------------------------------------------------------------------
def _collect_subgraph(g, ctx, pops, s):
    """s closest accessed nodes, induced subgraph, adjacency as sets+bitsets."""
    dm = {}
    for d, i in pops:
        if i not in dm or d < dm[i]:
            dm[i] = d
    for dn, i in ctx.res:
        d = -dn
        if i not in dm or d < dm[i]:
            dm[i] = d
    order = sorted(dm.items(), key=lambda kv: kv[1])
    ids = [i for i, _ in order[:s]]
    seen = set(ids)
    if len(ids) < s:                       # top up with out-neighbours
        for v in list(ids):
            if len(ids) >= s:
                break
            for u in g.neighbors_of(v, 0):
                u = int(u)
                if u >= 0 and u not in seen:
                    seen.add(u)
                    ids.append(u)
                    if len(ids) >= s:
                        break
    idx = {v: j for j, v in enumerate(ids)}
    n = len(ids)
    adj = [set() for _ in range(n)]
    for v in ids:
        j = idx[v]
        for u in g.neighbors_of(v, 0):
            u = int(u)
            if u >= 0 and u != v and u in idx:
                adj[j].add(idx[u])
    mask = [(1 << n) - 1 & sum(1 << w for w in adj[j]) for j in range(n)]
    return ids, adj, mask


def _clustering(adj):
    n = len(adj)
    c = np.zeros(n, dtype=np.float32)
    for v in range(n):
        nb = adj[v]
        d = len(nb)
        if d >= 2:
            cnt = 0
            for u in nb:
                cnt += len(nb & adj[u])
            c[v] = cnt / (d * (d - 1))
    return c


def _reach(mask, sources, h, n):
    """Average number of newly reachable nodes at each hop (Alg. 4)."""
    full = (1 << n) - 1
    out = np.zeros((len(sources), h), dtype=np.float32)
    for i, src in enumerate(sources):
        frontier = mask[src] & ~(1 << src) & full
        visited = frontier
        out[i, 0] = frontier.bit_count()
        for j in range(1, h):
            acc = 0
            f = frontier
            while f:
                low = f & -f
                acc |= mask[low.bit_length() - 1]
                f ^= low
            frontier = acc & ~visited & full
            out[i, j] = frontier.bit_count()
            if not frontier:
                break
            visited |= frontier
    return out


def build_features(g, ctx, pops, k, h, s, rt):
    ids, adj, mask = _collect_subgraph(g, ctx, pops, s)
    n = len(ids)
    cl = _clustering(adj)
    sources = list(range(min(k, n)))
    src_cl = np.zeros(k, dtype=np.float32)
    src_cl[:len(sources)] = cl[:len(sources)]
    reach = np.zeros(h, dtype=np.float32)
    if sources:
        reach = _reach(mask, sources, h, n).mean(axis=0)

    d = ctx.knn_dists()
    if d.size:
        basic = [ctx.ndc, ctx.step, ctx.inserts, ctx.first_nn,
                 float(d[0]), float(d[-1]), float(d.mean()), float(d.var()),
                 float(np.median(d)), float(np.percentile(d, 25)),
                 float(np.percentile(d, 75))]
    else:
        basic = [ctx.ndc, ctx.step, ctx.inserts, ctx.first_nn] + [ctx.pop_dist] * 7
    return np.concatenate([
        np.asarray([cl.mean()], dtype=np.float32),   # avg clustering coeff
        src_cl,                                      # per-BSF-node density
        reach,                                       # h-hop reachability
        np.asarray(basic, dtype=np.float32),         # runtime statistics
        np.asarray([rt], dtype=np.float32),          # recall target
    ])


class _Homing:
    def __init__(self, w, kp):
        self.window = deque(maxlen=int(w))
        self.kp = int(kp)
        self.done = False
        self.ndc = None

    def update(self, ctx):
        self.window.append(ctx.pop_dist)
        if not self.done and len(self.window) == self.window.maxlen:
            d = ctx.dists()
            if d.size >= self.kp and min(self.window) > float(d[self.kp - 1]):
                self.done = True
                self.ndc = ctx.ndc
        return self.done


# --------------------------------------------------------------------------
# training set: label C(q) + features at the phase transition
# --------------------------------------------------------------------------
def collect(g, queries, gt, k, rt, w, kp, h, s, max_ndc,
            need_features=True):
    need = int(np.ceil(rt * k - 1e-9))
    X, y, ratio, unreachable = [], [], [], 0
    for q, gts in zip(queries, gt):
        hom = _Homing(w, kp)
        pops = []
        st = {"feat": None, "t": None}
        def stop(c, hom=hom, pops=pops, st=st):
            if not hom.done:
                pops.append((c.pop_dist, c.pop_id))
                hom.update(c)
                if hom.done:
                    st["t"] = hom.ndc
                    if need_features:
                        st["feat"] = build_features(g, c, pops, k, h, s, rt)
            return c.hit >= need
        o = run_query(g, q, k=k, ef=None, gt_set=set(gts[:k].tolist()),
                      max_ndc=max_ndc, stop=stop)
        if o["recall"] is None or o["recall"] < rt - 1e-6:
            unreachable += 1
            continue
        if st["t"] is None or (need_features and st["feat"] is None):
            continue
        ratio.append(st["t"] / max(o["ndc"], 1.0))
        if need_features:
            X.append(st["feat"])
            y.append(o["ndc"])
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64),
            np.asarray(ratio, dtype=np.float32), unreachable)


def calibrate(g, queries, gt, k, rt, cands, h, s, max_ndc, t_bar=0.2):
    """Pick (w, k') whose mean homing fraction is closest to t_bar (§4.3)."""
    best = None
    for (w, kp) in cands:
        X, y, ratio, _ = collect(g, queries, gt, k, rt, w, kp, h, s, max_ndc,
                                 need_features=False)
        if ratio.size == 0:
            continue
        score = abs(float(ratio.mean()) - t_bar)
        print(f"    homing (w={w:>3}, k'={kp:>3}): mean t={ratio.mean():.3f} "
              f"n={ratio.size} |t-{t_bar}|={score:.3f}")
        if best is None or score < best[0]:
            best = (score, w, kp, float(ratio.mean()))
    if best is None:
        return 10, 10, float("nan")
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------
# model + online search
# --------------------------------------------------------------------------
class ANNiE:
    def __init__(self, model, rt, w, kp, h, s):
        self.model = model
        self.rt = float(rt)
        self.w = int(w)
        self.kp = int(kp)
        self.h = int(h)
        self.s = int(s)

    @classmethod
    def train(cls, X, y, rt, p, w, kp, h, s, seed=0):
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=float(p), n_estimators=400,
            learning_rate=0.05, num_leaves=6,
            min_child_samples=min(1000, max(20, len(X) // 50)),
            random_state=seed, n_jobs=4, verbose=-1)
        model.fit(X, y)
        return cls(model, rt, w, kp, h, s)


def run_annie_s(g, queries, gt, k, method, max_ndc):
    """ANNiE-S (Alg. 5): no ef, no qualification check, stop at Chat."""
    n = len(queries)
    rec = np.empty(n, dtype=np.float32)
    ndc = np.empty(n, dtype=np.float64)
    chat = np.full(n, np.nan, dtype=np.float64)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        hom = _Homing(method.w, method.kp)
        pops = []
        st = {"c": None, "called": False}
        def stop(c, hom=hom, pops=pops, st=st):
            if not hom.done:
                pops.append((c.pop_dist, c.pop_id))
            newly = hom.update(c)
            if newly and not st["called"]:
                st["called"] = True
                f = build_features(g, c, pops, k, method.h, method.s,
                                   method.rt)
                st["c"] = float(method.model.predict(f[None, :])[0])
            if st["c"] is not None and c.ndc >= st["c"]:
                return True
            return False
        o = run_query(g, q, k=k, ef=None, gt_set=set(gts[:k].tolist()),
                      max_ndc=max_ndc, stop=stop)
        rec[i] = o["recall"]
        ndc[i] = o["ndc"]
        chat[i] = st["c"] if st["c"] is not None else np.nan
    return {"recall": rec, "ndc": ndc, "chat": chat}


def estimate_all(g, queries, gt, k, method, max_ndc):
    """Offline cost estimation (for MAPE / Q-error / coverage diagnostics)."""
    pred = np.full(len(queries), np.nan, dtype=np.float64)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        hom = _Homing(method.w, method.kp)
        pops = []
        st = {"c": None, "called": False}
        def stop(c, hom=hom, pops=pops, st=st):
            if not hom.done:
                pops.append((c.pop_dist, c.pop_id))
            if hom.update(c) and not st["called"]:
                st["called"] = True
                f = build_features(g, c, pops, k, method.h, method.s,
                                   method.rt)
                st["c"] = float(method.model.predict(f[None, :])[0])
                return True
            return False
        run_query(g, q, k=k, ef=None, gt_set=set(gts[:k].tolist()),
                  max_ndc=max_ndc, stop=stop)
        pred[i] = st["c"] if st["c"] is not None else np.nan
    return pred
