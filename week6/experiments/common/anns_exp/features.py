"""Feature sets for LAET, DARTH and ANNiE, and the transfer compositions.

All definitions are taken from the papers / the audited reference code; see
exp01_feature_transfer/reports/source_audit.md for the line-by-line mapping.

LAET (SIGMOD'20 Table 5, HNSW)
  q(960) | d_start | d_1st | d_10th | d_1st/d_start | d_10th/d_start

DARTH (SIGMOD'25 Table 1)
  step | ndc | inserts | d_first | d_closest | d_furthest |
  avg | var | median | p25 | p75

ANNiE (VLDB'26 §4.2 + §4.5)
  basic: q | ndc | step | inserts | d_first | d_closest | d_furthest |
         avg | var | median | p25 | p75 | recall-target
  graph: avg clustering coefficient | per-BSF clustering (k) |
         h-hop reachability (h)
"""
from __future__ import annotations

import numpy as np

from .annie import _collect_subgraph, _clustering, _reach

LAET_TAIL = ["d_start", "d_1st", "d_10th", "d_1st/d_start", "d_10th/d_start"]
DARTH_NAMES = ["step", "ndc", "inserts", "d_first", "d_closest", "d_furthest",
               "avg", "var", "median", "p25", "p75"]
ANNIE_BASIC_NAMES = ["ndc", "step", "inserts", "d_first", "d_closest",
                     "d_furthest", "avg", "var", "median", "p25", "p75",
                     "recall_target"]
# DARTH features that ANNiE's basic block adds beyond DARTH's 11
ANNIE_ADDED_TO_DARTH = ["ndc", "step", "inserts", "d_furthest", "avg",
                        "var", "median", "p25", "p75", "recall_target"]
# ANNiE basic entries that duplicate LAET's tail (first_nn/d_closest)
ANNIE_ADDED_TO_LAET = ["ndc", "step", "inserts", "d_furthest", "avg", "var",
                       "median", "p25", "p75", "recall_target"]


def _tails(ctx):
    d = ctx.dists()
    if d.size == 0:
        d1 = d10 = ctx.pop_dist
    else:
        d1 = float(d[0])
        d10 = float(d[min(9, d.size - 1)])
    dstart = float(ctx.first_nn)
    return dstart, d1, d10


def laet_tail(ctx):
    dstart, d1, d10 = _tails(ctx)
    return [dstart, d1, d10,
            d1 / max(dstart, 1e-9), d10 / max(dstart, 1e-9)]


def darth_vec(ctx):
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


def annie_basic(ctx, rt):
    d = ctx.knn_dists()
    if d.size:
        first = ctx.first_nn
        vals = [float(d[0]), float(d[-1]), float(d.mean()), float(d.var()),
                float(np.median(d)), float(np.percentile(d, 25)),
                float(np.percentile(d, 75))]
    else:
        first = ctx.pop_dist
        vals = [ctx.pop_dist] * 7
    return np.asarray([ctx.ndc, ctx.step, ctx.inserts, first, *vals,
                       float(rt)], dtype=np.float32)


def _collect_subgraph_fast(g, ctx, pops, s):
    """s closest *accessed* nodes; pops first, result heap only if short.

    The result heap of a bounded-ef search can hold ef (up to ~2000) entries,
    so iterating it at every checkpoint would dominate the trace cost.  The
    popped nodes already contain the search frontier; when there are at least
    s of them we do not need the unpopped result set.  This is the same
    Def. 7 approximation used by the audited ANNiE implementation.
    """
    dm = {}
    for d, i in pops:
        if i not in dm or d < dm[i]:
            dm[i] = d
    if len(dm) < s:
        for dn, i in ctx.res:
            d = -dn
            if i not in dm or d < dm[i]:
                dm[i] = d
    order = sorted(dm.items(), key=lambda kv: kv[1])
    ids = [i for i, _ in order[:s]]
    seen = set(ids)
    if len(ids) < s:
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


def annie_graph(g, ctx, pops, k, h, s):
    """§4.5 graph features (avg CC, per-BSF CC, h-hop reachability)."""
    ids, adj, mask = _collect_subgraph_fast(g, ctx, pops, s)
    n = len(ids)
    cl = _clustering(adj)
    ns = min(k, n)
    src_cl = np.zeros(k, dtype=np.float32)
    src_cl[:ns] = cl[:ns]
    reach = np.zeros(h, dtype=np.float32)
    if ns:
        reach = _reach(mask, list(range(ns)), h, n).mean(axis=0)
    return np.concatenate([np.asarray([cl.mean()], dtype=np.float32),
                           src_cl, reach])


def annie_complete(g, ctx, pops, k, h, s, rt):
    return np.concatenate([np.asarray(ctx.q, dtype=np.float32),
                           annie_basic(ctx, rt),
                           annie_graph(g, ctx, pops, k, h, s)]).astype(np.float32)


def build(group, g, ctx, pops, rt, k, h, s):
    """Build one of the six transfer groups or a complete ANNiE vector."""
    q = np.asarray(ctx.q, dtype=np.float32)
    if group == "L0":
        return np.concatenate([q, laet_tail(ctx)]).astype(np.float32)
    if group == "L1":
        tail = laet_tail(ctx)
        b = annie_basic(ctx, rt)
        # added block indices: ndc, step, inserts, d_furthest, avg, var,
        # median, p25, p75, rt  -> [0,1,2,5,6,7,8,9,10,11]
        added = b[[0, 1, 2, 5, 6, 7, 8, 9, 10, 11]]
        graph = annie_graph(g, ctx, pops, k, h, s)
        return np.concatenate([q, tail, added, graph]).astype(np.float32)
    if group == "L2":
        return annie_complete(g, ctx, pops, k, h, s, rt)
    if group == "D0":
        return darth_vec(ctx)
    if group in ("D1", "D2"):
        # D1 = DARTH ∪ ANNiE, D2 = ANNiE only.  DARTH's 11 features are a
        # strict subset of ANNiE basic, so after de-duplication D1 == D2.
        return annie_complete(g, ctx, pops, k, h, s, rt)
    raise ValueError(group)


def dim(group, k, h):
    return {"L0": 960 + 5,
            "L1": 960 + 5 + 10 + (1 + k + h),
            "L2": 960 + 12 + (1 + k + h),
            "D0": 11,
            "D1": 960 + 12 + (1 + k + h),
            "D2": 960 + 12 + (1 + k + h)}[group]


def registry_rows(query_dim=960, k=10, h=4):
    rows = []
    def add(name, paper, cat, definition, dim, stage, state, cost, loc):
        rows.append(dict(feature_name=name, source_paper=paper,
                         category=cat, definition=definition, dimension=dim,
                         computation_stage=stage, required_state=state,
                         extraction_cost=cost, implementation_location=loc))
    add("F0_query", "LAET", "basic", "raw query vector", query_dim,
        "at checkpoint", "query", "O(1)", "features.laet_tail")
    for nm, defn in [("d_start", "distance of entry/first NN"),
                     ("d_1st", "closest result distance"),
                     ("d_10th", "10th closest result distance")]:
        add(nm, "LAET", "distance", defn, 1, "at checkpoint", "result heap",
            "O(k)", "features.laet_tail")
    add("d_1st/d_start", "LAET", "distance", "ratio", 1, "at checkpoint",
        "result heap", "O(1)", "features.laet_tail")
    add("d_10th/d_start", "LAET", "distance", "ratio", 1, "at checkpoint",
        "result heap", "O(1)", "features.laet_tail")
    for nm in DARTH_NAMES:
        add(nm, "DARTH", "runtime", nm, 1, "at prediction", "search state",
            "O(k)", "features.darth_vec")
    for nm in ANNIE_BASIC_NAMES:
        add(nm, "ANNiE", "runtime", nm, 1, "at homing transition",
            "search state", "O(k)", "features.annie_basic")
    add("avg_clustering", "ANNiE", "graph", "mean local clustering coefficient",
        1, "at checkpoint", "local subgraph", "O(s*d)", "features.annie_graph")
    add("bsf_clustering", "ANNiE", "graph",
        "per-BSF-node clustering coefficient", k, "at checkpoint",
        "local subgraph", "O(s*d)", "features.annie_graph")
    add("h_hop_reachability", "ANNiE", "graph",
        "newly reachable nodes per hop", h, "at checkpoint",
        "local subgraph", "O(h*s^2/64)", "features.annie_graph")
    return rows


ANNI_BASIC_NAMES = ANNIE_BASIC_NAMES


def assemble(group, q, tail=None, suffix=None, k=10, h=4, rt=0.9):
    """Re-attach the (stored-once) query vector to a stored feature suffix.

    `q` is (n, 960); `tail` is (n, 5) for LAET groups; `suffix` is
    (n, 12 + 1 + k + h) = ANNiE basic + graph.
    """
    q = np.asarray(q, dtype=np.float32)
    if group == "L0":
        return np.concatenate([q, np.asarray(tail, np.float32)], axis=1)
    basic = np.asarray(suffix[:, :12], np.float32)
    graph = np.asarray(suffix[:, 12:], np.float32)
    if group == "L1":
        added = basic[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]]
        return np.concatenate([q, np.asarray(tail, np.float32), added, graph],
                              axis=1)
    if group == "L2":
        return np.concatenate([q, basic, graph], axis=1)
    if group == "D0":
        return basic[:, :11]
    if group in ("D1", "D2"):
        b = basic.copy()
        b[:, 11] = rt
        return np.concatenate([q, b, graph], axis=1)
    raise ValueError(group)
