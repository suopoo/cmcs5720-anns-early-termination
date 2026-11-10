"""Baselines for E1: fixed-ef (A) and Oracle (E).

Both are defined on the *same* instrumented search loop as the learned
methods, so NDC is directly comparable across all five groups.

Oracle semantics (DARTH §4.2.4 / ANNiE Def. 3): the smallest NDC at which
the running recall@k first reaches the target, observed with an unlimited
beam and with the ground truth available (god's eye -- not obtainable
online).
"""
from __future__ import annotations

import numpy as np

from core.search import run_query


def per_query_recall(ids, gt, k):
    return np.asarray([len(set(ids[i]) & set(gt[i][:k])) / k
                       for i in range(len(ids))], dtype=np.float32)


def run_fixed(g, queries, gt, k, ef):
    """Standard HNSW beam search with a fixed efSearch."""
    outs = [run_query(g, q, k=k, ef=ef) for q in queries]
    return {
        "recall": per_query_recall([o["ids"] for o in outs], gt, k),
        "ndc": np.asarray([o["ndc"] for o in outs], dtype=np.float64),
        "ids": [o["ids"] for o in outs],
        "reason": [o["reason"] for o in outs],
    }


def tune_fixed_ef(g, queries, gt, k, target, ef_min=8, ef_max=4096, bisect=6):
    """Smallest ef whose mean recall@k reaches `target` (A group).

    Doubling scan, then integer bisection.  Tune on a validation subset,
    evaluate on the disjoint test set.
    """
    lo, hi = None, None
    ef = ef_min
    while ef <= ef_max:
        r = run_fixed(g, queries, gt, k, ef)["recall"].mean()
        if r >= target:
            hi = ef
            break
        lo = ef
        ef *= 2
    if hi is None:                     # target unreachable even at ef_max
        return ef_max
    if lo is None:                     # reached at the very first ef
        return ef_min
    for _ in range(bisect):
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break
        r = run_fixed(g, queries, gt, k, mid)["recall"].mean()
        if r >= target:
            hi = mid
        else:
            lo = mid
    return hi


def tune_fixed_hitrate(g, queries, gt, k, target, hit_target,
                       ef_min=8, ef_max=16384, bisect=8):
    """Smallest ef such that `hit_target` of the queries reach `target`.

    This is the quality-matched fixed baseline: it answers the same question
    as ANNiE-S's RTG (p% of queries reach the target), so NDC is comparable.
    """
    def hit(ef):
        r = run_fixed(g, queries, gt, k, ef)["recall"]
        return float(np.mean(r >= target - 1e-6))

    lo, hi = None, None
    ef = ef_min
    while ef <= ef_max:
        if hit(ef) >= hit_target:
            hi = ef
            break
        lo = ef
        ef *= 2
    if hi is None:
        return ef_max
    if lo is None:
        return ef_min
    for _ in range(bisect):
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break
        if hit(mid) >= hit_target:
            hi = mid
        else:
            lo = mid
    return hi


def oracle(g, queries, gt, k, target, max_ndc, ef=None):
    """Per-query minimum NDC to reach recall@k >= target, along one trajectory.

    Returns (ndc, recall, reached).  `reached` is False when the search
    terminates naturally before the target -- such queries have no finite
    cost (ANNiE's C(q) = N).
    """
    need = int(np.ceil(target * k - 1e-9))
    n = len(queries)
    ndc = np.empty(n, dtype=np.float64)
    rec = np.empty(n, dtype=np.float32)
    for i, (q, gts) in enumerate(zip(queries, gt)):
        s = set(gts[:k].tolist())
        o = run_query(g, q, k=k, ef=ef, gt_set=s, max_ndc=max_ndc,
                      stop=lambda c: c.hit >= need)
        ndc[i] = o["ndc"]
        rec[i] = o["recall"]
    return ndc, rec, rec >= (target - 1e-6)


def oracle_min(g, queries, gt, k, target, max_ndc, ef_configs):
    """God's-eye lower bound over the search modes the methods actually use.

    The unlimited-beam trajectory is *not* cheaper than a bounded beam on
    easy queries (it admits every fresh neighbour into the queue), so a
    genuine per-query minimum must be taken over all trajectories:
    ef95, ef99 and unlimited.  Taking only one of them can make "Oracle"
    cost more than fixed-ef, which is nonsense.
    """
    n = len(queries)
    best = np.full(n, np.inf, dtype=np.float64)
    rec = np.zeros(n, dtype=np.float32)
    reached = np.zeros(n, dtype=bool)
    for ef in ef_configs:
        ndc, r, ok = oracle(g, queries, gt, k, target, max_ndc, ef=ef)
        # a run that *reaches* the target always beats one that does not,
        # whatever its NDC; among successful runs keep the cheapest.
        take = ok & (~reached | (ndc < best))
        best[take] = ndc[take]
        rec[take] = r[take]
        reached[take] = True
        # queries that never reach the target: keep the cheapest failed run
        miss = ~ok & ~reached & (ndc < best)
        best[miss] = ndc[miss]
        rec[miss] = r[miss]
    return best, rec, reached


def static_budget(g, queries, gt, k, budget):
    """DARTH §3.2.2 baseline: everybody searches exactly `budget` NDC."""
    outs = [run_query(g, q, k=k, ef=None, max_ndc=budget) for q in queries]
    return {
        "recall": per_query_recall([o["ids"] for o in outs], gt, k),
        "ndc": np.asarray([o["ndc"] for o in outs], dtype=np.float64),
        "ids": [o["ids"] for o in outs],
    }
