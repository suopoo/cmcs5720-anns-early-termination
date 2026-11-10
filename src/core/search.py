"""Instrumented HNSW base-layer search.

Every distance computation is counted (NDC) and any stop rule can be plugged
in.  Validated against faiss: identical recall at every efSearch.

Three search modes, matching the three papers:
  * `ef=int`      standard HNSW beam search (plain / DARTH): candidates are
                  admitted only if closer than the worst kept result, and the
                  search ends naturally when the best candidate is worse.
  * `ef=None`     "unlimited beam" (LAET / ANNiE-S): every freshly visited
                  node enters the candidate queue; only a budget / stop rule
                  can end the search.
"""
from __future__ import annotations

import heapq
import numpy as np


class Ctx:
    """Snapshot handed to stop rules / probes.  Expensive fields are lazy."""

    __slots__ = ("ndc", "step", "inserts", "res", "first_nn", "closest",
                 "furthest", "q", "qn", "_d", "topk", "pop_dist", "pop_id",
                 "hit", "k", "_kd")

    def __init__(self, ndc, step, inserts, res, first_nn, q, qn,
                 topk=None, pop_dist=None, pop_id=None, hit=None, k=None):
        self.ndc = ndc
        self.step = step
        self.inserts = inserts
        self.res = res                      # max-heap of (-dist, id)
        self.first_nn = first_nn
        self.closest = -res[0][0] if res else first_nn
        self.furthest = self.closest        # top of the max-heap
        self.q = q
        self.qn = qn
        self._d = None
        self.topk = topk if topk is not None else []   # max-heap, cap k
        self.pop_dist = pop_dist            # distance of the node just popped
        self.pop_id = pop_id                # its id
        self.hit = hit                      # |results so far ∩ gt|
        self.k = k
        self._kd = None

    def dists(self):
        """Distances of the current result set, ascending (float32)."""
        if self._d is None:
            self._d = -np.fromiter((d for d, _ in self.res), dtype=np.float32,
                                   count=len(self.res))
            self._d.sort()
        return self._d

    def knn_dists(self):
        """Distances of the best-k set, ascending (DARTH's NN features)."""
        if self._kd is None:
            if self.topk:
                self._kd = np.sort(-np.fromiter((d for d, _ in self.topk),
                                                dtype=np.float32,
                                                count=len(self.topk)))
            else:
                self._kd = np.empty(0, dtype=np.float32)
        return self._kd


def run_query(g, q, k=10, ef=None, stop=None, probe=None,
              gt_set=None, trace_every=0, max_ndc=None):
    """Run one query and return ids / ndc / probes / trace.

    stop(ctx)  -> bool  called after every expansion batch; True terminates.
    probe(ctx) -> any   called at the same points; non-None results are kept.
    gt_set     -> set   ground-truth ids, used to track recall@k online.
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    qn = float(q @ q)
    unbounded = ef is None
    cap = k if unbounded else ef
    ndc = 0
    reason = "natural"

    # ---------- upper layers: greedy, beam width 1 ----------
    cur = g.entry
    cur_d = float(g.dist_batch(q, qn, np.array([cur], dtype=np.int32))[0])
    ndc += 1
    for layer in range(int(g.levels[cur]), 0, -1):
        changed = True
        while changed:
            changed = False
            nb = g.neighbors_of(cur, layer)
            nb = nb[nb >= 0]
            if nb.size == 0:
                continue
            d = g.dist_batch(q, qn, nb)
            ndc += int(nb.size)
            j = int(np.argmin(d))
            if d[j] < cur_d:
                cur, cur_d, changed = int(nb[j]), float(d[j]), True
    first_nn = cur_d

    # ---------- base layer ----------
    stamp = g.stamp_buffer()               # reused int32 buffer, O(1) clear
    sv = g.next_stamp()
    stamp[cur] = sv

    cand = [(cur_d, cur)]
    res = [(-cur_d, cur)]                  # max-heap, capacity cap
    topk = [(-cur_d, cur)]                 # max-heap, capacity k
    hit = 1 if (gt_set is not None and cur in gt_set) else 0
    inserts = 1

    step = 0
    probes, trace = [], []

    while cand:
        cd, c = heapq.heappop(cand)
        if not unbounded and len(res) >= ef and -res[0][0] < cd:
            break

        nb = g.neighbors_of(c, 0)
        nb = nb[nb >= 0]
        if nb.size:
            fresh = nb[stamp[nb] != sv]
            if fresh.size:
                stamp[fresh] = sv
                d = g.dist_batch(q, qn, fresh)
                ndc += int(fresh.size)
                for j in np.argsort(d):
                    nj, dj = int(fresh[j]), float(d[j])
                    admit = unbounded or len(res) < cap or dj < -res[0][0]
                    if admit:
                        heapq.heappush(cand, (dj, nj))
                        heapq.heappush(res, (-dj, nj))
                        inserts += 1
                        if len(res) > cap:
                            heapq.heappop(res)
                        if len(topk) < k:
                            heapq.heappush(topk, (-dj, nj))
                            if gt_set is not None and nj in gt_set:
                                hit += 1
                        elif dj < -topk[0][0]:
                            old = heapq.heapreplace(topk, (-dj, nj))
                            if gt_set is not None:
                                hit += (nj in gt_set) - (old[1] in gt_set)
                    if max_ndc is not None and ndc >= max_ndc:
                        break
                if max_ndc is not None and ndc >= max_ndc:
                    reason = "budget"
                    break
        step += 1

        if trace_every and step % trace_every == 0 and gt_set is not None:
            trace.append((ndc, step, min(hit, k) / k))

        if stop is not None or probe is not None:
            ctx = Ctx(ndc, step, inserts, res, first_nn, q, qn,
                      topk=topk, pop_dist=cd, pop_id=c,
                      hit=min(hit, k), k=k)
            if probe is not None:
                r = probe(ctx)
                if r is not None:
                    probes.append(r)
            if stop is not None and stop(ctx):
                reason = "early"
                break
        if max_ndc is not None and ndc >= max_ndc:
            reason = "budget"
            break

    ids = [x[1] for x in heapq.nlargest(k, topk, key=lambda t: t[0])]
    return {
        "ids": ids, "ndc": ndc, "inserts": inserts,
        "recall": (min(hit, k) / k) if gt_set is not None else None,
        "probes": probes, "trace": trace, "reason": reason,
        "first_nn": first_nn,
    }
