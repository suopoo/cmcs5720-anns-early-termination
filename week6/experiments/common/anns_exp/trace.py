"""Search-trace collection shared by the three experiments.

A trace is the *complete* instrumented trajectory of one query for one search
mode.  Because every learned method only decides *when to stop* (it never
changes which node is expanded next), every model / loss / multiplier /
prediction-interval variant can be replayed exactly on the same trace.  This
lets us run dozens of controlled conditions without repeating the graph
search, while keeping the NDC / recall numbers identical to an online run.

Storage convention: `step_ptr[i]:step_ptr[i+1]` slices the per-query step
arrays.  Feature suffixes (ANNiE basic+graph, LAET tail) are stored per query
or per checkpoint; the raw query vector is stored once and re-attached at
model-training time.
"""
from __future__ import annotations

import os
import time

import h5py
import numpy as np

from .core.paths import GT_DIR, SPLIT_DIR, CACHE_DIR, dataset_path
from .core import paths
from .core.graph import build_graph
from .core.search import run_query
from .features import laet_tail, darth_vec, annie_basic, annie_graph
from .annie import _Homing

N_INDEX = 992_000
TRAIN_SLICE = (992_000, 999_000)
VAL_SLICE = (999_000, 999_500)
TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "shared", "traces")
os.makedirs(TRACE_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_split(which):
    with h5py.File(dataset_path("gist"), "r") as f:
        if which == "train":
            q = f["train"][TRAIN_SLICE[0]:TRAIN_SLICE[1]].astype(np.float32, copy=False)
        elif which == "val":
            q = f["train"][VAL_SLICE[0]:VAL_SLICE[1]].astype(np.float32, copy=False)
        elif which == "test":
            q = f["test"][:].astype(np.float32, copy=False)
        else:
            raise ValueError(which)
    gt = np.load(os.path.join(GT_DIR, f"gist_{which}_k100.npy"))
    return q, gt


def load_base():
    """Read-only memmap of the 992k x 960 index base (3.8 GB on disk).

    A plain h5py read creates a private 3.8 GB allocation and a second copy
    in the page file, which made every trace pass page-thrash.  The memmap is
    file-backed and evictable; the graph search only reads rows.
    """
    p = os.path.join(paths.CACHE_DIR, "gist_base992000.npy")
    if os.path.exists(p):
        # Load the .npy once into RAM.  A memmap is file-backed and random
        # distance batches then page-fault on every step (measured ~5x
        # slower); a plain load keeps the 3.8 GB resident and leaves enough
        # headroom for the 0.4 GB graph on a 16 GB machine.
        return np.load(p)
    with h5py.File(dataset_path("gist"), "r") as f:
        return f["train"][:N_INDEX].astype(np.float32, copy=False)


def load_graph(base):
    return build_graph(base, M=16, efc=200,
                       cache_name="gist_idx992000")


def suffix_at(g, ctx, pops, k, h, s, rt):
    """ANNiE basic (12) + graph (1+k+h) at the current state."""
    return (annie_basic(ctx, rt).astype(np.float32),
            annie_graph(g, ctx, pops, k, h, s).astype(np.float32))


def _trace_path(name):
    return os.path.join(TRACE_DIR, name + ".npz")


# --------------------------------------------------------------------------
# LAET / unlimited-beam traces (k=10 for exp1/3, k=100 for exp2)
# --------------------------------------------------------------------------
def collect_unlimited(g, queries, gt, k, fs, max_ndc, keep_steps,
                      need_suffix_f, h=4, s=200, rt=0.9,
                      stop_at_watch_after=None, watch_top1=False,
                      rt_grid=(0.99,), homing=None, name="",
                      stop_at_full_recall=False, stop_at_recall=None,
                      max_steps=20000, stop_floor=0):
    """Unlimited-beam trace.

    Records: every step (ndc, recall@k); LAET tail features at each f in
    `fs`; optionally ANNiE basic+graph suffix at one f; optionally homing
    transition state and cost-to-recall for each rt in `rt_grid`.
    """
    n = len(queries)
    ptr = np.zeros(n + 1, dtype=np.int64)
    sn, sr = [], []
    tails = {f: np.full((n, 5), np.nan, dtype=np.float32) for f in fs}
    suf = (np.full((n, 12 + 1 + k + h), np.nan, dtype=np.float32)
           if need_suffix_f is not None else None)
    hom_feat = (np.full((n, 12 + 1 + k + h), np.nan, dtype=np.float32)
                if homing is not None else None)
    hom_ndc = np.full(n, np.nan)
    costs = {r: np.full(n, np.nan) for r in rt_grid}
    c1 = np.full(n, np.nan)
    wat = np.full(n, np.nan)
    reasons = []
    t0 = time.time()
    for i, (q, gts) in enumerate(zip(queries, gt)):
        gid = {int(x) for x in gts[:k].tolist()}
        pops = []
        st = {"tails": {}, "suf": False, "hom": None,
              "cost": {r: np.nan for r in rt_grid}}
        hom = _Homing(homing[0], homing[1]) if homing is not None else None
        watch = {int(gts[0])} if watch_top1 else None

        def stop(c, st=st, pops=pops, hom=hom):
            pops.append((c.pop_dist, c.pop_id))
            for f in fs:
                if f not in st["tails"] and c.ndc >= f:
                    st["tails"][f] = laet_tail(c)
                    if need_suffix_f is not None and f == need_suffix_f:
                        st["suf"] = suffix_at(g, c, pops, k, h, s, rt)
            if hom is not None:
                newly = hom.update(c)
                if newly and st["hom"] is None:
                    st["hom"] = (c.ndc, suffix_at(g, c, pops, k, h, s, rt))
            for r in rt_grid:
                if np.isnan(st["cost"][r]) and c.hit >= r * k - 1e-9:
                    st["cost"][r] = c.ndc
            if (stop_at_watch_after is not None and c.watch_hit
                    and c.ndc >= stop_at_watch_after):
                return True
            reached_feature_point = c.ndc >= stop_floor
            if (stop_at_full_recall and c.hit >= k
                    and reached_feature_point):
                return True
            if (stop_at_recall is not None
                    and c.hit >= stop_at_recall * k - 1e-9
                    and reached_feature_point):
                return True
            return False

        o = run_query(g, q, k=k, ef=None, gt_set=gid, max_ndc=max_ndc,
                      stop=stop, trace_every=1 if keep_steps else 0,
                      watch=watch, max_steps=max_steps)
        for f in fs:
            if f in st["tails"]:
                tails[f][i] = st["tails"][f]
        if st["suf"]:
            suf[i] = np.concatenate(st["suf"])
        if st["hom"] is not None:
            hom_ndc[i] = st["hom"][0]
            hom_feat[i] = np.concatenate(st["hom"][1])
        for r in rt_grid:
            costs[r][i] = st["cost"][r]
        if watch_top1:
            c1[i] = o["watch_ndc"] if o["watch_ndc"] is not None else np.nan
        if keep_steps:
            for ndc, _, rec in o["trace"]:
                sn.append(ndc); sr.append(rec)
        ptr[i + 1] = len(sn)
        reasons.append(o["reason"])
        if (i + 1) % (50 if n <= 500 else 500) == 0:
            log(f"  {name} {i+1}/{n} steps={len(sn):,} "
                f"{time.time()-t0:.0f}s")
    out = {
        "ptr": ptr,
        "step_ndc": np.asarray(sn, dtype=np.int32),
        "step_recall": np.asarray(sr, dtype=np.float32),
        "c1": c1.astype(np.float32),
        "watch_to_target": wat,
        "reason": np.asarray(reasons, dtype="U16"),
    }
    for f in fs:
        out[f"tail_f{f}"] = tails[f]
    if suf is not None:
        out["suffix"] = suf
    if hom_feat is not None:
        out["hom_feat"] = hom_feat
        out["hom_ndc"] = hom_ndc
    for r in rt_grid:
        out[f"cost_rt{r}"] = costs[r]
    return out


# --------------------------------------------------------------------------
# DARTH bounded-beam traces
# --------------------------------------------------------------------------
def collect_darth(g, queries, gt, k, ef, rt_grid, max_ndc, keep_steps,
                  delta_obs=25, transfer_grid=(),
                  need_transfer_suffix=False, h=4, s=200, rt_feat=0.9,
                  name=""):
    """Bounded efSearch trace with DARTH's 11 features at every step.

    Also records (a) training observations every `delta_obs` NDC,
    (b) cost-to-recall for each rt, and optionally (c) ANNiE suffix + DARTH
    features on a fixed NDC grid (for the unified-checkpoint transfer groups).
    """
    n = len(queries)
    ptr = np.zeros(n + 1, dtype=np.int64)
    sn, sr, sf = [], [], []
    ptr_obs = np.zeros(n + 1, dtype=np.int64)
    on, of, oy = [], [], []
    ptr_tr = np.zeros(n + 1, dtype=np.int64)
    tn, tf, td, tr_rec = [], [], [], []
    costs = {r: np.full(n, np.nan) for r in rt_grid}
    reasons = []
    grid = np.asarray(sorted(set(transfer_grid)), dtype=np.int64)
    t0 = time.time()
    for i, (q, gts) in enumerate(zip(queries, gt)):
        target = {int(x) for x in gts[:k].tolist()}
        st = {"last_obs": -1e9, "cost": {r: np.nan for r in rt_grid},
              "gidx": 0}
        pops = []

        def stop(c, st=st, pops=pops):
            pops.append((c.pop_dist, c.pop_id))
            if keep_steps:
                sn.append(c.ndc); sr.append(c.hit / k); sf.append(darth_vec(c))
            for r in rt_grid:
                if np.isnan(st["cost"][r]) and c.hit >= r * k - 1e-9:
                    st["cost"][r] = c.ndc
            if c.ndc - st["last_obs"] >= delta_obs:
                st["last_obs"] = c.ndc
                on.append(c.ndc); oy.append(c.hit / k)
                of.append(darth_vec(c))
            while st["gidx"] < len(grid) and c.ndc >= grid[st["gidx"]]:
                st["gidx"] += 1
                tn.append(c.ndc)
                td.append(darth_vec(c))
                tr_rec.append(c.hit / k)
                if need_transfer_suffix:
                    tf.append(np.concatenate(suffix_at(g, c, pops, k, h, s,
                                                       rt_feat)))
            return False

        o = run_query(g, q, k=k, ef=ef, gt_set=target, max_ndc=max_ndc,
                      stop=stop, trace_every=0, max_steps=6000)
        for r in rt_grid:
            costs[r][i] = st["cost"][r]
        ptr[i + 1] = len(sn)
        ptr_obs[i + 1] = len(on)
        ptr_tr[i + 1] = len(tn)
        reasons.append(o["reason"])
        if (i + 1) % (50 if n <= 500 else 500) == 0:
            log(f"  {name} {i+1}/{n} steps={len(sn):,} obs={len(on):,} "
                f"tr={len(tn):,} {time.time()-t0:.0f}s")
    out = {
        "ptr": ptr,
        "step_ndc": np.asarray(sn, dtype=np.int32),
        "step_recall": np.asarray(sr, dtype=np.float32),
        "step_feat": (np.asarray(sf, dtype=np.float32) if sf else
                      np.zeros((0, 11), dtype=np.float32)),
        "obs_ptr": ptr_obs,
        "obs_ndc": np.asarray(on, dtype=np.int32),
        "obs_recall": np.asarray(oy, dtype=np.float32),
        "obs_feat": (np.asarray(of, dtype=np.float32) if of else
                     np.zeros((0, 11), dtype=np.float32)),
        "tr_ptr": ptr_tr,
        "tr_ndc": np.asarray(tn, dtype=np.int32),
        "tr_darth": np.asarray(td, dtype=np.float32),
        "tr_recall": np.asarray(tr_rec, dtype=np.float32),
        "tr_suffix": (np.asarray(tf, dtype=np.float32) if need_transfer_suffix
                      else np.zeros((0, 0), dtype=np.float32)),
        "reason": np.asarray(reasons, dtype="U16"),
    }
    for r in rt_grid:
        out[f"cost_rt{r}"] = costs[r]
    return out
