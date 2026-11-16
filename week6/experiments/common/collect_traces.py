"""Collect and cache all search traces needed by Experiments 1-3.

Run once (each stage is resumable):
  python collect_traces.py --stage exp13
  python collect_traces.py --stage exp2

exp13 : k=10, rt=0.90 traces for LAET (unlimited) and DARTH (bounded ef99)
exp2  : k=100, rt=0.90/0.95/0.99 traces for LAET+ANNiE (unlimited) and DARTH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core import paths
from anns_exp import trace as T
from anns_exp import baselines
from anns_exp import annie

TRACE_DIR = os.path.join(paths.EXP_ROOT, "shared", "traces")
os.makedirs(TRACE_DIR, exist_ok=True)
META = os.path.join(TRACE_DIR, "meta.json")


def load_meta():
    if os.path.exists(META):
        with open(META) as f:
            return json.load(f)
    return {}


def save_meta(meta):
    with open(META, "w") as f:
        json.dump(meta, f, indent=2)


def path(name):
    return os.path.join(TRACE_DIR, name + ".npz")


def savez_atomic(p, out):
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


def tune_ef(g, q, gt, k, target, name, meta, sub=200, ef_max=2048):
    if name in meta:
        return meta[name]
    t0 = time.time()
    ef = baselines.tune_fixed_ef(g, q[:sub], gt[:sub], k, target,
                                 ef_min=32, ef_max=ef_max, bisect=6,
                                 max_ndc=50000)
    # verify on the whole validation split
    rr = baselines.run_fixed(g, q, gt, k, ef, max_ndc=50000)["recall"].mean()
    meta[name] = {"ef": int(ef), "val_mean_recall": float(rr),
                  "tuned_on": int(min(sub, len(q))), "seconds": time.time() - t0}
    save_meta(meta)
    T.log(f"tuned {name}: ef={ef} val_recall={rr:.4f} "
          f"({time.time()-t0:.0f}s)")
    return meta[name]


def stage_exp13(g, meta, tag="", ntr=5000, nva=500, nte=1000):
    qtr, gtr = T.load_split("train")
    qva, gva = T.load_split("val")
    qte, gte = T.load_split("test")
    qtr, gtr = qtr[:ntr], gtr[:ntr]
    qva, gva = qva[:nva], gva[:nva]
    qte, gte = qte[:nte], gte[:nte]

    ef10 = tune_ef(g, qva, gva, 10, 0.99, "ef99_k10" + tag, meta,
                   sub=min(200, nva))["ef"]

    fs = [256, 512, 630, 945, 1260, 1575, 2048]
    jobs = [("train", qtr, gtr, False, 40000, 2048),
            ("val", qva, gva, True, 40000, None),
            ("test", qte, gte, True, 40000, None)]
    for split, q, gt, keep, max_ndc, after in jobs:
        name = f"gist_k10_laet_{split}{tag}"
        if os.path.exists(path(name)):
            T.log(f"skip {name}")
            continue
        T.log(f"collect {name}")
        out = T.collect_unlimited(
            g, q, gt, k=10, fs=fs, max_ndc=max_ndc, keep_steps=keep,
            need_suffix_f=1260, watch_top1=True, stop_at_watch_after=after,
            rt_grid=(0.90, 0.95, 0.99), stop_at_full_recall=keep,
            stop_floor=1260,
            max_steps=20000 if not keep else 6000, name=name)
        savez_atomic(path(name), out)
        T.log(f"saved {name} ({os.path.getsize(path(name))/1e6:.1f}MB)")

    grid = [256, 512, 2048] + [int(x) for x in
                np.linspace(1260, 12000, 24)]
    jobs = [("train", qtr, gtr, False), ("val", qva, gva, True),
            ("test", qte, gte, True)]
    for split, q, gt, keep in jobs:
        name = f"gist_k10_darth_{split}{tag}"
        if os.path.exists(path(name)):
            T.log(f"skip {name}")
            continue
        T.log(f"collect {name}")
        out = T.collect_darth(
            g, q, gt, k=10, ef=ef10, rt_grid=[0.90], max_ndc=40000,
            keep_steps=keep, delta_obs=25, transfer_grid=grid,
            need_transfer_suffix=True, h=4, s=200, rt_feat=0.9, name=name)
        savez_atomic(path(name), out)
        T.log(f"saved {name} ({os.path.getsize(path(name))/1e6:.1f}MB)")


def stage_exp2(g, meta, tag="", ntr=5000, nva=500, nte=1000):
    qtr, gtr = T.load_split("train")
    qva, gva = T.load_split("val")
    qte, gte = T.load_split("test")
    qtr, gtr = qtr[:ntr], gtr[:ntr]
    qva, gva = qva[:nva], gva[:nva]
    qte, gte = qte[:nte], gte[:nte]

    ef100 = tune_ef(g, qva, gva, 100, 0.99, "ef99_k100" + tag, meta,
                    sub=min(100, nva), ef_max=2048)["ef"]

    # homing calibration on 60 training queries (paper targets t_bar=0.2)
    hkey = "homing" + tag
    if hkey not in meta:
        t0 = time.time()
        w, kp, tbar = annie.calibrate(
            g, qtr[:min(60, ntr)], gtr[:min(60, ntr)], 100, 0.95,
            cands=[(5, 5), (10, 10), (20, 10), (20, 20)],
            h=4, s=200, max_ndc=80000)
        meta[hkey] = {"w": int(w), "kp": int(kp),
                      "mean_homing_fraction": float(tbar),
                      "seconds": time.time() - t0}
        save_meta(meta)
    w, kp = meta[hkey]["w"], meta[hkey]["kp"]
    T.log(f"homing w={w} k'={kp}")

    rts = (0.90, 0.95, 0.99)
    jobs = [("train", qtr, gtr, False), ("val", qva, gva, True),
            ("test", qte, gte, True)]
    for split, q, gt, keep in jobs:
        name = f"gist_k100_unlim_{split}{tag}"
        if os.path.exists(path(name)):
            T.log(f"skip {name}")
            continue
        T.log(f"collect {name}")
        out = T.collect_unlimited(
            g, q, gt, k=100, fs=[1260], max_ndc=50000, keep_steps=keep,
            need_suffix_f=None, watch_top1=True, stop_at_watch_after=None,
            rt_grid=rts, homing=(w, kp), rt=0.95,
            stop_at_recall=0.99, stop_floor=1260,
            max_steps=10000 if not keep else 8000, name=name)
        savez_atomic(path(name), out)
        T.log(f"saved {name} ({os.path.getsize(path(name))/1e6:.1f}MB)")

    for split, q, gt, keep in jobs:
        name = f"gist_k100_darth_{split}{tag}"
        if os.path.exists(path(name)):
            T.log(f"skip {name}")
            continue
        T.log(f"collect {name}")
        out = T.collect_darth(
            g, q, gt, k=100, ef=ef100, rt_grid=list(rts), max_ndc=80000,
            keep_steps=keep, delta_obs=25, transfer_grid=(),
            need_transfer_suffix=False, name=name)
        savez_atomic(path(name), out)
        T.log(f"saved {name} ({os.path.getsize(path(name))/1e6:.1f}MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["exp13", "exp2"], required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    a = ap.parse_args()
    T.log("load base")
    base = T.load_base()
    T.log(f"base {base.shape}")
    g = T.load_graph(base)
    T.log(f"graph degree0={g.degree0} max_level={g.max_level()}")
    meta = load_meta()
    if a.stage == "exp13":
        stage_exp13(g, meta, a.tag, a.n_train, a.n_val, a.n_test)
    else:
        stage_exp2(g, meta, a.tag, a.n_train, a.n_val, a.n_test)
    T.log("done")
