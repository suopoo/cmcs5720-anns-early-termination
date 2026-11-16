"""Tiny end-to-end pilot on a 5k GIST graph (debug only, no results)."""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anns_exp.core.paths import dataset_path, GT_DIR
from anns_exp.core.gt import exact_gt_chunked
from anns_exp.core.graph import build_graph
from anns_exp import trace as T

NB = 5000
GT_PATH = os.path.join(GT_DIR, "pilot5k_gt.npy")
QS = {"train": (4100, 4150), "val": (4150, 4170), "test": (4170, 4210)}


def main():
    with h5py.File(dataset_path("gist"), "r") as f:
        base = f["train"][:NB].astype(np.float32)
        q = f["train"][4100:4210].astype(np.float32)
    g = build_graph(base, M=16, efc=100, cache_name="smoke5k")
    if os.path.exists(GT_PATH):
        gt_all = np.load(GT_PATH)
    else:
        gt_all = exact_gt_chunked(base, q, k=100, qblock=16, verbose=False)
        np.save(GT_PATH, gt_all)
    qs = {k: q[a - 4100:b - 4100] for k, (a, b) in QS.items()}
    gts = {k: gt_all[a - 4100:b - 4100] for k, (a, b) in QS.items()}
    for split in ("train", "val", "test"):
        name = f"gist_k10_laet_{split}_smoke"
        if not os.path.exists(os.path.join(T.TRACE_DIR, name + ".npz")):
            out = T.collect_unlimited(
                g, qs[split], gts[split], k=10,
                fs=[256, 512, 630, 945, 1260, 1575, 2048], max_ndc=8000,
                keep_steps=(split != "train"), need_suffix_f=1260,
                watch_top1=True,
                stop_at_watch_after=1575 if split == "train" else None,
                rt_grid=(0.9, 0.95, 0.99), name=name)
            _p = os.path.join(T.TRACE_DIR, name + ".npz")
            np.savez_compressed(_p + ".tmp.npz", **out); os.replace(_p + ".tmp.npz", _p)
        name = f"gist_k10_darth_{split}_smoke"
        if not os.path.exists(os.path.join(T.TRACE_DIR, name + ".npz")):
            out = T.collect_darth(
                g, qs[split], gts[split], k=10, ef=128, rt_grid=[0.9],
                max_ndc=8000, keep_steps=(split != "train"), delta_obs=10,
                transfer_grid=[256, 512, 1260, 1460, 2048],
                need_transfer_suffix=True, name=name)
            _p = os.path.join(T.TRACE_DIR, name + ".npz")
            np.savez_compressed(_p + ".tmp.npz", **out); os.replace(_p + ".tmp.npz", _p)
        name = f"gist_k100_unlim_{split}_smoke"
        if not os.path.exists(os.path.join(T.TRACE_DIR, name + ".npz")):
            out = T.collect_unlimited(
                g, qs[split], gts[split], k=100, fs=[1260], max_ndc=8000,
                keep_steps=(split != "train"), need_suffix_f=None,
                watch_top1=True, rt_grid=(0.9, 0.95, 0.99),
                homing=(5, 5), rt=0.95, name=name)
            _p = os.path.join(T.TRACE_DIR, name + ".npz")
            np.savez_compressed(_p + ".tmp.npz", **out); os.replace(_p + ".tmp.npz", _p)
        name = f"gist_k100_darth_{split}_smoke"
        if not os.path.exists(os.path.join(T.TRACE_DIR, name + ".npz")):
            out = T.collect_darth(
                g, qs[split], gts[split], k=100, ef=128,
                rt_grid=[0.9, 0.95, 0.99], max_ndc=8000,
                keep_steps=(split != "train"), delta_obs=10,
                transfer_grid=(), need_transfer_suffix=False, name=name)
            _p = os.path.join(T.TRACE_DIR, name + ".npz")
            np.savez_compressed(_p + ".tmp.npz", **out); os.replace(_p + ".tmp.npz", _p)
    print("pilot traces ready")


if __name__ == "__main__":
    main()
