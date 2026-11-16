"""Re-collect the LAET validation/test traces with a stop_floor=1260.

The first full pass stopped as soon as recall@10 reached 1.0, which for the
easiest queries happened before NDC 1,260.  LAET's budget floor is 1,260, so
the state at f must exist even for those queries.  Training traces are
unaffected (they stop after NDC 2,048).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anns_exp import trace as T

FS = [256, 512, 630, 945, 1260, 1575, 2048]


def main():
    base = T.load_base()
    g = T.load_graph(base)
    for split in ("val", "test"):
        q, gt = T.load_split(split)
        name = f"gist_k10_laet_{split}"
        out = T.collect_unlimited(
            g, q, gt, k=10, fs=FS, max_ndc=40000, keep_steps=True,
            need_suffix_f=1260, watch_top1=True, stop_at_watch_after=None,
            rt_grid=(0.90, 0.95, 0.99), stop_at_full_recall=True,
            stop_floor=1260, max_steps=6000, name=name)
        p = os.path.join(T.TRACE_DIR, name + ".npz")
        np.savez_compressed(p + ".tmp.npz", **out)
        os.replace(p + ".tmp.npz", p)
        print("saved", p, os.path.getsize(p) / 1e6, "MB", flush=True)


if __name__ == "__main__":
    main()
