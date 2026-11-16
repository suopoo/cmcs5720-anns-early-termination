"""Prepare the shared GIST1M evaluation environment.

Splits (all on D:, nothing in C:):
  index base   : gist/train[0 : 992000]           (992,000 vectors, d=960)
  train pool   : gist/train[992000 : 999000]      (7,000 queries)
  validation   : gist/train[999000 : 999500]      (500 queries)
  test         : gist/test[0 : 1000]              (official benchmark queries)

The three pools are disjoint by construction.  Ground truth is recomputed
against the 992k index base (the shipped GT is for the full 1M and would leak
the 8k held-out vectors).

Stages (run `gt` first, then `graph`; `all` does both):
  python prepare_gist.py --stage gt
  python prepare_gist.py --stage graph
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core.paths import (GT_DIR, SPLIT_DIR, CACHE_DIR,
                                 dataset_path)          # noqa: E402
from anns_exp.core.gt import exact_gt_chunked            # noqa: E402
from anns_exp.core.graph import build_graph              # noqa: E402

N_INDEX = 992_000
TRAIN_SLICE = (992_000, 999_000)
VAL_SLICE = (999_000, 999_500)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem(msg):
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    print(f"[mem] {msg}: availPhys={m.ullAvailPhys/1e9:.2f}GB "
          f"availPage={m.ullAvailPageFile/1e9:.2f}GB", flush=True)


def load_base():
    p = dataset_path("gist")
    with h5py.File(p, "r") as f:
        base = f["train"][:N_INDEX].astype(np.float32, copy=False)
        test = f["test"][:].astype(np.float32, copy=False)
        train_q = f["train"][TRAIN_SLICE[0]:TRAIN_SLICE[1]].astype(np.float32)
        val_q = f["train"][VAL_SLICE[0]:VAL_SLICE[1]].astype(np.float32)
    print(f"[data] base={base.shape} test={test.shape} "
          f"train_pool={train_q.shape} val={val_q.shape}", flush=True)
    return base, test, train_q, val_q


def stage_gt():
    base, test, train_q, val_q = load_base()
    mem("after load, before GT")
    sets = {"test": test, "train": train_q, "val": val_q}
    for name, qs in sets.items():
        out = os.path.join(GT_DIR, f"gist_{name}_k100.npy")
        if os.path.exists(out):
            print(f"[gt] {out} exists, skip", flush=True)
            continue
        print(f"[gt] computing {name} {qs.shape}", flush=True)
        gt = exact_gt_chunked(base, qs, k=100, qblock=64)
        np.save(out, gt)
        print(f"[gt] saved {out}", flush=True)
    del base
    splits = {
        "dataset": "GIST1M",
        "source_file": dataset_path("gist"),
        "dimension": 960,
        "metric": "squared L2",
        "index_base": {"start": 0, "stop": N_INDEX, "n": N_INDEX},
        "train_pool": {"start": TRAIN_SLICE[0], "stop": TRAIN_SLICE[1],
                       "n": TRAIN_SLICE[1] - TRAIN_SLICE[0]},
        "validation": {"start": VAL_SLICE[0], "stop": VAL_SLICE[1],
                       "n": VAL_SLICE[1] - VAL_SLICE[0]},
        "test": {"start": 0, "stop": len(test), "n": len(test)},
        "seed": 42,
    }
    with open(os.path.join(SPLIT_DIR, "gist_splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    print("[gt] splits saved", flush=True)


def stage_graph():
    base, _, _, _ = load_base()
    mem("before graph build")
    g = build_graph(base, M=16, efc=200,
                    cache_name=f"gist_idx{N_INDEX}_M16_efc200")
    mem("after graph build")
    print(f"[graph] degree0={g.degree0} max_level={g.max_level()} "
          f"entry={g.entry}", flush=True)
    print(f"[cache] {sorted(os.listdir(CACHE_DIR))}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gt", "graph", "all"], default="all")
    a = ap.parse_args()
    if a.stage in ("gt", "all"):
        stage_gt()
    if a.stage in ("graph", "all"):
        stage_graph()
