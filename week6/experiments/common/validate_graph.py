"""Validate the parsed CSR graph against hnswlib's own search."""
from __future__ import annotations

import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core.paths import CACHE_DIR, GT_DIR, dataset_path  # noqa
from anns_exp.core.graph import build_graph                     # noqa
from anns_exp.core.search import run_query                      # noqa

NQ, EF, K = 50, 100, 10


def main():
    with h5py.File(dataset_path("gist"), "r") as f:
        base = f["train"][:992000].astype(np.float32, copy=False)
        q = f["test"][:NQ].astype(np.float32)
    g = build_graph(base, M=16, efc=200, cache_name="gist_idx992000")
    gt = np.load(os.path.join(GT_DIR, "gist_test_k100.npy"))[:NQ, :K]
    ref = np.load(os.path.join(GT_DIR, "hnswlib_ref_ids_k10.npz"))["ids"]
    ours, overlap, ndc = [], [], []
    t0 = time.time()
    for i in range(NQ):
        o = run_query(g, q[i], k=K, ef=EF)
        ours.append(o["ids"])
        overlap.append(len(set(o["ids"]) & set(ref[i])) / K)
        ndc.append(o["ndc"])
    rec = np.mean([len(set(ours[i]) & set(gt[i])) / K for i in range(NQ)])
    ov = np.mean(overlap)
    lines = [
        "# Graph / search validation",
        "",
        f"- Index: hnswlib M=16 efc=200 on GIST1M `train[:992000]`",
        f"- Parsed CSR: degree0={g.degree0}, max_level={g.max_level()}, "
        f"entry={g.entry}",
        f"- Queries checked: {NQ} official test queries, efSearch={EF}, k={K}",
        f"- Mean recall@10 vs exact GT: **{rec:.4f}**",
        f"- Mean ID overlap with hnswlib's own search: **{ov:.4f}**",
        f"- Mean NDC per query (our loop): {np.mean(ndc):.1f}",
        f"- Wall time: {time.time()-t0:.1f}s",
        "",
        "The overlap is computed as |our top-10 ∩ hnswlib top-10| / 10.  A "
        "correctly parsed adjacency gives ~1.0; tie-breaking can account for "
        "tiny differences.",
    ]
    out = os.path.join(os.path.dirname(HERE), "shared", "logs",
                       "validation_results.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written", out)


if __name__ == "__main__":
    main()
