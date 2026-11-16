"""Reference search with hnswlib itself (for graph-parser validation)."""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import hnswlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anns_exp.core.paths import CACHE_DIR, dataset_path, GT_DIR  # noqa

NQ, EF, K = 50, 100, 10


def main():
    idx = "D:/tmp/anns_wk6/gist_992000_M16_efc200.hnsw"
    p = hnswlib.Index(space="l2", dim=960)
    p.load_index(idx, max_elements=992000)
    p.set_ef(EF)
    with h5py.File(dataset_path("gist"), "r") as f:
        q = f["test"][:NQ].astype(np.float32)
    labels, dist = p.knn_query(q, k=K)
    out = os.path.join(GT_DIR, "hnswlib_ref_ids_k10.npz")
    np.savez(out, ids=labels, dist=dist)
    # recall against the recomputed exact GT on the 992k base
    gt = np.load(os.path.join(GT_DIR, "gist_test_k100.npy"))[:NQ, :K]
    rec = np.mean([len(set(labels[i]) & set(gt[i])) / K for i in range(NQ)])
    print("saved", out, "mean recall@10 =", rec)


if __name__ == "__main__":
    main()
