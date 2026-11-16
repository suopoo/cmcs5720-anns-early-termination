"""Build the shared GIST index with hnswlib (multi-threaded).

faiss' HNSW construction is single-threaded and would need hours for 992k
960-d vectors; hnswlib uses all 14 cores.  The index is then parsed into the
CSR cache consumed by the instrumented search (parse_hnswlib.py).
"""
from __future__ import annotations

import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anns_exp.core.paths import dataset_path, CACHE_DIR  # noqa

import hnswlib  # noqa

N = 992_000
M, EFC = 16, 200
OUT = "D:/tmp/anns_wk6/gist_{}_M{}_efc{}.hnsw".format(N, M, EFC)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        print("exists", OUT)
        return
    t0 = time.time()
    with h5py.File(dataset_path("gist"), "r") as f:
        data = f["train"][:N].astype(np.float32)
    print(f"loaded {data.shape} in {time.time()-t0:.0f}s", flush=True)
    p = hnswlib.Index(space="l2", dim=data.shape[1])
    p.init_index(max_elements=N, ef_construction=EFC, M=M, random_seed=42)
    t0 = time.time()
    p.add_items(data, num_threads=14)
    print(f"built in {time.time()-t0:.0f}s", flush=True)
    p.save_index(OUT)
    print("saved", OUT, os.path.getsize(OUT) / 1e9, "GB", flush=True)


if __name__ == "__main__":
    main()
