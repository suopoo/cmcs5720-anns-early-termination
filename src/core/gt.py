"""Ground truth + dataset loading.

ann-benchmarks hdf5 layout: `train` (base), `test` (queries),
`neighbors` (top-100 GT ids for the *full* base set), `distances`.
If you slice the base, the shipped GT is invalid -> recompute.
"""
from __future__ import annotations

import os
import numpy as np
import h5py
import faiss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Dataset directories, in priority order.  ANNS_DATA overrides everything;
# the legacy path keeps the author's local layout working.
_LEGACY = r"D:\大学\CUCS\CMCS5720\hnswlib\data"
DATA_DIRS = [d for d in (os.environ.get("ANNS_DATA"), _LEGACY, DATA_DIR) if d]


def _resolve(filename):
    for d in DATA_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, filename)      # canonical location


DATASETS = {
    "sift":     _resolve("sift-128-euclidean.hdf5"),
    "glove":    _resolve("glove-100-angular.hdf5"),
    "gist":     _resolve("gist-960-euclidean.hdf5"),
    "fashion":  _resolve("fashion-mnist-784-euclidean.hdf5"),
    "deep":     _resolve("deep-image-96-angular.hdf5"),
}


def load_ann(name, nb=None, nq=None, normalize=False):
    """Return (base, queries, gt_ids_or_None).

    gt is only returned when the full base is used (shipped GT is valid then).
    """
    path = DATASETS[name]
    with h5py.File(path, "r") as f:
        total = f["train"].shape[0]
        n = total if nb is None else min(nb, total)
        base = f["train"][:n].astype(np.float32)
        q = f["test"][: (nq or f["test"].shape[0])].astype(np.float32)
        gt = f["neighbors"][: len(q)].astype(np.int32) if n == total else None
    if normalize:                       # angular datasets (glove/deep)
        base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return base, q, gt


def exact_gt(base, queries, k=100, batch=256):
    """Exact k-NN ids with faiss (BLAS), memory-friendly batching."""
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    out = np.empty((len(queries), k), dtype=np.int32)
    for s in range(0, len(queries), batch):
        _, I = index.search(queries[s:s + batch], k)
        out[s:s + batch] = I
    del index
    return out


def holdout_train_queries(base, n_train=10_000, seed=0):
    """Split off `n_train` vectors from base to serve as *training queries*.

    The held-out vectors are removed from the index, so there is no leakage
    (they are not in the graph).  Returns (index_base, train_queries, keep_idx).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base), size=n_train, replace=False)
    mask = np.ones(len(base), dtype=bool)
    mask[idx] = False
    return base[mask], base[idx], np.flatnonzero(mask)
