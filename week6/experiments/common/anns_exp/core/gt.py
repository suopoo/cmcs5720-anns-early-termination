"""Ground truth + dataset loading.

ann-benchmarks hdf5 layout: `train` (base), `test` (queries),
`neighbors` (top-100 GT ids for the *full* base set), `distances`.
If you slice the base, the shipped GT is invalid -> recompute.
"""
from __future__ import annotations

import os
import numpy as np
import h5py

try:
    from .paths import DATA_DIRS, DATA_DIRS as _D
except ImportError:
    from paths import DATA_DIRS, DATA_DIRS as _D
DATA_DIR = DATA_DIRS[-1]


def _resolve(filename):
    for d in DATA_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, filename)      # canonical location


DATASETS = {
    "sift":     _resolve("sift-128-euclidean.hdf5"),
    "glove":    _resolve("glove-100-angular.hdf5"),
    "gist":     _resolve("gist.hdf5"),
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
    import faiss
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


def exact_gt_chunked(base, queries, k=100, qblock=64, verbose=True):
    """Exact k-NN ids by blocked BLAS GEMM (memory ~ qblock*n float32).

    Avoids a second full copy of `base` (IndexFlatL2 would duplicate it),
    which matters for GIST 1M on a 16 GB machine.
    """
    import time
    n, d = base.shape
    out = np.empty((len(queries), k), dtype=np.int32)
    bt = np.ascontiguousarray(base.T)          # F-contiguous view for sgemm
    bn = np.einsum("ij,ij->i", base, base).astype(np.float32)
    t0 = time.time()
    for s in range(0, len(queries), qblock):
        qb = np.ascontiguousarray(queries[s:s + qblock], dtype=np.float32)
        qn = np.einsum("ij,ij->i", qb, qb).astype(np.float32)
        D = qb @ bt                            # (b, n) float32
        D *= -2.0
        D += bn[None, :]
        D += qn[:, None]
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(D, part, axis=1)
        order = np.argsort(vals, axis=1)
        out[s:s + qb.shape[0]] = np.take_along_axis(part, order, axis=1)
        del D, part, vals
        if verbose and (s // qblock) % 10 == 0:
            print(f"  gt {s + len(qb)}/{len(queries)}  {time.time()-t0:.1f}s",
                  flush=True)
    return out
