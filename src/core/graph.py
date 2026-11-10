"""HNSW graph construction / export / caching.

Uses faiss to build the graph (fast, C++), then exports the CSR adjacency so
that our own instrumented search can run on it.  We never hold two copies of
the base vectors: the faiss index is released right after export.
"""
from __future__ import annotations

import os
import gc
import numpy as np
import faiss

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cache")


class Graph:
    """Compressed adjacency of an HNSW index + the vectors it indexes."""

    __slots__ = ("base", "levels", "offsets", "neighbors", "entry",
                 "degree0", "degree_up", "n", "d", "_bn", "_stamp", "_stamp_val")

    def __init__(self, base, levels, offsets, neighbors, entry):
        self.base = base
        self.levels = levels
        self.offsets = offsets
        self.neighbors = neighbors
        self.entry = int(entry)
        self.n, self.d = base.shape
        self.degree0 = int(offsets[1] - offsets[0])
        self.degree_up = self.degree0 // 2
        # ||x||^2, used by the BLAS-based L2^2 evaluation
        self._bn = np.einsum("ij,ij->i", base, base).astype(np.float32)
        self._stamp = None
        self._stamp_val = 0

    # -- visited-set buffer (one int32 per node, O(1) reset per query) ------
    def stamp_buffer(self):
        if self._stamp is None or self._stamp.shape[0] != self.n:
            self._stamp = np.zeros(self.n, dtype=np.int32)
            self._stamp_val = 0
        return self._stamp

    def next_stamp(self):
        self._stamp_val += 1
        return self._stamp_val

    # -- distances ---------------------------------------------------------
    def dist_batch(self, q, qn, ids):
        """Squared L2 from q to base[ids] (BLAS gemv; ~10x faster than norm)."""
        return self._bn[ids] - 2.0 * (self.base[ids] @ q) + qn

    # -- adjacency ---------------------------------------------------------
    def neighbors_of(self, node, layer):
        off = self.offsets[node]
        if layer == 0:
            return self.neighbors[off:off + self.degree0]
        s = off + self.degree0 + (layer - 1) * self.degree_up
        return self.neighbors[s:s + self.degree_up]

    def max_level(self):
        return int(self.levels[self.entry])


def _cache_path(name, nb, M, efc):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}_n{nb}_M{M}_efc{efc}_graph.npz")


def build_graph(base, M=16, efc=200, cache_name=None, verbose=True):
    """Build (or load from cache) the HNSW graph for `base`."""
    nb = base.shape[0]
    path = _cache_path(cache_name, nb, M, efc) if cache_name else None
    if path and os.path.exists(path):
        z = np.load(path)
        g = Graph(base, z["levels"], z["offsets"], z["neighbors"], z["entry"][0])
        if verbose:
            print(f"[graph] loaded cache {os.path.basename(path)}")
        return g

    index = faiss.IndexHNSWFlat(base.shape[1], M)
    index.hnsw.efConstruction = efc
    index.add(base)
    h = index.hnsw
    levels = faiss.vector_to_array(h.levels).astype(np.int32)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int32)
    entry = int(h.entry_point)
    del index
    gc.collect()

    g = Graph(base, levels, offsets, neighbors, entry)
    if path:
        np.savez(path, levels=levels, offsets=offsets,
                 neighbors=neighbors, entry=np.array([entry], dtype=np.int32))
    if verbose:
        print(f"[graph] built n={nb} d={base.shape[1]} M={M} efc={efc} "
              f"entry={entry} max_level={g.max_level()}")
    return g
