"""Prototype: export faiss HNSW graph -> instrumented base-layer search in numpy.

Goal: verify we can (1) build HNSW fast with faiss, (2) export the graph,
(3) re-implement the base-layer greedy beam search exactly enough to reproduce
faiss recalls while counting distance computations per query.
"""
import sys, time, json
import numpy as np
import h5py
import faiss

DATA = r"D:\大学\CUCS\CMCS5720\hnswlib\data\sift-128-euclidean.hdf5"
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
M = 16
EFC = 200
EFS = 100
K = 10

t0 = time.time()
with h5py.File(DATA, "r") as f:
    print("keys:", list(f.keys()))
    base = f["train"][:NB].astype("float32")
    queries = f["test"][:1000].astype("float32")
    gt = f["neighbors"][:1000, :K].astype("int64")
print(f"load {time.time()-t0:.1f}s base={base.shape} q={queries.shape}")

d = base.shape[1]
t0 = time.time()
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = EFC
index.add(base)
print(f"build {time.time()-t0:.1f}s  ntotal={index.ntotal}")

# ---- export graph ----
t0 = time.time()
h = index.hnsw
levels = faiss.vector_to_array(h.levels).astype(np.int32)
offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
neighbors = faiss.vector_to_array(h.neighbors).astype(np.int32)
entry = int(h.entry_point)
print(f"export {time.time()-t0:.1f}s levels={levels.shape} offsets={offsets.shape} "
      f"neighbors={neighbors.shape} entry={entry}")
print("neighbors range:", neighbors.min(), neighbors.max(), "n=", index.ntotal)
print("offset[0:5]", offsets[:5], "offset[-1]", offsets[-1])

# ---- faiss baseline search ----
index.hnsw.efSearch = EFS
t0 = time.time()
D_f, I_f = index.search(queries, K)
t_faiss = (time.time() - t0) / len(queries)
recall_faiss = np.mean([len(set(I_f[i]) & set(gt[i])) / K for i in range(len(queries))])
print(f"faiss search: {t_faiss*1e3:.2f} ms/q recall@{K}={recall_faiss:.4f}")

np.savez_compressed(
    r"D:\大学\CUCS\CMCS5720\week5\proto_graph.npz",
    levels=levels, offsets=offsets, neighbors=neighbors,
    entry=np.array([entry]), base=base, queries=queries, gt=gt,
)
print("saved proto_graph.npz")
