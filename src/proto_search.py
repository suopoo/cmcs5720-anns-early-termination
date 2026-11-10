"""Prototype 2: instrumented HNSW base-layer search in numpy + faiss ground truth.

Validates that our own search reproduces faiss's recall while exposing
per-query NDC (number of distance computations) and full search trace.
"""
import time, heapq
import numpy as np

Z = np.load(r"D:\大学\CUCS\CMCS5720\week5\proto_graph.npz")
base, queries, gt0 = Z["base"], Z["queries"], Z["gt"]
levels, offsets, neighbors = Z["levels"], Z["offsets"], Z["neighbors"]
entry = int(Z["entry"][0])
n, d = base.shape
K = 10

# recompute ground truth on THIS subset (ann-benchmarks GT refers to full 1M)
def gt_exact(queries, base, k):
    out = np.empty((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries):
        dd = np.sum((base - q) ** 2, axis=1)
        out[i] = np.argpartition(dd, k)[:k][np.argsort(dd[np.argpartition(dd, k)[:k]])]
    return out

t0 = time.time(); gt = gt_exact(queries, base, K); print(f"gt_exact {time.time()-t0:.1f}s")

M2 = int(offsets[1] - offsets[0])
print("level0 slots per node =", M2)

def neighbors_of(node, layer):
    """faiss layout: level0 = [off, off+2M); level l>=1 = [off+2M+(l-1)*M, ...+M)."""
    off = offsets[node]
    if layer == 0:
        return neighbors[off:off + M2]
    s = off + M2 + (layer - 1) * (M2 // 2)
    return neighbors[s:s + M2 // 2]

def search_instrumented(q, ef, k=K, trace=False):
    nndc = 0
    # ---- upper layers: greedy width 1 ----
    cur = entry
    cur_d = float(np.linalg.norm(q - base[cur])); nndc += 1
    max_level = int(levels[cur])
    for layer in range(max_level, 0, -1):
        changed = True
        while changed:
            changed = False
            for nb in neighbors_of(cur, layer):
                if nb < 0:
                    continue
                dd = float(np.linalg.norm(q - base[nb])); nndc += 1
                if dd < cur_d:
                    cur, cur_d, changed = nb, dd, True
    # ---- base layer: beam search ----
    visited = {cur}
    cand = [(cur_d, cur)]           # min-heap
    res = [(-cur_d, cur)]           # max-heap (as negative)
    trace_rows = []
    while cand:
        cd, c = heapq.heappop(cand)
        if -res[0][0] < cd and len(res) >= ef:
            break
        for nb in neighbors_of(c, 0):
            if nb < 0 or nb in visited:
                continue
            visited.add(nb)
            dd = float(np.linalg.norm(q - base[nb])); nndc += 1
            if len(res) < ef or dd < -res[0][0]:
                heapq.heappush(cand, (dd, nb))
                heapq.heappush(res, (-dd, nb))
                if len(res) > ef:
                    heapq.heappop(res)
            if trace:
                trace_rows.append((nndc, len(res), -res[0][0], dd))
    ids = [x[1] for x in heapq.nlargest(k, res, key=lambda t: t[0])]
    return ids, nndc, trace_rows

for ef in (50, 100, 200):
    t0 = time.time(); recs, ndcs = [], []
    for i, q in enumerate(queries[:200]):
        ids, ndc, _ = search_instrumented(q, ef)
        recs.append(len(set(ids) & set(gt[i])) / K); ndcs.append(ndc)
    dt = (time.time() - t0) / 200
    print(f"ef={ef:4d}  recall@{K}={np.mean(recs):.4f}  NDC={np.mean(ndcs):7.1f}  "
          f"{dt*1e3:6.2f} ms/query (python)")

# faiss reference on same subset
import faiss
idx = faiss.IndexHNSWFlat(d, 16); idx.hnsw.efConstruction = 200
idx.add(base)
for ef in (50, 100, 200):
    idx.hnsw.efSearch = ef
    _, I = idx.search(queries[:200], K)
    print(f"faiss ef={ef:4d} recall@{K}={np.mean([len(set(I[i]) & set(gt[i]))/K for i in range(200)]):.4f}")
