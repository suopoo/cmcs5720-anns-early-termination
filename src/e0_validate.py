"""E0 fidelity check: our instrumented search vs faiss, on a dataset subset.

Run:
  .venv\\Scripts\\python.exe -I src\\e0_validate.py [nb] [dataset]

`dataset` defaults to sift; angular datasets (glove, deep) are L2-normalised
so that L2 ranking matches the cosine/angular ground truth.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.gt import load_ann, exact_gt
from core.graph import build_graph
from core.search import run_query

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
DS = sys.argv[2] if len(sys.argv) > 2 else "sift"
NQ = 200
K = 10
M, EFC = 16, 200
NORMALIZE = DS in ("deep", "glove")

print(f"== loading {DS} nb={NB} nq={NQ} normalize={NORMALIZE} ==")
t0 = time.time()
base, queries, _ = load_ann(DS, nb=NB, nq=NQ, normalize=NORMALIZE)
print(f"load {time.time()-t0:.1f}s base={base.shape} queries={queries.shape}")

t0 = time.time()
gt = exact_gt(base, queries, k=100)
print(f"exact GT {time.time()-t0:.1f}s")

t0 = time.time()
g = build_graph(base, M=M, efc=EFC, cache_name=DS)
print(f"graph {time.time()-t0:.1f}s  degree0={g.degree0} max_level={g.max_level()}")

# faiss reference index: build once, sweep efSearch (rebuilding per ef was
# O(3x) the graph construction time and polluted the timing).
import faiss
ref = faiss.IndexHNSWFlat(base.shape[1], M)
ref.hnsw.efConstruction = EFC
ref.add(base)

print("\n== recall: ours vs faiss ==")
print(f"{'ef':>4} {'ours':>8} {'faiss':>8} {'match':>6} {'NDC':>8} {'ms/q':>7}")
for ef in (50, 100, 200):
    t0 = time.time()
    ours = [run_query(g, q, k=K, ef=ef) for q in queries]
    ms = (time.time() - t0) / len(queries) * 1e3
    r_ours = np.mean([len(set(o["ids"]) & set(gt[i][:K])) / K for i, o in enumerate(ours)])
    ndc = np.mean([o["ndc"] for o in ours])

    ref.hnsw.efSearch = ef
    _, I = ref.search(queries, K)
    r_faiss = np.mean([len(set(I[i]) & set(gt[i][:K])) / K for i in range(len(queries))])
    print(f"{ef:>4} {r_ours:>8.4f} {r_faiss:>8.4f} {'OK' if abs(r_ours-r_faiss)<1e-9 else 'DIFF':>6} "
          f"{ndc:>8.0f} {ms:>7.2f}")
del ref

print("\n== unlimited-beam (LAET/ANNiE-S) mode smoke test ==")
t0 = time.time()
out = [run_query(g, q, k=K, ef=None, max_ndc=3000) for q in queries[:20]]
ms = (time.time() - t0) / 20 * 1e3
print(f"20 queries, budget=3000: mean NDC={np.mean([o['ndc'] for o in out]):.0f} "
      f"{ms:.1f} ms/q, reasons={set(o['reason'] for o in out)}")
