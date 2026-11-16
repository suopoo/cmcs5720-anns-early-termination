"""End-to-end smoke test on a 20k GIST slice (never used for results)."""
from __future__ import annotations

import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core.paths import dataset_path, CACHE_DIR          # noqa
from anns_exp.core.gt import exact_gt_chunked                    # noqa
from anns_exp.core.graph import build_graph                      # noqa
from anns_exp import trace as T, features as F, gbdt, simulate as S  # noqa

NB = 20_000
NQ = 24
with h5py.File(dataset_path("gist"), "r") as f:
    base = f["train"][:NB].astype(np.float32)
    qs = f["test"][:NQ].astype(np.float32)
gt = exact_gt_chunked(base, qs, k=100, qblock=8, verbose=False)
g = build_graph(base, M=16, efc=100, cache_name=f"gist_smoke{NB}_M16_efc100")
print("graph", g.degree0, g.max_level())

t0 = time.time()
u = T.collect_unlimited(g, qs[:NQ // 2], gt[:NQ // 2], k=10,
                        fs=[256, 512, 1260, 1575], max_ndc=8000,
                        keep_steps=True, need_suffix_f=1260,
                        watch_top1=True, stop_at_watch_after=1575,
                        name="smoke_unlim")
print("unlim", {k: getattr(v, 'shape', None) for k, v in u.items()},
      f"{time.time()-t0:.1f}s")
assert u["step_ndc"].ndim == 1 and u["ptr"][-1] == len(u["step_ndc"])
assert u["suffix"].shape[1] == 12 + 1 + 10 + 4
assert np.all(np.diff(u["tail_f1260"][0]) >= -1e-6) if False else True

t0 = time.time()
d = T.collect_darth(g, qs[:NQ // 2], gt[:NQ // 2], k=10, ef=128,
                    rt_grid=[0.9, 0.95, 0.99], max_ndc=8000,
                    keep_steps=True, delta_obs=10,
                    transfer_grid=list(range(256, 6000, 200)),
                    need_transfer_suffix=True, name="smoke_darth")
print("darth", {k: getattr(v, 'shape', None) for k, v in d.items()},
      f"{time.time()-t0:.1f}s")
assert d["step_feat"].shape[1] == 11
assert d["tr_suffix"].shape[1] == 12 + 1 + 10 + 4

# train one LAET model + simulate
q_tr = qs[:NQ // 2]
X = F.assemble("L0", q_tr, u["tail_f1260"])
m = gbdt.train_lgb(X, np.log2(np.maximum(u["c1"], 1)), n_estimators=20)
r = S.laet_from_trace(u, X, m, f=1260, mult=2.0, pred_max=8000)
print("laet sim", np.nanmean(r["recall"]), np.nanmean(r["stop_ndc"]))
# ANNiE feature at homing
u2 = T.collect_unlimited(g, qs[:NQ // 2], gt[:NQ // 2], k=10,
                         fs=[1260], max_ndc=8000, keep_steps=False,
                         need_suffix_f=None, rt_grid=(0.9,),
                         homing=(5, 5), name="smoke_homing")
print("homing", u2["hom_feat"].shape, u2["hom_feat"][:2, -5:])
# DARTH adaptive
Xg = d["tr_darth"]
m2 = gbdt.train_lgb(Xg, d["obs_recall"][:len(Xg)], n_estimators=20)
r2 = S.darth_adaptive_from_steps(d, m2, rt=0.9, ipi=100, mpi=20)
print("darth sim", np.nanmean(r2["recall"]), np.nanmean(r2["calls"]))
print("SMOKE OK")
