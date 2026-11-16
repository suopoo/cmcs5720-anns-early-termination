"""End-to-end latency measurement on a 30-query subset.

NDC is the portable cost, but the task also requires feature-extraction and
inference time.  This script runs the actual online loop (not replay) for
each group and measures wall time, feature time and prediction time.
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core import paths
from anns_exp.core.graph import build_graph
from anns_exp.core.search import run_query
from anns_exp.features import build, darth_vec, laet_tail
from anns_exp import gbdt
from anns_exp.exp_common import write_csv, log
from anns_exp.trace import load_split, load_base

EXP = paths.exp_dir("exp01")
MODELS = os.path.join(EXP, "models")
NQ = 30
K, RT, F0, H, S = 10, 0.9, 1260, 4, 200


def main():
    base = load_base()
    g = build_graph(base, M=16, efc=200, cache_name="gist_idx992000")
    q, gt = load_split("test"); q = q[:NQ]; gt = gt[:NQ]
    summary = pd.read_csv(os.path.join(EXP, "data", "summary_metrics.csv"))
    meta = json.load(open(os.path.join(paths.EXP_ROOT, "shared", "traces", "meta.json")))
    ef = int(meta["ef99_k10"]["ef"])
    rows, perq = [], []
    for group in ("L0", "L1", "L2"):
        model = joblib.load(os.path.join(MODELS, f"laet_{group}.joblib"))
        row = summary[(summary.method == "LAET") & (summary.group == group)].iloc[0]
        mult, pmax = float(row["multiplier"]), float(row["pred_max"])
        tots, fs, ps = [], [], []
        for i, (qq, gts) in enumerate(zip(q, gt)):
            st = {"feat": None, "b": None, "ft": 0.0, "pt": 0.0}
            pops = []
            def stop(c, st=st, pops=pops):
                pops.append((c.pop_dist, c.pop_id))
                if st["b"] is None:
                    if c.ndc >= F0:
                        t0 = time.perf_counter()
                        st["feat"] = build(group, g, c, pops, RT, K, H, S)
                        st["ft"] = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        p = float(gbdt.predict(model, st["feat"][None, :])[0])
                        st["pt"] = time.perf_counter() - t0
                        st["b"] = max(F0, min(2.0 ** max(p, 0.0) * mult, pmax))
                    return False
                return c.ndc >= st["b"]
            t0 = time.perf_counter()
            o = run_query(g, qq, k=K, ef=None, gt_set=set(gts[:K].tolist()),
                          max_ndc=40000, stop=stop)
            tot = (time.perf_counter() - t0) * 1e3
            tots.append(tot); fs.append(st["ft"] * 1e3); ps.append(st["pt"] * 1e3)
            perq.append({"method": "LAET", "group": group, "query_id": i,
                         "total_ms": tot, "feature_ms": st["ft"] * 1e3,
                         "predict_ms": st["pt"] * 1e3, "ndc": o["ndc"],
                         "recall": o["recall"]})
        rows.append({"method": "LAET", "group": group,
                     "mean_ms": float(np.mean(tots)),
                     "p50_ms": float(np.median(tots)),
                     "p95_ms": float(np.percentile(tots, 95)),
                     "feature_ms": float(np.mean(fs)),
                     "predict_ms": float(np.mean(ps)),
                     "calls": 1.0,
                     "mean_ndc": float(np.mean([r["ndc"] for r in perq
                                                if r["group"] == group]))})
        log(f"{group}: {rows[-1]['mean_ms']:.2f} ms/query")
    for group in ("D0", "D1", "D2"):
        model = joblib.load(os.path.join(MODELS, f"darth_{group}.joblib"))
        tots, fs, ps, calls = [], [], [], []
        for i, (qq, gts) in enumerate(zip(q, gt)):
            st = {"last": -1, "next": F0, "calls": 0, "ft": 0.0, "pt": 0.0}
            pops = []
            def stop(c, st=st, pops=pops):
                pops.append((c.pop_dist, c.pop_id))
                if c.ndc < st["next"]:
                    return False
                t0 = time.perf_counter()
                f = build(group, g, c, pops, RT, K, H, S)
                st["ft"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                p = float(gbdt.predict(model, f[None, :])[0])
                st["pt"] += time.perf_counter() - t0
                st["calls"] += 1
                st["next"] = c.ndc + 200
                return p >= RT
            t0 = time.perf_counter()
            o = run_query(g, qq, k=K, ef=ef, gt_set=set(gts[:K].tolist()),
                          max_ndc=40000, stop=stop)
            tot = (time.perf_counter() - t0) * 1e3
            tots.append(tot); fs.append(st["ft"] * 1e3); ps.append(st["pt"] * 1e3)
            calls.append(st["calls"])
            perq.append({"method": "DARTH", "group": group, "query_id": i,
                         "total_ms": tot, "feature_ms": st["ft"] * 1e3,
                         "predict_ms": st["pt"] * 1e3, "ndc": o["ndc"],
                         "recall": o["recall"], "calls": st["calls"]})
        rows.append({"method": "DARTH", "group": group,
                     "mean_ms": float(np.mean(tots)),
                     "p50_ms": float(np.median(tots)),
                     "p95_ms": float(np.percentile(tots, 95)),
                     "feature_ms": float(np.mean(fs)),
                     "predict_ms": float(np.mean(ps)),
                     "calls": float(np.mean(calls)),
                     "mean_ndc": float(np.mean([r["ndc"] for r in perq
                                                if r["group"] == group]))})
        log(f"{group}: {rows[-1]['mean_ms']:.2f} ms/query")
    write_csv(rows, os.path.join(EXP, "analysis", "latency_summary.csv"))
    write_csv(perq, os.path.join(EXP, "data", "latency_per_query.csv"))


if __name__ == "__main__":
    main()
