"""E1 companion: quality-matched fixed-ef ("fixed-ef@p").

The A group in e1_compare tunes ef so that the *mean* recall is 0.95, which
leaves ~39% of queries below the target.  ANNiE-S promises that p% of
queries reach the target; to compare cost at the same per-query quality we
must tune ef until p% of the queries reach 0.95 instead.

Run:
  .venv\\Scripts\\python.exe -I src\\exp\\e1_fixed_at_p.py --dataset sift --nb 0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from core.gt import load_ann, exact_gt, holdout_train_queries   # noqa: E402
from core.graph import build_graph                             # noqa: E402
from methods import baselines                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--nb", type=int, default=0)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--nval", type=int, default=300)
    ap.add_argument("--ntrain", type=int, default=3000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rt", type=float, default=0.95)
    ap.add_argument("--hit", type=float, default=0.95)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    args = ap.parse_args()

    nb = args.nb or None
    base, q_all, _ = load_ann(args.dataset, nb=nb,
                              normalize=args.dataset in ("deep", "glove"))
    test_q, val_q = q_all[:args.nq], q_all[args.nq:args.nq + args.nval]
    index_base, train_q, _ = holdout_train_queries(base, args.ntrain, seed=0)
    del base
    gt_test = exact_gt(index_base, test_q, k=args.k)
    gt_val = exact_gt(index_base, val_q, k=args.k)
    g = build_graph(index_base, M=args.m, efc=args.efc,
                    cache_name=f"{args.dataset}_hold{args.ntrain}")

    t0 = time.time()
    ef = baselines.tune_fixed_hitrate(g, val_q, gt_val, args.k, args.rt,
                                      args.hit)
    print(f"[{args.dataset}] ef@{args.hit:.2f} = {ef} "
          f"(tuned in {time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    res = baselines.run_fixed(g, test_q, gt_test, args.k, ef)
    secs = time.time() - t0
    rec, ndc = res["recall"], res["ndc"]
    row = {
        "dataset": args.dataset,
        "nb": args.nb or "full",
        "ef": ef,
        "avg_ndc": float(ndc.mean()),
        "median_ndc": float(np.median(ndc)),
        "p95_ndc": float(np.percentile(ndc, 95)),
        "hit_rate": float(np.mean(rec >= args.rt - 1e-6)),
        "mean_recall": float(rec.mean()),
        "ms_per_query": secs / len(rec) * 1e3,
        "note": f"fixed-ef tuned so {args.hit:.0%} of queries reach {args.rt}",
    }
    print(f"   avg NDC={row['avg_ndc']:.0f}  hit={row['hit_rate']:.3f}  "
          f"mean recall={row['mean_recall']:.4f}  ms/q={row['ms_per_query']:.2f}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"e1_fixed_at_p_{args.dataset}.csv")
    header = list(row.keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"   appended to {path}")


if __name__ == "__main__":
    main()
