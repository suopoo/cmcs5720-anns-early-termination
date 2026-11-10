"""E1 -- five-way comparison at a fixed declarative recall target.

Protocol (实验1):
  1000 test queries, k=100, R_t=0.95
    A  Fixed-ef  : tune ef so the *mean* recall hits 0.95, then measure
    B  LAET      : tune `multiplier` so the mean recall hits 0.95
    C  DARTH     : run untuned, report NDC and the fraction reaching 0.95
    D  ANNiE-S   : p=95% quantile model, run untuned
    E  Oracle    : per-query minimum NDC to reach 0.95 (not online)

Output: results/e1_compare_<tag>.csv and figures/fig_e1_compare_<tag>.png

Run:
  .venv\\Scripts\\python.exe -I src\\exp\\e1_compare.py --dataset sift --nb 200000
  .venv\\Scripts\\python.exe -I src\\exp\\e1_compare.py --dataset sift --nb 0      # full 1M
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
from methods import annie, baselines, darth, laet              # noqa: E402

RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def timed_run(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0


def summarize(name, rec, ndc, secs, rt, **extra):
    rec = np.asarray(rec, dtype=np.float64)
    ndc = np.asarray(ndc, dtype=np.float64)
    row = {
        "method": name,
        "avg_ndc": float(ndc.mean()),
        "median_ndc": float(np.median(ndc)),
        "p95_ndc": float(np.percentile(ndc, 95)),
        "hit_rate": float(np.mean(rec >= rt - 1e-6)),
        "mean_recall": float(rec.mean()),
        "ms_per_query": float(secs / max(len(rec), 1) * 1e3),
    }
    row.update(extra)
    return row


def plot(rows, tag, rt, k, nq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # documented categorical order (light mode); identity is carried by the
    # direct label + marker shape, never by hue alone.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    markers = ["o", "s", "^", "D", "*"]
    order = ["Fixed-ef", "LAET", "DARTH", "ANNiE-S", "Oracle"]
    by = {r["method"]: r for r in rows}

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(surface)

    # ---- panel A: cost vs. target-hit rate ------------------------------
    ax = axes[0]
    ax.set_facecolor(surface)
    ax.axhline(rt, color=muted, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for i, name in enumerate(order):
        if name not in by:
            continue
        r = by[name]
        ax.scatter(r["avg_ndc"], r["hit_rate"], s=110, marker=markers[i],
                   color=colors[i], edgecolors=surface, linewidths=2, zorder=3)
        ax.annotate(f"{name}", (r["avg_ndc"], r["hit_rate"]),
                    textcoords="offset points", xytext=(9, 7),
                    fontsize=9, color=ink)
    ax.text(0.01, rt, f"target recall {rt}",
            transform=ax.get_yaxis_transform(), color=muted, fontsize=8,
            va="bottom", ha="left")
    ax.set_xlabel("Average NDC per query", fontsize=9, color=muted)
    ax.set_ylabel(f"Fraction of queries reaching recall@{k} >= {rt}", fontsize=9,
                  color=muted)
    ax.set_ylim(min(0.80, min(r["hit_rate"] for r in rows) - 0.05), 1.02)
    ax.set_title("A. Same quality, different cost", fontsize=11, color=ink,
                 loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8)

    # ---- panel B: average NDC per method --------------------------------
    ax = axes[1]
    ax.set_facecolor(surface)
    names = [n for n in order if n in by]
    vals = [by[n]["avg_ndc"] for n in names]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, height=0.62, color="#2a78d6", zorder=2)
    for y, v, n in zip(ypos, vals, names):
        ax.text(v * 1.06, y, f"{v:,.0f}", va="center", fontsize=9, color=ink)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9, color=ink)
    ax.set_xlabel("Average NDC per query", fontsize=9, color=muted)
    ax.set_title("B. Distance calculations spent", fontsize=11, color=ink,
                 loc="left")
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(vals) * 1.18)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8)

    fig.text(0.02, 0.015,
             "NDC = distance calculations per query. Oracle = per-query minimum"
             " over all trajectories used (ef95 / ef99 / unlimited beam), with"
             " ground-truth recall: an offline bound, not achievable online.",
             fontsize=7.5, color=muted, ha="left")
    fig.suptitle(f"E1: {nq} queries, k={k}, target recall={rt}  ({tag})",
                 fontsize=12, color=ink, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = os.path.join(FIG_DIR, f"fig_e1_compare_{tag}.png")
    fig.savefig(out, facecolor=surface)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--nb", type=int, default=0, help="base size, 0 = full")
    ap.add_argument("--nq", type=int, default=1000, help="test queries")
    ap.add_argument("--nval", type=int, default=300, help="validation queries")
    ap.add_argument("--ntrain", type=int, default=3000, help="training queries")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rt", type=float, default=0.95)
    ap.add_argument("--p", type=float, default=0.95, help="ANNiE RTG probability")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--max-ndc", type=int, default=40000)
    ap.add_argument("--h", type=int, default=4, help="ANNiE hop count")
    ap.add_argument("--s", type=int, default=200, help="ANNiE subgraph size")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    global RT
    RT, K = args.rt, args.k
    nb = args.nb or None
    tag = args.tag or f"{args.dataset}_n{args.nb or 'full'}_k{K}_rt{RT}"
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    log(f"load {args.dataset} nb={nb or 'full'}")
    base, q_all, _ = load_ann(args.dataset, nb=nb,
                              normalize=args.dataset in ("deep", "glove"))
    log(f"base={base.shape} queries={q_all.shape}")

    nq = args.nq
    if len(q_all) >= nq + args.nval:
        test_q, val_q = q_all[:nq], q_all[nq:nq + args.nval]
    else:
        cut = int(len(q_all) * 0.7)
        test_q, val_q = q_all[:cut], q_all[cut:]
        nq = len(test_q)
    log(f"test={test_q.shape[0]} val={val_q.shape[0]}")

    index_base, train_q, _ = holdout_train_queries(base, args.ntrain, seed=0)
    del base
    log(f"index base={index_base.shape} training queries={train_q.shape[0]}")

    t0 = time.time()
    gt_test = exact_gt(index_base, test_q, k=K)
    gt_val = exact_gt(index_base, val_q, k=K)
    gt_train = exact_gt(index_base, train_q, k=K)
    log(f"ground truth {time.time()-t0:.1f}s")

    t0 = time.time()
    g = build_graph(index_base, M=args.m, efc=args.efc,
                    cache_name=f"{args.dataset}_hold{args.ntrain}")
    log(f"graph {time.time()-t0:.1f}s degree0={g.degree0} "
        f"max_level={g.max_level()}")

    rows = []

    # ---- A: fixed ef ---------------------------------------------------
    log("A. fixed-ef: tuning ef on validation queries")
    ef95 = baselines.tune_fixed_ef(g, val_q, gt_val, K, RT)
    log(f"   ef* = {ef95}")
    resA, secs = timed_run(baselines.run_fixed, g, test_q, gt_test, K, ef95)
    rows.append(summarize("Fixed-ef", resA["recall"], resA["ndc"], secs, RT,
                          note=f"ef={ef95}"))
    log(f"   avg NDC={rows[-1]['avg_ndc']:.0f} hit={rows[-1]['hit_rate']:.3f} "
        f"recall={rows[-1]['mean_recall']:.4f}")

    # ---- index config used by DARTH (plain HNSW able to reach ~0.99) ----
    log("   tuning ef for plain HNSW at ~0.99 (DARTH's base configuration)")
    ef99 = baselines.tune_fixed_ef(g, val_q, gt_val, K, 0.99)
    log(f"   ef_99 = {ef99}")

    # ---- E: oracle -----------------------------------------------------
    log("E. oracle: per-query minimum NDC over all trajectories used")
    (ndc_o, rec_o, reached), secs = timed_run(
        baselines.oracle_min, g, test_q, gt_test, K, RT, args.max_ndc,
        [ef95, ef99, None])
    rows.append(summarize("Oracle", rec_o, ndc_o, secs, RT,
                          note=f"unreachable={int((~reached).sum())}"))
    log(f"   avg NDC={rows[-1]['avg_ndc']:.0f} hit={rows[-1]['hit_rate']:.3f}")

    # ---- B: LAET -------------------------------------------------------
    log("B. LAET: training")
    t0 = time.time()
    lm = laet.train(g, train_q, gt_train, K, args.max_ndc, use_query=True)
    log(f"   model trained in {time.time()-t0:.0f}s (f={lm.f}, "
        f"pred_max={lm.pred_max:.0f})")
    mult, mean_rec, mults = laet.tune_multiplier(g, val_q, gt_val, K, lm,
                                                 args.max_ndc, RT)
    j = int(np.argmin(np.abs(mults - mult)))
    log(f"   tuned multiplier={mult:.3f} (val mean recall={mean_rec[j]:.4f})")
    resB, secs = timed_run(laet.run_laet, g, test_q, gt_test, K, lm, mult,
                           args.max_ndc)
    rows.append(summarize("LAET", resB["recall"], resB["ndc"], secs, RT,
                          note=f"mult={mult:.3f}, f={lm.f}"))
    log(f"   avg NDC={rows[-1]['avg_ndc']:.0f} hit={rows[-1]['hit_rate']:.3f} "
        f"recall={rows[-1]['mean_recall']:.4f}")

    # ---- C: DARTH ------------------------------------------------------
    log("C. DARTH: collecting training observations, then training")
    t0 = time.time()
    X, y, d_rt = darth.collect_training(g, train_q, gt_train, K, ef99, RT,
                                        args.max_ndc, delta=5, max_obs=200)
    log(f"   {len(X):,} observations in {time.time()-t0:.0f}s")
    dm = darth.DARTH.train(X, y, d_rt, RT)
    log(f"   i_pi={dm.ipi:.0f} m_pi={dm.mpi:.0f}")
    resC, secs = timed_run(darth.run_darth, g, test_q, gt_test, K, ef99, dm,
                           args.max_ndc)
    rows.append(summarize("DARTH", resC["recall"], resC["ndc"], secs, RT,
                          note=f"ef={ef99}, calls={resC['calls'].mean():.1f}"))
    log(f"   avg NDC={rows[-1]['avg_ndc']:.0f} hit={rows[-1]['hit_rate']:.3f} "
        f"recall={rows[-1]['mean_recall']:.4f} calls={resC['calls'].mean():.1f}")

    # ---- D: ANNiE-S ----------------------------------------------------
    log("D. ANNiE: calibrating homing on 80 training queries")
    w, kp, tbar = annie.calibrate(
        g, train_q[:80], gt_train[:80], K, RT,
        cands=[(5, 5), (10, 10), (20, 10), (20, 20)], h=args.h, s=args.s,
        max_ndc=args.max_ndc)
    log(f"   (w={w}, k'={kp}) mean homing fraction={tbar:.3f}")
    t0 = time.time()
    Xa, ya, ratio, unreach = annie.collect(g, train_q, gt_train, K, RT, w, kp,
                                           args.h, args.s, args.max_ndc)
    log(f"   {len(Xa):,} cost samples in {time.time()-t0:.0f}s "
        f"(unreachable={unreach})")
    am = annie.ANNiE.train(Xa, ya, RT, args.p, w, kp, args.h, args.s)
    resD, secs = timed_run(annie.run_annie_s, g, test_q, gt_test, K, am,
                           args.max_ndc)
    cov = np.nanmean(resD["chat"] >= ndc_o)
    rows.append(summarize("ANNiE-S", resD["recall"], resD["ndc"], secs, RT,
                          note=f"p={args.p}, w={w}, k'={kp}, "
                               f"coverage={cov:.3f}"))
    log(f"   avg NDC={rows[-1]['avg_ndc']:.0f} hit={rows[-1]['hit_rate']:.3f} "
        f"recall={rows[-1]['mean_recall']:.4f} coverage={cov:.3f}")

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"{'method':<10}{'avg NDC':>10}{'med NDC':>10}{'p95 NDC':>10}"
          f"{'hit@Rt':>9}{'avg rec':>9}{'ms/q':>8}   note")
    print("-" * 96)
    for r in rows:
        print(f"{r['method']:<10}{r['avg_ndc']:>10.0f}{r['median_ndc']:>10.0f}"
              f"{r['p95_ndc']:>10.0f}{r['hit_rate']:>9.3f}"
              f"{r['mean_recall']:>9.4f}{r['ms_per_query']:>8.2f}   "
              f"{r.get('note','')}")
    print("=" * 96)

    base_row = next(r for r in rows if r["method"] == "Fixed-ef")
    for r in rows:
        r["speedup_vs_fixed"] = base_row["avg_ndc"] / r["avg_ndc"]

    csv_path = os.path.join(RESULT_DIR, f"e1_compare_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    log(f"saved {csv_path}")

    fig_path = plot(rows, tag, RT, K, nq)
    log(f"saved {fig_path}")


if __name__ == "__main__":
    main()
