"""Figures for Experiment 1 (feature transfer)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core import paths
from anns_exp import plotting as P

EXP = paths.exp_dir("exp01")
DATA = os.path.join(EXP, "data")
ANA = os.path.join(EXP, "analysis")
FIG = os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)
K, RT, F0 = 10, 0.9, 1260
TAG = ""


def namemap(group):
    if group in ("L0", "L1"):
        base = ["d_start", "d_1st", "d_10th", "r1st", "r10th"]
        if group == "L1":
            base += ["ndc", "step", "inserts", "furthest", "avg", "var",
                     "median", "p25", "p75", "rt"]
            base += ["avg_cc"] + [f"bsf_cc{i}" for i in range(K)] + \
                    [f"reach{h}" for h in range(4)]
    else:
        base = ["ndc", "step", "inserts", "first", "closest", "furthest",
                "avg", "var", "median", "p25", "p75", "rt"]
        base += ["avg_cc"] + [f"bsf_cc{i}" for i in range(K)] + \
                [f"reach{h}" for h in range(4)]
    return lambda i: (f"q[{i}]" if i < 960 else base[i - 960])


def fig1(summary, ckdf):
    P.setup()
    fig, axes = P.plt.subplots(2, 2, figsize=(11, 7.2))
    laet = summary[summary.method == "LAET"].set_index("group")
    groups = ["L0", "L1", "L2"]
    x = np.arange(len(groups))
    ax = axes[0, 0]
    vals = [laet.loc[g, "native_mae"] for g in groups]
    ax.bar(x, vals, color=P.COLORS[:3], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("MAE (NDC)"); ax.set_title("A. LAET cost error (native NDC)", loc="left")
    P.clean(ax)
    ax = axes[0, 1]
    vals = [laet.loc[g, "native_mape"] for g in groups]
    ax.bar(x, vals, color=P.COLORS[:3], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("MAPE"); ax.set_title("B. LAET relative cost error", loc="left")
    P.clean(ax)
    d = ckdf.copy()
    d["abs_err"] = np.abs(d["recall_prediction_error"])
    gstats = d.groupby("group")["abs_err"].agg(["mean", "median"])
    mse = d.groupby("group")["recall_prediction_error"].apply(
        lambda s: float(np.mean(np.asarray(s) ** 2)))
    dg = ["D0", "D1", "D2"]
    ax = axes[1, 0]
    vals = [gstats.loc[g, "mean"] for g in dg]
    ax.bar(np.arange(3), vals, color=P.COLORS[:3], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(dg)
    ax.set_ylabel("MAE (recall)"); ax.set_title("C. DARTH checkpoint recall error", loc="left")
    P.clean(ax)
    ax = axes[1, 1]
    vals = [mse[g] for g in dg]
    ax.bar(np.arange(3), vals, color=P.COLORS[:3], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(dg)
    ax.set_ylabel("MSE (recall)"); ax.set_title("D. DARTH checkpoint MSE", loc="left")
    P.clean(ax)
    fig.suptitle("Experiment 1 — prediction accuracy of the six transfer groups "
                 f"(GIST1M, k={K}, R_t={RT}, checkpoint={F0})", x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig1_prediction_accuracy.png"),
                  "Bars are group means over the 1000 test queries; DARTH errors "
                  "are per checkpoint on the shared fixed schedule.")


def fig2(curves):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, method, groups, title in (
            (axes[0], "LAET", ["L0", "L1", "L2"], "A. LAET (multiplier sweep)"),
            (axes[1], "DARTH", ["D0", "D1", "D2"], "B. DARTH (threshold sweep)")):
        sub = curves[curves.method == method]
        for i, g in enumerate(groups):
            s = sub[sub.group == g].sort_values("mean_ndc")
            ax.plot(s["mean_ndc"], s["mean_recall"], color=P.COLORS[i],
                    lw=2, marker=P.MARKERS[i], ms=4,
                    markevery=max(1, len(s) // 8), label=g)
            ax.annotate(g, (s["mean_ndc"].iloc[-1], s["mean_recall"].iloc[-1]),
                        textcoords="offset points", xytext=(6, -3), fontsize=8,
                        color=P.INK)
        ax.axhline(RT, color=P.MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xlabel("Mean NDC per query")
        ax.set_ylabel(f"Mean recall@{K}")
        ax.set_title(title, loc="left")
        P.clean(ax)
        ax.legend(loc="lower right")
    fig.suptitle("Experiment 1 — recall–cost trade-off (same test queries)",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig2_search_quality_cost.png"),
                  "LAET curves sweep the multiplier around each model's operating "
                  "point; DARTH curves sweep the predicted-recall stop threshold. "
                  "Dashed line = target recall.")


def fig3(pred, ckdf):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for i, g in enumerate(["L0", "L1", "L2"]):
        s = pred[pred.group == g]
        e = np.log2(np.maximum(s["pred_ndc"], 1)) - np.log2(
            np.maximum(s["true_c1_ndc"], 1))
        P.ecdf(ax, e, g, P.COLORS[i], P.MARKERS[i])
    ax.axvline(0, color=P.MUTED, lw=1)
    ax.set_xlabel("log2(predicted NDC) − log2(true C1)")
    ax.set_ylabel("CDF over test queries")
    ax.set_title("A. LAET signed cost error", loc="left")
    P.clean(ax); ax.legend(loc="lower right")
    ax = axes[1]
    for i, g in enumerate(["D0", "D1", "D2"]):
        s = ckdf[ckdf.group == g]
        P.ecdf(ax, s["recall_prediction_error"], g, P.COLORS[i], P.MARKERS[i])
    ax.axvline(0, color=P.MUTED, lw=1)
    ax.set_xlabel("predicted recall − actual recall")
    ax.set_ylabel("CDF over checkpoints")
    ax.set_title("B. DARTH signed recall error", loc="left")
    P.clean(ax); ax.legend(loc="lower right")
    fig.suptitle("Experiment 1 — per-query / per-checkpoint signed errors",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig3_per_query_error.png"),
                  "LAET: 1000 test queries. DARTH: all fixed-schedule checkpoints "
                  "on those queries (repeated observations, not independent queries).")


def fig4(pred, ckdf):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.4))
    piv = pred.pivot_table(index="query_id", columns="group",
                           values=["pred_ndc", "true_c1_ndc"])
    abs_err = {}
    for g in ["L0", "L1", "L2"]:
        abs_err[g] = np.abs(np.log2(np.maximum(piv[("pred_ndc", g)], 1)) -
                            np.log2(np.maximum(piv[("true_c1_ndc", g)], 1)))
    ax = axes[0]
    for i, (x, y, lab) in enumerate((("L1", "L0", "L1 − L0"),
                                     ("L2", "L0", "L2 − L0"))):
        d = (abs_err[x] - abs_err[y]).dropna()
        P.ecdf(ax, d, lab, P.COLORS[i], P.MARKERS[i])
    ax.axvline(0, color=P.MUTED, lw=1)
    ax.set_xlabel("Δ absolute log2 cost error (negative = better)")
    ax.set_ylabel("CDF over queries")
    ax.set_title("A. LAET feature transfer", loc="left")
    P.clean(ax); ax.legend(loc="lower right")
    ck = ckdf.copy()
    ck["abs_err"] = np.abs(ck["recall_prediction_error"])
    q = ck.groupby(["query_id", "group"])["abs_err"].mean().unstack()
    ax = axes[1]
    for i, (x, lab) in enumerate((("D1", "D1 − D0"), ("D2", "D2 − D0"))):
        d = (q[x] - q["D0"]).dropna()
        P.ecdf(ax, d, lab, P.COLORS[i], P.MARKERS[i])
    ax.axvline(0, color=P.MUTED, lw=1)
    ax.set_xlabel("Δ mean absolute recall error (negative = better)")
    ax.set_ylabel("CDF over queries")
    ax.set_title("B. DARTH feature transfer", loc="left")
    P.clean(ax); ax.legend(loc="lower right")
    fig.suptitle("Experiment 1 — per-query effect of adding ANNiE features",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig4_feature_transfer_improvement.png"),
                  "Paired over the same Query IDs; DARTH values are first averaged "
                  "per query across its checkpoints before differencing.")


def fig5():
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 5))
    for ax, group, title in ((axes[0], "L1", "A. LAET + ANNiE (L1)"),
                             (axes[1], "D1", "B. DARTH + ANNiE (D1)")):
        path = os.path.join(ANA, f"importance_{'laet' if group == 'L1' else 'darth'}_{group}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).head(14).iloc[::-1]
        nm = namemap(group)
        labels = [nm(int(i)) for i in df["feature"].str[1:].astype(int)]
        ax.barh(np.arange(len(df)), df["gain"], color=P.COLORS[0], height=0.62)
        ax.set_yticks(np.arange(len(df))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("LightGBM gain")
        ax.set_title(title, loc="left")
        P.clean(ax, grid="x")
    fig.suptitle("Experiment 1 — top features after transfer "
                 "(q[·] = raw query dimensions)", x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig5_feature_importance.png"),
                  "Gain importance from the transferred models. Graph features "
                  "(reachability / clustering) carry large gain; individual query "
                  "dimensions are shown as q[·].")


def fig8():
    from anns_exp.exp_common import load_trace
    from anns_exp.trace import load_split
    import sys as _s
    _s.path.insert(0, HERE)
    ltr = load_trace("gist_k10_laet_test" + TAG)
    dtr = load_trace("gist_k10_darth_test" + TAG)
    perq = pd.read_csv(os.path.join(DATA, "per_query_results.csv"))
    laet = perq[perq.method == "LAET"]
    piv = laet.pivot_table(index="query_id", columns="group", values="actual_recall")
    dn = laet.pivot_table(index="query_id", columns="group", values="stop_ndc")
    abs_err = laet.pivot_table(index="query_id", columns="group",
                               values="native_prediction_error").abs()
    improve = (abs_err["L0"] - abs_err["L1"]).sort_values(ascending=False)
    degrade = (abs_err["L1"] - abs_err["L0"]).sort_values(ascending=False)
    cases = []
    cases.append(("L1 improves most", int(improve.index[0]), "laet"))
    cases.append(("L1 degrades most", int(degrade.index[0]), "laet"))
    fail = piv[(piv["L0"] < RT) & (piv["L1"] < RT)].index
    if len(fail):
        cases.append(("all LAET groups fail", int(fail[0]), "laet"))
    d_perq = perq[perq.method == "DARTH"]
    dfail = d_perq[~d_perq.target_hit.astype(bool)]
    if len(dfail):
        cases.append(("DARTH quality failure", int(dfail.iloc[0].query_id), "darth"))
    dsucc = d_perq[d_perq.target_hit.astype(bool)].sort_values(
        "relative_extra_ndc", ascending=False)
    if len(dsucc):
        cases.append(("DARTH expensive success", int(dsucc.iloc[0].query_id), "darth"))
    P.setup()
    fig, axes = P.plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.6))
    if len(cases) == 1:
        axes = [axes]
    for ax, (title, qi, kind) in zip(axes, cases):
        if kind == "laet":
            tr = ltr
        else:
            tr = dtr
        s, e = int(tr["ptr"][qi]), int(tr["ptr"][qi + 1])
        ndc = tr["step_ndc"][s:e]
        rec = tr["step_recall"][s:e]
        ax.plot(ndc, rec, color=P.MUTED, lw=1.4)
        if kind == "laet":
            for j, g in enumerate(["L0", "L1", "L2"]):
                ax.axvline(dn.loc[qi, g], color=P.COLORS[j], lw=1.5, ls="--",
                           label=g)
        else:
            for j, g in enumerate(["D0", "D1", "D2"]):
                rr = perq[(perq.method == "DARTH") & (perq.group == g) &
                          (perq.query_id == qi)]
                if len(rr):
                    ax.axvline(float(rr.iloc[0].stop_ndc), color=P.COLORS[j],
                               lw=1.5, ls="--", label=g)
        ax.axhline(RT, color=P.MUTED, lw=1, ls=(0, (2, 2)))
        ax.set_title(f"{title}\nquery {qi}", fontsize=9, loc="left")
        ax.set_xlabel("NDC"); ax.set_ylabel(f"recall@{K}")
        P.clean(ax)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Experiment 1 — representative search trajectories",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig8_case_study.png"),
                  "Grey curve = the query's full search trajectory; dashed "
                  "verticals = each group's stop position.")




def fig6():
    p = os.path.join(DATA, "checkpoint_sensitivity.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.3))
    for i, method in enumerate(["LAET", "DARTH"]):
        s = d[d.method == method].sort_values("f")
        if s.empty:
            continue
        axes[0].plot(s["f"], s["hit_rate"], color=P.COLORS[i], lw=2,
                     marker=P.MARKERS[i], ms=5, label=method)
        axes[0].plot(s["f"], s["mean_recall"], color=P.COLORS[i], lw=1.2,
                     ls="--", alpha=0.7)
        axes[1].plot(s["f"], s["avg_ndc"], color=P.COLORS[i], lw=2,
                     marker=P.MARKERS[i], ms=5, label=method)
    axes[0].axhline(RT, color=P.MUTED, lw=1, ls=(0, (4, 3)))
    axes[0].set_xlabel("Checkpoint f (NDC)")
    axes[0].set_ylabel("Hit rate (solid) / mean recall (dashed)")
    axes[0].set_title("A. Quality vs preliminary-search depth", loc="left")
    P.clean(axes[0]); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Checkpoint f (NDC)"); axes[1].set_ylabel("Mean NDC")
    axes[1].set_title("B. Search cost", loc="left")
    P.clean(axes[1]); axes[1].legend(fontsize=7)
    fig.suptitle("Experiment 1 — preliminary-search sensitivity "
                 "(original-feature models)", x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig6_checkpoint_sensitivity.png"),
                  "L0/D0 retrained at each f on the corresponding state "
                  "features. L1/L2 and D1/D2 are only evaluated at f=1260; "
                  "ANNiE suffix features were not cached at other depths.")


def fig7():
    p = os.path.join(ANA, "latency_summary.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    P.setup()
    fig, ax = P.plt.subplots(figsize=(7.2, 4.4))
    labels = [f"{r.method}-{r.group}" for r in d.itertuples()]
    feat = d["feature_ms"].values
    pred = d["predict_ms"].values
    other = np.maximum(d["mean_ms"].values - feat - pred, 0)
    y = np.arange(len(d))[::-1]
    ax.barh(y, other, color=P.COLORS[0], label="search / other", height=0.62)
    ax.barh(y, feat, left=other, color=P.COLORS[1], label="feature extraction",
            height=0.62)
    ax.barh(y, pred, left=other + feat, color=P.COLORS[2], label="model inference",
            height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean end-to-end latency (ms/query, 30 test queries)")
    ax.set_title("Experiment 1 — measured latency breakdown", loc="left")
    P.clean(ax, grid="x"); ax.legend(fontsize=8)
    return P.save(fig, os.path.join(FIG, "fig7_latency_overhead.png"),
                  "Same hardware and process for all groups; NDC is the portable "
                  "cost, latency is Python-implementation specific.")


def main():
    global TAG
    if "--tag" in sys.argv:
        TAG = sys.argv[sys.argv.index("--tag") + 1]
    summary = pd.read_csv(os.path.join(DATA, "summary_metrics.csv"))
    curves = pd.read_csv(os.path.join(DATA, "quality_cost_curves.csv"))
    pred = pd.read_csv(os.path.join(DATA, "prediction_results.csv"))
    ckdf = pd.read_csv(os.path.join(DATA, "darth_checkpoint_predictions.csv"))
    outs = [fig1(summary, ckdf), fig2(curves), fig3(pred, ckdf),
            fig4(pred, ckdf), fig5(), fig6(), fig7(), fig8()]
    outs = [o for o in outs if o]
    print("\n".join(outs))


if __name__ == "__main__":
    main()
