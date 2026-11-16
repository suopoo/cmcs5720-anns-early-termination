"""Figures for Experiment 2 (per-query prediction errors)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core import paths
from anns_exp import plotting as P

EXP = paths.exp_dir("exp02")
DATA = os.path.join(EXP, "data")
ANA = os.path.join(EXP, "analysis")
FIG = os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)
K, RT = 100, 0.95


def load():
    pq = pd.read_csv(os.path.join(DATA, "per_query.csv"))
    ck = pd.read_csv(os.path.join(DATA, "per_checkpoint.csv"))
    ov = pd.read_csv(os.path.join(ANA, "failure_overlap.csv"))
    return pq, ck, ov


def method_errors(pq, ck):
    out = {}
    l = pq[pq.method == "LAET"].sort_values("query_id")
    a = pq[pq.method == "ANNiE-S"].sort_values("query_id")
    d = pq[pq.method == "DARTH"].sort_values("query_id")
    out["LAET_budget"] = (l["effective_budget_error"].values,
                          l["effective_budget"], l["oracle_ndc"])
    out["ANNiE_cost"] = (a["predicted_cost_raw"].values - a["oracle_ndc"].values,
                         a["predicted_cost_raw"], a["oracle_ndc"])
    # DARTH query-level mean signed recall error
    qm = ck.groupby("query_id")["recall_prediction_error"].mean()
    out["DARTH_recall"] = (qm.values, None, None)
    return out


def fig_errors(pq, ck):
    errs = method_errors(pq, ck)
    P.setup()
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 4.2))
    spec = [("LAET_budget", "A. LAET budget error (NDC)", "budget − oracle NDC"),
            ("ANNiE_cost", "B. ANNiE cost error (NDC)", "predicted − oracle NDC"),
            ("DARTH_recall", "C. DARTH recall error", "predicted − actual recall")]
    for ax, (key, title, xlabel) in zip(axes, spec):
        e = np.asarray(errs[key][0], float)
        e = e[np.isfinite(e)]
        ax.hist(e, bins=60, color=P.COLORS[0], alpha=0.85)
        ax.axvline(0, color=P.MUTED, lw=1)
        ax.axvline(np.mean(e), color=P.COLORS[1], lw=1.8, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel); ax.set_ylabel("Queries / checkpoints")
        P.clean(ax)
    fig.suptitle(f"Experiment 2 — prediction-error distributions (GIST1M, k={K}, "
                 f"R_t={RT}); dashed = mean", x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig1_error_distributions.png"),
                  "LAET/ANNiE: one value per test query. DARTH: query-level mean "
                  "over its checkpoints (repeated observations are not treated as "
                  "independent queries).")


def fig_cdf(pq, ck):
    errs = method_errors(pq, ck)
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.3))
    for i, key in enumerate(["LAET_budget", "ANNiE_cost"]):
        e = np.asarray(errs[key][0], float)
        mname = "LAET" if key == "LAET_budget" else "ANNiE-S"
        orc = pq[pq.method == mname].sort_values("query_id")["oracle_ndc"].values
        rel = e / np.maximum(np.asarray(orc, float), 1)
        P.ecdf(axes[0], rel, key.split("_")[0], P.COLORS[i], P.MARKERS[i])
        P.ecdf(axes[1], np.abs(rel), key.split("_")[0], P.COLORS[i],
               P.MARKERS[i])
    axes[0].axvline(0, color=P.MUTED, lw=1)
    axes[0].set_xlabel("relative signed cost error")
    axes[0].set_ylabel("CDF over queries")
    axes[0].set_title("A. Signed cost error", loc="left")
    axes[1].set_xlabel("relative absolute cost error")
    axes[1].set_title("B. Absolute cost error", loc="left")
    for ax in axes:
        P.clean(ax); ax.legend(loc="lower right")
    fig.suptitle("Experiment 2 — cost-predictor error CDFs", x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig2_error_cdf.png"),
                  "Relative to each query's oracle NDC; clipped at the plotted range.")


def fig_over_under(pq):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.3))
    labels, over, under, sev_over, sev_under = [], [], [], [], []
    for m, key in (("LAET", "effective_budget_error"),
                   ("ANNiE-S", "predicted_cost_raw")):
        sub = pq[pq.method == m]
        if m == "ANNiE-S":
            e = sub["predicted_cost_raw"] - sub["oracle_ndc"]
        else:
            e = sub[key]
        orc = sub["oracle_ndc"].values
        ok = np.isfinite(e.values) & np.isfinite(orc)
        e = e.values[ok]; orc = orc[ok]
        labels.append(m)
        over.append(float(np.mean(e > 0))); under.append(float(np.mean(e < 0)))
        sev_over.append(float(np.mean(e > 0.2 * np.maximum(orc, 1))))
        sev_under.append(float(np.mean(e < -0.2 * np.maximum(orc, 1))))
    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, over, 0.34, color=P.COLORS[0], label="overestimate")
    ax.bar(x + 0.18, under, 0.34, color=P.COLORS[1], label="underestimate")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of queries")
    ax.set_title("A. Direction of cost error", loc="left")
    P.clean(ax); ax.legend()
    ax = axes[1]
    ax.bar(x - 0.18, sev_over, 0.34, color=P.COLORS[0], label=">20% over")
    ax.bar(x + 0.18, sev_under, 0.34, color=P.COLORS[1], label=">20% under")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of queries")
    ax.set_title("B. Severe errors", loc="left")
    P.clean(ax); ax.legend()
    fig.suptitle("Experiment 2 — cost under/over-estimation rates", x=0.01,
                 ha="left")
    return P.save(fig, os.path.join(FIG, "fig3_over_under.png"),
                  "LAET uses its effective budget; ANNiE uses the predicted Chat. "
                  "Queries without an attainable oracle are excluded from the "
                  "relative severe-error count.")


def fig_scatter(pq):
    P.setup()
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 4.2))
    l = pq[pq.method == "LAET"]
    a = pq[pq.method == "ANNiE-S"]
    d = pq[pq.method == "DARTH"]
    ax = axes[0]
    ax.scatter(l["oracle_ndc"], l["effective_budget"], s=12, color=P.COLORS[0],
               alpha=0.6, edgecolors="none")
    lim = [1, max(l["oracle_ndc"].max(), l["effective_budget"].max()) * 1.1]
    ax.plot(lim, lim, color=P.MUTED, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("oracle NDC"); ax.set_ylabel("effective budget")
    ax.set_title("A. LAET", loc="left"); P.clean(ax, grid="both")
    ax = axes[1]
    ax.scatter(a["oracle_ndc"], a["predicted_cost_raw"], s=12,
               color=P.COLORS[1], alpha=0.6, edgecolors="none")
    lim = [1, max(a["oracle_ndc"].max(), a["predicted_cost_raw"].max()) * 1.1]
    ax.plot(lim, lim, color=P.MUTED, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("oracle NDC"); ax.set_ylabel("predicted cost")
    ax.set_title("B. ANNiE-S", loc="left"); P.clean(ax, grid="both")
    ax = axes[2]
    ax.scatter(d["actual_recall"], d["predicted_recall_at_stop"], s=12,
               color=P.COLORS[2], alpha=0.6, edgecolors="none")
    ax.plot([0.7, 1.0], [0.7, 1.0], color=P.MUTED, lw=1, ls="--")
    ax.axhline(RT, color=P.MUTED, lw=1, ls=(0, (2, 2)))
    ax.axvline(RT, color=P.MUTED, lw=1, ls=(0, (2, 2)))
    ax.set_xlabel("actual recall at stop"); ax.set_ylabel("predicted recall")
    ax.set_title("C. DARTH", loc="left"); P.clean(ax, grid="both")
    fig.suptitle("Experiment 2 — predicted vs true values", x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig4_predicted_vs_true.png"),
                  "Diagonal = perfect prediction. DARTH points only at the stop "
                  "decision; points below the horizontal line are false stops.")


def fig_extra(pq):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.3))
    for i, m in enumerate(["LAET", "ANNiE-S", "DARTH"]):
        s = pq[(pq.method == m) & pq.target_hit.astype(bool)]
        P.ecdf(axes[0], s["relative_extra_ndc"].dropna(), m, P.COLORS[i],
               P.MARKERS[i])
    axes[0].set_xlabel("relative extra NDC (successful queries)")
    axes[0].set_ylabel("CDF over queries")
    axes[0].set_title("A. Extra NDC distribution", loc="left")
    axes[0].set_xlim(left=-0.01)
    P.clean(axes[0]); axes[0].legend(loc="lower right")
    methods = ["LAET", "DARTH", "ANNiE-S"]
    fail = [float(np.mean(~pq[pq.method == m]["target_hit"].astype(bool)))
            for m in methods]
    axes[1].bar(np.arange(3), fail, color=P.COLORS[:3], width=0.62)
    for i, v in enumerate(fail):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(np.arange(3)); axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Target failure rate")
    axes[1].set_title("B. Recall-target failures", loc="left")
    P.clean(axes[1])
    fig.suptitle(f"Experiment 2 — success cost and failure rate (R_t={RT})",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig5_extra_ndc_failures.png"),
                  "Relative extra NDC = max(0, stop − oracle) / oracle; only "
                  "queries that reached the target contribute. Failures are "
                  "counted separately and never folded into the zero-extra bin.")


def fig_error_vs_cost(pq, ck):
    P.setup()
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 4.2))
    l = pq[pq.method == "LAET"]
    a = pq[pq.method == "ANNiE-S"]
    ck2 = ck.copy()
    ck2["abs"] = np.abs(ck2["recall_prediction_error"])
    qm = ck2.groupby("query_id").agg(abs_err=("abs", "mean"),
                                     oracle=("ndc_at_checkpoint", "max"))
    ax = axes[0]
    ax.scatter(l["oracle_ndc"], np.abs(l["effective_budget_error"]), s=10,
               color=P.COLORS[0], alpha=0.55, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("symlog")
    ax.set_xlabel("oracle NDC"); ax.set_ylabel("|budget error| (NDC)")
    ax.set_title("A. LAET", loc="left"); P.clean(ax, grid="both")
    ax = axes[1]
    ax.scatter(a["oracle_ndc"], np.abs(a["predicted_cost_raw"] - a["oracle_ndc"]),
               s=10, color=P.COLORS[1], alpha=0.55, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("symlog")
    ax.set_xlabel("oracle NDC"); ax.set_ylabel("|cost error| (NDC)")
    ax.set_title("B. ANNiE-S", loc="left"); P.clean(ax, grid="both")
    ax = axes[2]
    ax.scatter(qm["oracle"], qm["abs_err"], s=10, color=P.COLORS[2],
               alpha=0.55, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("final NDC"); ax.set_ylabel("mean |recall error|")
    ax.set_title("C. DARTH", loc="left"); P.clean(ax, grid="both")
    fig.suptitle("Experiment 2 — error magnitude vs search cost", x=0.01,
                 ha="left")
    return P.save(fig, os.path.join(FIG, "fig6_error_vs_oracle_cost.png"),
                  "Each point is one query (DARTH: query-level mean error vs its "
                  "last checkpoint).")


def fig_overlap(pq, ck, ov):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, ftype, title in ((axes[0], "quality_fail", "A. Quality failures"),
                             (axes[1], "efficiency_fail", "B. Efficiency failures")):
        s = ov[ov.failure_type == ftype]
        if s.empty:
            continue
        pair = s[s.A.isin(["LAET", "DARTH", "ANNiE-S"]) &
                 s.B.isin(["LAET", "DARTH", "ANNiE-S"])]
        labels = [f"{r.A} & {r.B}" for r in pair.itertuples()]
        vals = pair["intersection"].values
        ax.bar(np.arange(len(labels)), vals, color=P.COLORS[0], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{int(v)}", ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Queries in both failure sets")
        ax.set_title(title, loc="left")
        P.clean(ax)
    fig.suptitle("Experiment 2 — failure-set overlaps on identical test queries",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig7_failure_overlap.png"),
                  "Efficiency failure = successful query whose relative extra NDC "
                  "is in the method's worst decile.")


def fig_jaccard(ov):
    methods = ["LAET", "DARTH", "ANNiE-S"]
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ftype, title in ((axes[0], "quality_fail", "A. Quality failure"),
                             (axes[1], "efficiency_fail", "B. Efficiency failure")):
        M = np.full((3, 3), np.nan)
        np.fill_diagonal(M, 1.0)
        s = ov[ov.failure_type == ftype]
        for r in s.itertuples():
            if r.A in methods and r.B in methods:
                i, j = methods.index(r.A), methods.index(r.B)
                if np.isfinite(r.jaccard):
                    M[i, j] = M[j, i] = r.jaccard
        im = ax.imshow(M, cmap=matplotlib_cmap(), vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(methods, fontsize=8); ax.set_yticklabels(methods, fontsize=8)
        for i in range(3):
            for j in range(3):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, color=P.INK)
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    fig.suptitle("Experiment 2 — pairwise Jaccard similarity of failure sets",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig8_jaccard_heatmap.png"),
                  "Diagonal is 1 by definition. Values are computed on the same "
                  "1000 query IDs.")


def matplotlib_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seqblue", P.SEQ)


def main():
    pq, ck, ov = load()
    outs = [fig_errors(pq, ck), fig_cdf(pq, ck), fig_over_under(pq),
            fig_scatter(pq), fig_extra(pq), fig_error_vs_cost(pq, ck),
            fig_overlap(pq, ck, ov), fig_jaccard(ov)]
    print("\n".join(outs))


if __name__ == "__main__":
    main()
