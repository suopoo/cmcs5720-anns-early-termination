"""Figures for Experiment 3 (asymmetric losses and prediction timing)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anns_exp.core import paths
from anns_exp import plotting as P

EXP = paths.exp_dir("exp03")
DATA = os.path.join(EXP, "data")
FIG = os.path.join(EXP, "figures")
os.makedirs(FIG, exist_ok=True)
K, RT = 10, 0.90


def fig_laet_errors(pred):
    P.setup()
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 4.2))
    s = pred[pred.experiment == "3A-LAET"]
    for i, cfg in enumerate(["L0", "L1", "L2", "L3", "L4"]):
        d = s[s.config == cfg]
        e = np.log2(np.maximum(d["pred_ndc"], 1)) - np.log2(
            np.maximum(d["true_c1_ndc"], 1))
        P.ecdf(axes[0], e, cfg, P.COLORS[i], P.MARKERS[i])
        axes[1].hist(e, bins=50, histtype="step", lw=1.8, color=P.COLORS[i],
                     label=cfg)
        over = np.mean(e > 0); under = np.mean(e < 0)
        axes[2].bar(i - 0.18, over, 0.34, color=P.COLORS[i])
        axes[2].bar(i + 0.18, under, 0.34, color=P.COLORS[i], alpha=0.5)
    axes[0].axvline(0, color=P.MUTED, lw=1)
    axes[0].set_xlabel("log2 predicted − log2 true C1")
    axes[0].set_ylabel("CDF"); axes[0].set_title("A. Signed cost error CDF", loc="left")
    P.clean(axes[0]); axes[0].legend(fontsize=7)
    axes[1].axvline(0, color=P.MUTED, lw=1)
    axes[1].set_xlabel("log2 error"); axes[1].set_ylabel("Queries")
    axes[1].set_title("B. Error histograms", loc="left")
    P.clean(axes[1]); axes[1].legend(fontsize=7)
    axes[2].set_xticks(range(5)); axes[2].set_xticklabels(["L0", "L1", "L2", "L3", "L4"])
    axes[2].set_ylabel("Fraction")
    axes[2].set_title("C. Over (solid) / under (faded)", loc="left")
    P.clean(axes[2])
    fig.suptitle("Experiment 3A — LAET cost-prediction errors by training loss "
                 f"(k={K}, R_t={RT})", x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig1_laet_loss_errors.png"),
                  "L0 = original L2; L1-L4 = quantile tau 0.50/0.75/0.90/0.95. "
                  "Errors are in log2 NDC, the model's native label space.")


def fig_laet_curves(curves):
    P.setup()
    fig, ax = P.plt.subplots(figsize=(6.4, 4.4))
    s = curves[curves.method == "LAET"]
    for i, cfg in enumerate(["L0", "L1", "L2", "L3", "L4"]):
        d = s[s.config == cfg].sort_values("mean_ndc")
        ax.plot(d["mean_ndc"], d["mean_recall"], color=P.COLORS[i], lw=2,
                marker=P.MARKERS[i], ms=4, markevery=max(1, len(d) // 8),
                label=cfg)
    ax.axhline(RT, color=P.MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Mean NDC per query"); ax.set_ylabel(f"Mean recall@{K}")
    ax.set_title("Experiment 3A — LAET recall–cost trade-off", loc="left")
    P.clean(ax); ax.legend(loc="lower right")
    return P.save(fig, os.path.join(FIG, "fig2_laet_recall_ndc.png"),
                  "Each curve sweeps the LAET multiplier for one trained loss. "
                  "The operating point used in the A2 comparison corresponds to "
                  "the validation-calibrated multiplier.")


def fig_darth_errors(off):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.3))
    for i, cfg in enumerate(["D0", "D1", "D2", "D3", "D4"]):
        d = off[off.config == cfg]
        P.ecdf(axes[0], d["error"], cfg, P.COLORS[i], P.MARKERS[i])
        axes[1].hist(d["error"], bins=60, histtype="step", lw=1.8,
                     color=P.COLORS[i], label=cfg)
    axes[0].axvline(0, color=P.MUTED, lw=1)
    axes[0].set_xlabel("predicted recall − actual recall")
    axes[0].set_ylabel("CDF over checkpoints")
    axes[0].set_title("A. Signed error CDF", loc="left")
    P.clean(axes[0]); axes[0].legend(fontsize=7)
    axes[1].axvline(0, color=P.MUTED, lw=1)
    axes[1].set_xlabel("recall error"); axes[1].set_ylabel("Checkpoints")
    axes[1].set_title("B. Error histograms", loc="left")
    P.clean(axes[1]); axes[1].legend(fontsize=7)
    fig.suptitle("Experiment 3A — DARTH recall-prediction errors by loss",
                 x=0.01, ha="left")
    return P.save(fig, os.path.join(FIG, "fig3_darth_loss_errors.png"),
                  "All models are evaluated at the identical checkpoint grid, so "
                  "the comparison is offline and independent of search dynamics. "
                  "D0 = original L2; D1-D4 = quantile tau 0.50/0.25/0.10/0.05.")


def fig_darth_e2e(pq):
    P.setup()
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 4.2))
    s = pq[pq.experiment == "3A-DARTH"]
    cfgs = ["D0", "D1", "D2", "D3", "D4"]
    hit = [float(np.mean(s[s.config == c]["actual_recall"] >= RT - 1e-9)) for c in cfgs]
    ndc = [float(np.nanmean(s[s.config == c]["stop_ndc"])) for c in cfgs]
    calls = [float(np.nanmean(s[s.config == c]["n_predictions"])) for c in cfgs]
    for ax, vals, ylab, title in (
            (axes[0], hit, "Target hit rate", "A. Quality"),
            (axes[1], ndc, "Mean stop NDC", "B. Search cost"),
            (axes[2], calls, "Mean predictor calls", "C. Predictor overhead")):
        ax.bar(np.arange(5), vals, color=P.COLORS[:5], width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.3f}" if "rate" in ylab else f"{v:.1f}",
                    ha="center", fontsize=8)
        ax.set_xticks(np.arange(5)); ax.set_xticklabels(cfgs)
        ax.set_ylabel(ylab); ax.set_title(title, loc="left"); P.clean(ax)
    fig.suptitle("Experiment 3A — DARTH end-to-end behaviour by loss", x=0.01,
                 ha="left")
    return P.save(fig, os.path.join(FIG, "fig4_darth_end_to_end.png"),
                  "Same ipi/mpi for all losses; stopping positions and call counts "
                  "differ because the predicted recall differs.")


def fig_sensitivity(m3b):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(11.5, 4.4))
    s = m3b[m3b.experiment == "3B-LAET"]
    for i, loss in enumerate(s["loss"].unique()):
        d = s[s.loss == loss].sort_values("f")
        axes[0].plot(d["f"], d["hit_rate"], color=P.COLORS[i], lw=2,
                     marker=P.MARKERS[i], ms=5, label=loss)
        axes[0].plot(d["f"], d["mean_recall"], color=P.COLORS[i], lw=1.2,
                     ls="--", alpha=0.7)
    axes[0].axhline(RT, color=P.MUTED, lw=1, ls=(0, (4, 3)))
    axes[0].set_xlabel("Preliminary search NDC f")
    axes[0].set_ylabel("Hit rate (solid) / mean recall (dashed)")
    axes[0].set_title("A. LAET: f × loss", loc="left")
    P.clean(axes[0]); axes[0].legend(fontsize=7)
    d = m3b[m3b.experiment == "3B-DARTH"]
    for i, loss in enumerate(d["loss"].unique()):
        dd = d[d.loss == loss]
        axes[1].plot(np.arange(len(dd)), dd["hit_rate"], color=P.COLORS[i],
                     lw=2, marker=P.MARKERS[i], ms=5, label=loss)
        axes[1].plot(np.arange(len(dd)), dd["mean_recall"], color=P.COLORS[i],
                     lw=1.2, ls="--", alpha=0.7)
    axes[1].set_xticks(np.arange(len(d["interval"].unique())))
    axes[1].set_xticklabels(sorted(d["interval"].unique()))
    axes[1].set_xlabel("Prediction-interval configuration")
    axes[1].set_ylabel("Hit rate (solid) / mean recall (dashed)")
    axes[1].set_title("B. DARTH: interval × loss", loc="left")
    P.clean(axes[1]); axes[1].legend(fontsize=7)
    fig.suptitle("Experiment 3B — prediction timing sensitivity", x=0.01,
                 ha="left")
    return P.save(fig, os.path.join(FIG, "fig5_timing_sensitivity.png"),
                  "LAET f values: 630/945/1260/1575 (0.5x/0.75x/1x/1.25x of the "
                  "paper's GIST value). DARTH P0-P3 are defined in the report.")


def main():
    pred = pd.read_csv(os.path.join(DATA, "prediction_traces_3A.csv"))
    curves = pd.read_csv(os.path.join(DATA, "quality_cost_curves_3A.csv"))
    off = pd.read_csv(os.path.join(DATA, "offline_predictions_3A_darth.csv"))
    pq = pd.read_csv(os.path.join(DATA, "per_query_3A.csv"))
    m3b = pd.read_csv(os.path.join(DATA, "metrics_3B.csv"))
    outs = [fig_laet_errors(pred), fig_laet_curves(curves),
            fig_darth_errors(off), fig_darth_e2e(pq), fig_sensitivity(m3b)]
    print("\n".join(outs))


if __name__ == "__main__":
    main()
