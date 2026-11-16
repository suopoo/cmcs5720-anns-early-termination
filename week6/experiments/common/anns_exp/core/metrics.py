"""Evaluation metrics used by the three papers (same names, same semantics).

All functions take per-query arrays and return scalars, except the
distribution helpers which return arrays for plotting.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# search quality
# --------------------------------------------------------------------------
def recall_at_k(retrieved, gt, k=None):
    """|A_k ∩ N_k| / k, averaged over queries.  `gt` may be wider than k."""
    if k is None:
        k = np.shape(retrieved)[1]
    r = [len(set(retrieved[i][:k]) & set(gt[i][:k])) / k for i in range(len(retrieved))]
    return float(np.mean(r)), np.asarray(r, dtype=np.float32)


def rqut(per_query_recall, target):
    """Ratio of Queries Under Target (DARTH): fraction with recall < R_t."""
    return float(np.mean(np.asarray(per_query_recall) < target))


def p99_error(per_query_recall, target):
    """99th percentile of |R_t - recall_q|."""
    return float(np.percentile(np.abs(np.asarray(per_query_recall) - target), 99))


def worst1_error(per_query_recall, target):
    """Mean error over the worst 1% of queries."""
    e = np.sort(np.abs(np.asarray(per_query_recall) - target))[::-1]
    n = max(1, int(round(0.01 * len(e))))
    return float(e[:n].mean())


def rde(retrieved_dist, gt_dist, k=None):
    """Relative Distance Error (Patella & Ciaccia; DARTH's get_RDE).

    RDE = mean_i (d(q, an_i) - d(q, nn_i)) / d(q, nn_i)
    Lower is better; 0 means the retrieved distances match the true ones.
    """
    if k is None:
        k = np.shape(retrieved_dist)[1]
    a = np.asarray(retrieved_dist)[:, :k]
    g = np.asarray(gt_dist)[:, :k]
    return float(np.mean((a - g) / np.maximum(g, 1e-12)))


def nrs(retrieved, gt, k=None):
    """Normalized Rank Sum (DARTH's get_NRS).

    For each returned id, its rank in the ground-truth list (max rank if
    absent), normalized by k * |GT list|.  Lower is better.
    """
    ret = np.asarray(retrieved)
    if k is None:
        k = ret.shape[1]
    m = np.asarray(gt).shape[1]
    vals = []
    for i in range(len(ret)):
        pos = {int(g): r for r, g in enumerate(gt[i])}
        vals.append(sum(pos.get(int(x), m) for x in ret[i][:k]) / (k * m))
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# cost estimation (ANNiE)
# --------------------------------------------------------------------------
def mape(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true) / np.maximum(true, 1e-9)))


def qerror(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.maximum(pred / np.maximum(true, 1e-9),
                                   true / np.maximum(pred, 1e-9))))


def pmape(pred, true):
    """ANNiE's pMAPE: error on over-estimated queries only (under-estimation
    is treated as a failed guarantee, not as error)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    over = pred > true
    if not over.any():
        return 0.0
    return float(np.mean(np.abs(pred[over] - true[over]) / np.maximum(true[over], 1e-9)))


def coverage(pred, true):
    """Empirical Pr[pred >= C] -- should be >= p when using quantile loss."""
    return float(np.mean(np.asarray(pred, float) >= np.asarray(true, float)))


# --------------------------------------------------------------------------
# regression diagnostics
# --------------------------------------------------------------------------
def regression_report(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mape": mape(pred, true),
        "qerror": qerror(pred, true),
    }
