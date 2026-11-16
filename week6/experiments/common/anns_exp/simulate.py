"""Exact replay of stopping rules on pre-recorded search traces.

The learned methods only choose when to stop; the underlying HNSW expansion
order is independent of the model.  Replaying on a trace therefore produces
exactly the online NDC / recall, while allowing hundreds of model variants
(multipliers, quantiles, intervals, training sizes) to be evaluated without
re-running the graph search.
"""
from __future__ import annotations

import numpy as np

from .features import assemble
from .gbdt import predict


def _seg(ptr, arr, i):
    return arr[ptr[i]:ptr[i + 1]]


def laet_from_trace(tr, X, model, f, mult, pred_max, name=""):
    """One prediction at NDC>=f; stop at B=max(f,min(2^p*mult,pred_max))."""
    n = len(tr["ptr"]) - 1
    p = predict(model, X)
    B = np.maximum(float(f), np.minimum(
        np.power(2.0, np.maximum(p, 0.0)) * float(mult), float(pred_max)))
    stop_ndc = np.full(n, np.nan)
    recall = np.full(n, np.nan)
    truncated = np.zeros(n, dtype=bool)
    for i in range(n):
        seg = _seg(tr["ptr"], tr["step_ndc"], i)
        if len(seg) == 0:
            continue
        j = int(np.searchsorted(seg, B[i], side="left"))
        if j >= len(seg):
            j = len(seg) - 1
            if seg[-1] and tr["step_recall"][tr["ptr"][i] + j] >= 1.0 - 1e-9:
                # trace stopped at recall 1.0; the method still pays its budget
                stop_ndc[i] = B[i]
                recall[i] = 1.0
                continue
            truncated[i] = B[i] > seg[-1] + 1e-9
        stop_ndc[i] = seg[j]
        recall[i] = tr["step_recall"][tr["ptr"][i] + j]
    return {"pred": p, "budget": B, "stop_ndc": stop_ndc,
            "recall": recall, "truncated": truncated, "name": name}


def annie_from_trace(tr, X, model, lower_ndc, name=""):
    """Stop at max(predicted cost, homing checkpoint)."""
    n = len(tr["ptr"]) - 1
    pred = np.maximum(predict(model, X), 0.0)
    target = np.maximum(pred, np.asarray(lower_ndc, dtype=np.float64))
    stop_ndc = np.full(n, np.nan)
    recall = np.full(n, np.nan)
    truncated = np.zeros(n, dtype=bool)
    for i in range(n):
        seg = _seg(tr["ptr"], tr["step_ndc"], i)
        if len(seg) == 0:
            continue
        j = int(np.searchsorted(seg, target[i], side="left"))
        if j >= len(seg):
            j = len(seg) - 1
            if seg[-1] and tr["step_recall"][tr["ptr"][i] + j] >= 1.0 - 1e-9:
                stop_ndc[i] = target[i]
                recall[i] = 1.0
                continue
            truncated[i] = target[i] > seg[-1] + 1e-9
        stop_ndc[i] = seg[j]
        recall[i] = tr["step_recall"][tr["ptr"][i] + j]
    return {"pred": pred, "budget": target, "stop_ndc": stop_ndc,
            "recall": recall, "truncated": truncated, "name": name}


def darth_fixed_from_grid(tr, X_grid, model, rt, grid_ptr=None,
                          name="", min_ndc=0):
    """Fixed prediction schedule on a precomputed checkpoint grid.

    X_grid rows must be grouped by query with `grid_ptr`.  All checkpoint
    predictions are independent of the stopping decision, so they are made in
    one batch; the stop is then the first checkpoint whose prediction >= rt.
    Returns stop at that checkpoint, or the natural end if none qualifies.
    """
    n = len(tr["ptr"]) - 1
    if grid_ptr is None:
        grid_ptr = tr["tr_ptr"]
    pred = predict(model, X_grid)
    stop_ndc = np.full(n, np.nan)
    recall = np.full(n, np.nan)
    calls = np.zeros(n, dtype=np.int32)
    stop_pred = np.full(n, np.nan)
    trn = tr["tr_ndc"] if "tr_ndc" in tr else None
    for i in range(n):
        gs, ge = grid_ptr[i], grid_ptr[i + 1]
        if min_ndc and trn is not None:
            while gs < ge and trn[gs] < min_ndc:
                gs += 1
        if ge > gs:
            pp = pred[gs:ge]
            hit = np.flatnonzero(pp >= rt - 1e-12)
            if hit.size:
                j = int(hit[0])
                stop_ndc[i] = tr["tr_ndc"][gs + j]
                # recall at that grid state: locate in step trace
                segn = _seg(tr["ptr"], tr["step_ndc"], i)
                segr = _seg(tr["ptr"], tr["step_recall"], i)
                k = int(np.searchsorted(segn, stop_ndc[i], side="left"))
                k = min(k, len(segn) - 1)
                recall[i] = segr[k]
                calls[i] = j + 1
                stop_pred[i] = pp[j]
                continue
        # no prediction crossed the target -> natural termination
        segn = _seg(tr["ptr"], tr["step_ndc"], i)
        segr = _seg(tr["ptr"], tr["step_recall"], i)
        if len(segn):
            stop_ndc[i] = segn[-1]
            recall[i] = segr[-1]
        calls[i] = max(0, ge - gs)
    return {"pred": pred, "stop_ndc": stop_ndc, "recall": recall,
            "calls": calls, "stop_pred": stop_pred, "name": name}


def darth_adaptive_from_steps(tr, model, rt, ipi, mpi, name="",
                              max_calls=10000):
    """DARTH original: adaptive interval p_i = mpi+(ipi-mpi)(rt-pred)."""
    n = len(tr["ptr"]) - 1
    stop_ndc = np.full(n, np.nan)
    recall = np.full(n, np.nan)
    calls = np.zeros(n, dtype=np.int32)
    stop_pred = np.full(n, np.nan)
    preds_out = [None] * n
    for i in range(n):
        s, e = tr["ptr"][i], tr["ptr"][i + 1]
        segn = tr["step_ndc"][s:e]
        segr = tr["step_recall"][s:e]
        segf = tr["step_feat"][s:e]
        if len(segn) == 0:
            continue
        last = 0.0
        pi = float(ipi)
        preds = []
        stopped = False
        for _ in range(max_calls):
            j = int(np.searchsorted(segn, last + pi, side="left"))
            if j >= len(segn):
                break
            p = float(predict(model, segf[j:j + 1])[0])
            preds.append((int(segn[j]), p))
            calls[i] += 1
            last = float(segn[j])
            if p >= rt - 1e-12:
                stop_ndc[i] = segn[j]
                recall[i] = segr[j]
                stop_pred[i] = p
                stopped = True
                break
            pi = max(float(mpi), float(mpi) + (float(ipi) - float(mpi))
                     * (rt - p))
        if not stopped:
            stop_ndc[i] = segn[-1]
            recall[i] = segr[-1]
        preds_out[i] = preds
    return {"stop_ndc": stop_ndc, "recall": recall, "calls": calls,
            "stop_pred": stop_pred, "preds": preds_out, "name": name}


def darth_offline_at_checkpoints(tr, model, rt, grid_ptr=None):
    """Predictions at stored checkpoints without stopping (offline eval)."""
    if grid_ptr is None:
        grid_ptr = tr["tr_ptr"]
    n = len(tr["ptr"]) - 1
    pred = predict(model, tr.get("tr_x", np.zeros((0, 11), np.float32)))
    return pred


def laet_with_budget(tr, B):
    """Replay a precomputed budget vector (used by multiplier calibration)."""
    n = len(tr["ptr"]) - 1
    stop_ndc = np.full(n, np.nan); recall = np.full(n, np.nan)
    truncated = np.zeros(n, dtype=bool)
    for i in range(n):
        seg = _seg(tr["ptr"], tr["step_ndc"], i)
        if len(seg) == 0:
            continue
        j = int(np.searchsorted(seg, B[i], side="left"))
        if j >= len(seg):
            j = len(seg) - 1
            truncated[i] = B[i] > seg[-1] + 1e-9
        stop_ndc[i] = seg[j]
        recall[i] = tr["step_recall"][tr["ptr"][i] + j]
    return {"stop_ndc": stop_ndc, "recall": recall, "truncated": truncated}


def annie_with_target(tr, target, lower_ndc):
    target = np.maximum(np.asarray(target, np.float64),
                        np.asarray(lower_ndc, np.float64))
    return laet_with_budget(tr, target)


def calibrate_multiplier(pred_log2, tr, f, pred_max, target=0.9,
                         mults=None, metric="mean_recall"):
    """Smallest multiplier whose validation mean recall reaches target."""
    if mults is None:
        mults = np.geomspace(0.05, 200.0, 80)
    p = np.asarray(pred_log2, np.float64)
    rec = np.zeros((len(p), len(mults)))
    base = np.power(2.0, np.maximum(p, 0.0))
    for j, m in enumerate(mults):
        B = np.maximum(float(f), np.minimum(base * m, float(pred_max)))
        rec[:, j] = laet_with_budget(tr, B)["recall"]
    if metric == "hit_rate":
        score = np.mean(rec >= target - 1e-12, axis=0)
        ok = np.flatnonzero(score >= target)
    else:
        score = rec.mean(axis=0)
        ok = np.flatnonzero(score >= target)
    j = int(ok[0]) if ok.size else int(np.argmax(score))
    return float(mults[j]), score, mults, rec


def calibrate_multiplier_hitrate(pred_log2, tr, f, pred_max, hit=0.95,
                                 mults=None):
    return calibrate_multiplier(pred_log2, tr, f, pred_max, target=hit,
                                mults=mults, metric="hit_rate")
