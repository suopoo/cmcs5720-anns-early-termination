"""Shared plumbing for the three experiment drivers."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .core import paths
from . import trace as T

TRACE_DIR = os.path.join(paths.EXP_ROOT, "shared", "traces")


def trpath(name):
    return os.path.join(TRACE_DIR, name + ".npz")


def load_trace(name):
    return dict(np.load(trpath(name), allow_pickle=False))


def has_trace(name):
    return os.path.exists(trpath(name))


def grid_query_index(ptr, nq=None):
    ptr = np.asarray(ptr, dtype=np.int64)
    n = len(ptr) - 1 if nq is None else nq
    return np.repeat(np.arange(n), np.diff(ptr))


def summarize(x, prefix=""):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    qs = [10, 25, 50, 75, 90, 95, 99]
    d = {f"{prefix}mean": float(x.mean()),
         f"{prefix}median": float(np.median(x)),
         f"{prefix}std": float(x.std()),
         f"{prefix}min": float(x.min()),
         f"{prefix}max": float(x.max()),
         f"{prefix}n": int(x.size)}
    for q in qs:
        d[f"{prefix}p{q}"] = float(np.percentile(x, q))
    return d


def error_stats(pred, true, prefix=""):
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[ok], true[ok]
    if p.size == 0:
        return {}
    e = p - t
    abs_e = np.abs(e)
    d = {f"{prefix}signed_mean": float(e.mean()),
         f"{prefix}signed_median": float(np.median(e)),
         f"{prefix}abs_mean": float(abs_e.mean()),
         f"{prefix}abs_median": float(np.median(abs_e)),
         f"{prefix}rmse": float(np.sqrt(np.mean(e ** 2))),
         f"{prefix}over_rate": float(np.mean(e > 0)),
         f"{prefix}under_rate": float(np.mean(e < 0)),
         f"{prefix}p95_abs": float(np.percentile(abs_e, 95)),
         f"{prefix}p99_abs": float(np.percentile(abs_e, 99)),
         f"{prefix}n": int(p.size)}
    nz = np.abs(t) > 1e-9
    if nz.any():
        d[f"{prefix}mape"] = float(np.mean(abs_e[nz] / np.abs(t[nz])))
    d[f"{prefix}qerror"] = float(np.mean(
        np.maximum(p / np.maximum(t, 1e-9), t / np.maximum(p, 1e-9))))
    return d


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_csv(rows, path):
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def safe_div(a, b, default=0.0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    out = np.full_like(a, default, dtype=float)
    ok = np.abs(b) > 1e-12
    out[ok] = a[ok] / b[ok]
    return out
