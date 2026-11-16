"""Query-level paired statistics: bootstrap CIs + stratified permutation."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=42):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return {"mean": float(x.mean()),
            "lo": float(np.percentile(means, 100 * alpha / 2)),
            "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
            "n": int(x.size)}


def paired_diff(a, b, n_boot=2000, alpha=0.05, seed=42):
    """bootstrap CI of mean(a-b) over paired queries."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "relative": float("nan")}
    ci = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    mb = np.nanmean(b[ok])
    return {"diff": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "n": ci["n"],
            "relative": float(ci["mean"] / mb) if mb not in (0, np.nan) else float("nan")}


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def permutation_overlap_test(A, B, universe, strata=None, n_perm=2000, seed=42):
    """Is |A∩B| larger than expected under independence, conditional on
    each set's size (and optionally on oracle-cost stratum)?

    Returns observed intersection, expected, p-value, Jaccard, and
    conditional failure probability P(B|A).
    """
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    A = np.intersect1d(A, universe)
    B = np.intersect1d(B, universe)
    obs = int(np.intersect1d(A, B).size)
    union = int(np.union1d(A, B).size)
    jac = obs / union if union else float("nan")
    condB = obs / len(A) if len(A) else float("nan")
    condA = obs / len(B) if len(B) else float("nan")
    exp = len(A) * len(B) / len(universe) if len(universe) else float("nan")
    rng = np.random.default_rng(seed)
    if strata is None:
        ge = 0
        for _ in range(n_perm):
            Bp = rng.choice(universe, size=len(B), replace=False)
            if np.intersect1d(A, Bp).size >= obs:
                ge += 1
    else:
        strata = np.asarray(strata, dtype=np.int64)
        pools = {s: universe[strata == s] for s in np.unique(strata)}
        ge = 0
        for _ in range(n_perm):
            Bp = []
            for s, pool in pools.items():
                nb = int(np.sum(strata[np.isin(universe, B)] == s))
                if nb and len(pool):
                    Bp.append(rng.choice(pool, size=min(nb, len(pool)),
                                         replace=False))
            Bp = np.concatenate(Bp) if Bp else np.array([], dtype=np.int64)
            if np.intersect1d(A, Bp).size >= obs:
                ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {"intersection": obs, "union": union, "jaccard": jac,
            "P_B_given_A": condB, "P_A_given_B": condA,
            "expected_independence": exp, "p_value": p,
            "n_A": int(len(A)), "n_B": int(len(B))}
