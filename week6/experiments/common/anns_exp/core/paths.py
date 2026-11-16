"""Central path configuration for the week6 experiment package.

Everything lives on D:.  Shared artefacts (GIST graph cache, ground truth,
query splits) go to `experiments/shared`; per-experiment outputs go to the
matching `exp0X_*` directory.
"""
from __future__ import annotations

import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# .../experiments/common/anns_exp/core -> .../experiments/common -> .../experiments
EXP_ROOT = os.path.dirname(PKG_DIR)
SHARED_DIR = os.path.join(EXP_ROOT, "shared")
CACHE_DIR = os.path.join(SHARED_DIR, "cache")
GT_DIR = os.path.join(SHARED_DIR, "ground_truth")
SPLIT_DIR = os.path.join(SHARED_DIR, "splits")
LOG_DIR = os.path.join(SHARED_DIR, "logs")

DATA_DIRS = [r"D:\HNSW_Project\datasets\gist", r"D:\大学\CUCS\CMCS5720\week5\data"]

for _d in (CACHE_DIR, GT_DIR, SPLIT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)


def dataset_path(name="gist"):
    fn = {"gist": "gist.hdf5", "sift": "sift-128-euclidean.hdf5"}[name]
    for d in DATA_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def exp_dir(exp):
    """exp in {'exp01','exp02','exp03'} -> absolute directory."""
    names = {
        "exp01": "exp01_feature_transfer",
        "exp02": "exp02_error_analysis",
        "exp03": "exp03_asymmetric_loss",
    }
    return os.path.join(EXP_ROOT, names[exp])


def ensure_exp_dirs(exp):
    for sub in ("configs", "data", "analysis", "figures", "reports", "logs"):
        os.makedirs(os.path.join(exp_dir(exp), sub), exist_ok=True)
