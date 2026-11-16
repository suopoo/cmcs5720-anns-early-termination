"""GBDT training helpers (LightGBM), shared by all experiments."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb


def train_lgb(X, y, objective="regression", alpha=None, seed=42,
              n_estimators=100, learning_rate=0.2, num_leaves=31,
              min_child_samples=20, n_jobs=4):
    kw = dict(objective=objective, n_estimators=n_estimators,
              learning_rate=learning_rate, num_leaves=num_leaves,
              min_child_samples=min_child_samples, subsample=1.0,
              colsample_bytree=1.0, random_state=seed, n_jobs=n_jobs,
              verbose=-1)
    if objective == "quantile":
        kw["alpha"] = float(alpha)
    model = lgb.LGBMRegressor(**kw)
    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64))
    return model


def predict(model, X):
    return np.asarray(model.predict(np.asarray(X, dtype=np.float32)),
                      dtype=np.float64)


def gain_importance(model, names=None):
    gain = model.booster_.feature_importance(importance_type="gain")
    if names is None:
        names = [f"f{i}" for i in range(len(gain))]
    order = np.argsort(gain)[::-1]
    return [(names[i], float(gain[i])) for i in order if gain[i] > 0]
