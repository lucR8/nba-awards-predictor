from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

def basic_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = None
    try:
        # clip probs to avoid log(0)
        eps = 1e-9
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        out["logloss"] = float(log_loss(y_true, y_prob))
    except Exception:
        out["logloss"] = None
    return out
