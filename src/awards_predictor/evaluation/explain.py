from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def explain_linear_coefficients(model: Any, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Works for LogisticRegression-like models with coef_.
    Returns top positive/negative coefficients.
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model has no coef_")

    coef = np.asarray(model.coef_).ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    df["abs"] = df["coef"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n).drop(columns=["abs"])
    return df


def explain_permutation_importance(model: Any, X, y, feature_names: List[str], top_n: int = 20, n_repeats: int = 10) -> pd.DataFrame:
    """
    Model-agnostic importance using permutation.
    Slower but defendable for tree models.
    """
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0, scoring="roc_auc")
    imp = pd.DataFrame(
        {"feature": feature_names, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
    ).sort_values("importance_mean", ascending=False)
    return imp.head(top_n)
