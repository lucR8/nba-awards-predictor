# src/awards_predictor/sanity/leakage_check.py
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from awards_predictor.features.columns import is_leakage_feature


# Award tokens used only to detect columns like "mvp_share", "dpoy_rank", etc.
_AWARD_TOKENS = ("mvp", "dpoy", "roy", "smoy", "mip")

# Conservative vote/award proxy patterns (avoid catching normal stats like "pts")
_AWARD_PROXY_PAT = re.compile(
    r"(^|_)(vote|voting|share|rank|award|points)($|_)", re.IGNORECASE
)


def _name_based_leakage(columns: Iterable[str]) -> list[str]:
    """
    Name-based leakage detection.

    Source of truth: features.columns.is_leakage_feature().
    Additionally flags award-specific vote/share/rank columns such as:
      - mvp_share, dpoy_rank, roy_votes, smoy_points, etc.
    """
    bad: list[str] = []
    for c in map(str, columns):
        cl = c.lower()

        # canonical rule-set (aligned with notebooks / columns.py)
        if is_leakage_feature(c):
            bad.append(c)
            continue

        # award-specific proxy columns (only when token + proxy keyword co-occur)
        if any(tok in cl for tok in _AWARD_TOKENS) and _AWARD_PROXY_PAT.search(cl):
            bad.append(c)
            continue

    return sorted(set(bad))


def _corr_based_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.95,
    min_non_null: int = 25,
) -> list[tuple[str, float]]:
    """
    Correlation-based leakage detection.

    We compute Pearson correlation between each numeric column and y.
    We skip:
      - non-numeric columns
      - columns with too few non-null values
      - near-constant columns
      - constant y (cannot compute correlation)
    """
    if X.empty:
        return []

    yv = pd.to_numeric(pd.Series(y), errors="coerce").astype(float)
    if yv.nunique(dropna=True) <= 1:
        return []

    # Keep only numeric-ish columns
    Xn = X.apply(pd.to_numeric, errors="coerce")

    # Drop columns with too few observations
    non_null = Xn.notna().sum(axis=0)
    Xn = Xn.loc[:, non_null >= min_non_null]
    if Xn.shape[1] == 0:
        return []

    # Drop near-constant columns (std ~ 0)
    std = Xn.std(axis=0, skipna=True)
    Xn = Xn.loc[:, std > 1e-12]
    if Xn.shape[1] == 0:
        return []

    # Compute correlations
    corr = Xn.corrwith(yv, axis=0, drop=False)
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna()

    leaks = [(c, float(v)) for c, v in corr.items() if abs(v) >= threshold]
    leaks.sort(key=lambda t: abs(t[1]), reverse=True)
    return leaks


def check_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    award: str,
    corr_threshold: float = 0.95,
    strict: bool = True,
) -> None:
    """
    Raises RuntimeError if leakage is detected.

    - Name-based detection uses the same leakage rules as feature selection.
    - Correlation-based detection catches accidental target proxies.
    """
    cols = list(map(str, X.columns))

    name_leaks = _name_based_leakage(cols)
    corr_leaks = _corr_based_leakage(X, y, threshold=corr_threshold)

    if not name_leaks and not corr_leaks:
        print(f"[LEAKAGE] OK ({award})")
        return

    msg = [f"\n🚨 DATA LEAKAGE DETECTED for award={award}"]

    if name_leaks:
        msg.append("\n🔴 Name-based leakage columns:")
        for c in name_leaks:
            msg.append(f"  - {c}")

    if corr_leaks:
        msg.append(f"\n🟠 High correlation leakage (|corr| ≥ {corr_threshold}):")
        for c, v in corr_leaks[:10]:
            msg.append(f"  - {c}: corr={v:.3f}")

    full_msg = "\n".join(msg)

    if strict:
        raise RuntimeError(full_msg)
    print(full_msg)
