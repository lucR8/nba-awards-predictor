# src/awards_predictor/sanity/leakage_check.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd


LEAK_KEYWORDS = [
    "winner", "award", "rank", "vote", "share",
    "points", "pts", "first", "second", "third",
    "all_nba", "all_def",
    "mvp", "dpoy", "roy", "smoy", "mip",
]


def _name_based_leakage(columns: list[str]) -> list[str]:
    bad = []
    for c in columns:
        cl = c.lower()
        for kw in LEAK_KEYWORDS:
            if re.search(rf"\b{kw}\b", cl):
                bad.append(c)
                break
    return sorted(set(bad))


def _corr_based_leakage(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95) -> list[tuple[str, float]]:
    leaks = []
    yv = pd.Series(y).astype(float)

    for c in X.columns:
        try:
            xv = pd.to_numeric(X[c], errors="coerce")
            if xv.notna().sum() < 10:
                continue
            corr = np.corrcoef(xv.fillna(0), yv.fillna(0))[0, 1]
            if np.isfinite(corr) and abs(corr) >= threshold:
                leaks.append((c, float(corr)))
        except Exception:
            continue

    return sorted(leaks, key=lambda x: abs(x[1]), reverse=True)


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
    else:
        print(full_msg)
