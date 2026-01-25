# src/awards_predictor/features/columns.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re


# ============================================================
# Identity / categorical columns
# ============================================================

ID_COLS = ["season", "Player", "Tm"]
CATEGORICAL_COLS = ["Pos"]


# ============================================================
# Anti-leakage (CRITICAL)
# ============================================================
# Aligns with notebook logic: remove any direct or proxy target/award columns,
# including rank/consideration and any award voting information.
#
# NOTE: We intentionally DO NOT include normal basketball stats (pts, ast, etc.)
# here because baseline is percentile-driven in V1.

# Base leakage keywords (match anywhere as token-ish boundaries)
_LEAKAGE_PAT = re.compile(
    r"(^|_)(winner|label|target)($|_)"          # labels / targets
    r"|(^|_)(rank)($|_)"                        # ranks are often derived from awards/votes
    r"|(^|_)(award)($|_)"                       # award proxies
    r"|(^|_)(vote|voting)($|_)"                 # voting proxies
    r"|(^|_)(share)($|_)"                       # share proxies (e.g., mvp_share)
    r"|(^|_)(all_nba|all_def|all_rookie)($|_)"  # end-of-season selections
    r"|(^|_)(consideration)($|_)"               # has_*consideration flags
    r"|(^|_)(has_.*consideration)($|_)",        # explicit has_*consideration variants
    re.IGNORECASE,
)


def is_leakage_feature(col: str) -> bool:
    """
    Return True if `col` is considered a leakage/proxy feature.
    This is intentionally conservative to keep V1 defensible.
    """
    c = str(col)

    # Direct match on leakage tokens
    if _LEAKAGE_PAT.search(c):
        return True

    # Common wrappers around leakage columns (pct_/prev_/delta_)
    # If the underlying column name contains leakage tokens, also treat it as leakage.
    wrappers = ("pct_", "percentile_", "pctile_", "prev_", "delta_", "diff_", "change_")
    for w in wrappers:
        if c.lower().startswith(w):
            base = c[len(w):]
            if _LEAKAGE_PAT.search(base):
                return True

    return False


# ============================================================
# Percentile feature detection
# ============================================================

PERCENTILE_PREFIXES = ("pct_", "percentile_", "pctile_")
PERCENTILE_SUFFIXES = ("_pct", "_pctile", "_percentile")


def infer_percentile_cols(columns: Iterable[str]) -> list[str]:
    """
    V1 policy (notebook-aligned):
      - Keep only percentile-like columns (pct_* or *_pct)
      - Exclude any leakage/proxy columns (winner/label/target/rank/vote/share/all_nba/consideration...)
    """
    cols = list(map(str, columns))
    out: list[str] = []

    for c in cols:
        if is_leakage_feature(c):
            continue

        if c.startswith(PERCENTILE_PREFIXES) or any(c.endswith(suf) for suf in PERCENTILE_SUFFIXES):
            out.append(c)

    return sorted(out)


def intersect_features(df_columns: Iterable[str], features: Iterable[str]) -> list[str]:
    df_cols = set(map(str, df_columns))
    out = [f for f in features if str(f) in df_cols]
    return list(dict.fromkeys(out))


# ============================================================
# FeatureSet
# ============================================================

@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric: list[str]
    categorical: list[str]

    def resolve_numeric(self, df_columns: Iterable[str]) -> list[str]:
        return intersect_features(df_columns, self.numeric)

    def resolve_categorical(self, df_columns: Iterable[str]) -> list[str]:
        return intersect_features(df_columns, self.categorical)


# ============================================================
# Feature sets used in the project (V1 notebook-aligned)
# ============================================================

def build_baseline_feature_set(df_columns: Iterable[str]) -> FeatureSet:
    df_cols = set(map(str, df_columns))

    # Baseline: percentiles only (notebook-aligned) + Pos categorical if present
    numeric = infer_percentile_cols(df_cols)

    return FeatureSet(
        name="baseline_pct_only",
        numeric=list(dict.fromkeys(numeric)),
        categorical=[c for c in CATEGORICAL_COLS if c in df_cols],
    )


def build_tree_feature_set(df_columns: Iterable[str]) -> FeatureSet:
    """
    V1 policy: same features as baseline (percentiles + Pos).
    Tree vs baseline differs by model family, not by feature space.

    Keeping the function allows a future V2 to expand tree features safely
    (raw stats, interactions, etc.) without changing downstream code.
    """
    fs = build_baseline_feature_set(df_columns)
    return FeatureSet(
        name="tree_pct_only",
        numeric=fs.numeric,
        categorical=fs.categorical,
    )
