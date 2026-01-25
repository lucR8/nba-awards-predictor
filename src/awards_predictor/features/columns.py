# src/awards_predictor/features/columns.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

# ============================================================
# Colonnes d'identité / catégorielles
# ============================================================

ID_COLS = ["season", "Player", "Tm"]
CATEGORICAL_COLS = ["Pos"]

# ============================================================
# Anti-leakage (CRITICAL)
# ============================================================

_LEAKAGE_PAT = re.compile(
    r"(^|_)is_.*winner($|_)"
    r"|(^|_)is_.*_winner($|_)"
    r"|(^|_)winner($|_)"
    r"|(^|_)target($|_)"
    r"|(^|_)label($|_)",
    re.IGNORECASE,
)

def is_leakage_feature(col: str) -> bool:
    c = str(col)

    # direct
    if _LEAKAGE_PAT.search(c):
        return True

    # variants fréquents (pct_/delta_ autour des labels)
    if c.startswith(("pct_is_", "delta_pct_is_", "delta_is_")):
        return True
    if c.startswith(("pct_winner", "delta_pct_winner", "delta_winner")):
        return True

    return False


# ============================================================
# Features autorisées
# ============================================================

VOLUME_COLS = ["G", "GS", "MP", "MP_per_g"]

PERCENTILE_PREFIXES = ("pct_", "percentile_", "pctile_")
PERCENTILE_SUFFIXES = ("_pct", "_pctile", "_percentile")


def infer_percentile_cols(columns: Iterable[str]) -> list[str]:
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
# Feature sets utilisés dans le projet
# ============================================================

def build_baseline_feature_set(df_columns: Iterable[str]) -> FeatureSet:
    df_cols = set(map(str, df_columns))

    numeric = infer_percentile_cols(df_cols)
    numeric += [c for c in VOLUME_COLS if c in df_cols and (not is_leakage_feature(c))]

    return FeatureSet(
        name="baseline_pct",
        numeric=list(dict.fromkeys(numeric)),
        categorical=[c for c in CATEGORICAL_COLS if c in df_cols],
    )


def build_tree_feature_set(df_columns: Iterable[str]) -> FeatureSet:
    return build_baseline_feature_set(df_columns)
