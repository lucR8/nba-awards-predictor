# src/awards_predictor/features/columns.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ============================================================
# Colonnes d'identité / catégorielles
# ============================================================

ID_COLS = ["season", "Player", "Tm"]
CATEGORICAL_COLS = ["Pos"]


# ============================================================
# Features autorisées (alignées avec votre pipeline)
# ============================================================

# Volume = crédibilité du signal (utilisé aussi dans notebook 06)
VOLUME_COLS = [
    "G",
    "GS",
    "MP",
    "MP_per_g",  # si présent
]

# Percentiles = signal principal
PERCENTILE_PREFIXES = ("pct_", "percentile_", "pctile_")
PERCENTILE_SUFFIXES = ("_pct", "_pctile", "_percentile")

def infer_percentile_cols(columns: Iterable[str]) -> list[str]:
    cols = list(map(str, columns))
    return sorted(
        c for c in cols
        if c.startswith(PERCENTILE_PREFIXES) or any(c.endswith(suf) for suf in PERCENTILE_SUFFIXES)
    )



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
    """
    Baseline (logreg) :
    - percentiles uniquement
    - + volume minimal
    """
    df_cols = set(map(str, df_columns))
    numeric = infer_percentile_cols(df_cols)
    numeric += [c for c in VOLUME_COLS if c in df_cols]

    return FeatureSet(
        name="baseline_pct",
        numeric=list(dict.fromkeys(numeric)),
        categorical=[c for c in CATEGORICAL_COLS if c in df_cols],
    )


def build_tree_feature_set(df_columns: Iterable[str]) -> FeatureSet:
    """
    Tree models :
    - mêmes percentiles
    - mêmes volumes
    (pas de raw advanced → cohérence avec notebook)
    """
    return build_baseline_feature_set(df_columns)
