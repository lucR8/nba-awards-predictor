# src/awards_predictor/features/build_matrix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from awards_predictor.data.eligibility import apply_eligibility
from awards_predictor.features.columns import (
    FeatureSet,
    build_baseline_feature_set,
    build_tree_feature_set,
)
from awards_predictor.io.paths import TargetSnapshotPaths


# ============================================================
# Small utilities
# ============================================================

def _require_cols(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{ctx}] Missing required columns: {missing}")


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _infer_end_year(season_val) -> Optional[int]:
    """
    Accept:
      - int-like (2026)
      - '2025-26' or '2025-2026'
    Return end-year as int, else None.
    """
    if season_val is None or (isinstance(season_val, float) and pd.isna(season_val)):
        return None
    if isinstance(season_val, (int,)):
        return int(season_val)
    try:
        s = str(season_val).strip()
        if s.isdigit():
            return int(s)
        if "-" in s:
            a, b = s.split("-", 1)
            a = a.strip()
            b = b.strip()
            if len(b) == 2 and a.isdigit():
                return int(a[:2] + b)  # 2025 + 26 -> 2026
            if b.isdigit():
                return int(b)
    except Exception:
        return None
    return None


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required ID columns exist:
      - Player
      - season (end-year int)
      - Tm (team abbrev)
    """
    x = df.copy()

    # Team column drift
    if "Tm" not in x.columns:
        for alt in ("Team", "team", "tm"):
            if alt in x.columns:
                x = x.rename(columns={alt: "Tm"})
                break

    # season drift
    if "season" not in x.columns:
        for alt in ("year", "Year"):
            if alt in x.columns:
                x = x.rename(columns={alt: "season"})
                break

    if "season" in x.columns:
        x["season"] = x["season"].apply(_infer_end_year)

    return x


def _one_hot_pos(df: pd.DataFrame, pos_col: str = "Pos") -> pd.DataFrame:
    if pos_col not in df.columns:
        return df
    x = df.copy()
    x[pos_col] = x[pos_col].astype("string").fillna("UNK")
    dummies = pd.get_dummies(x[pos_col], prefix="Pos")
    x = x.drop(columns=[pos_col])
    return pd.concat([x, dummies], axis=1)


# ============================================================
# prev_* and delta_* (vectorized, no fragmentation)
# ============================================================

def add_prev_features(
    df: pd.DataFrame,
    *,
    player_col: str = "Player",
    season_col: str = "season",
    prev_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute prev_* for selected numeric columns in ONE shot (no per-column insert).
    """
    _require_cols(df, [player_col, season_col], "add_prev_features")
    out = df.copy()

    out[season_col] = _safe_num(out[season_col]).astype("Int64")
    out = out.sort_values([player_col, season_col]).reset_index(drop=True)

    if prev_cols is None:
        prev_cols = [
            c for c in out.columns
            if c not in {player_col, season_col, "Tm", "__row_id"}
            and pd.api.types.is_numeric_dtype(out[c])
            and not str(c).startswith("prev_")
        ]

    # groupby-shift on a DATAFRAME -> fast + no fragmentation
    prev_block = out.groupby(player_col, sort=False)[prev_cols].shift(1)
    prev_block = prev_block.add_prefix("prev_")

    return pd.concat([out, prev_block], axis=1)


def add_delta_features(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    prefix: str = "delta_",
) -> pd.DataFrame:
    """
    Compute delta_{c} = c - prev_c for all numeric_cols in one concat (no fragmentation).
    """
    out = df.copy()

    # keep only pairs that exist
    cur_cols = [c for c in numeric_cols if c in out.columns and f"prev_{c}" in out.columns]
    if not cur_cols:
        return out

    cur = out[cur_cols].apply(pd.to_numeric, errors="coerce")
    prev = out[[f"prev_{c}" for c in cur_cols]].apply(pd.to_numeric, errors="coerce")
    prev.columns = cur_cols  # align for subtraction

    delta = (cur - prev).add_prefix(prefix)
    return pd.concat([out, delta], axis=1)


# ============================================================
# Matrix builders
# ============================================================

@dataclass(frozen=True)
class MatrixBundle:
    X: pd.DataFrame
    meta: pd.DataFrame  # season, Player, Tm, __row_id
    feature_set_name: str


def build_target_matrix(
    *,
    df_hist: pd.DataFrame,
    df_target: pd.DataFrame,
    target_year: int,
    award: str,
    rookies: Optional[pd.DataFrame] = None,
    feature_set: str = "baseline",  # "baseline" | "tree"
    add_prev: bool = True,
    prev_cols: Optional[list[str]] = None,
) -> MatrixBundle:
    """
    Build X for a season (target_year) without labels.

    Key points:
      - concat (hist + target) BEFORE prev_ so target rows can see N-1 season
      - for MIP, add delta_* and include them into X
      - keep __row_id on target rows for label recovery
    """
    df_hist = _normalize_schema(df_hist)
    df_target = _normalize_schema(df_target)

    # stable row_id for target rows (label recovery)
    if "__row_id" not in df_target.columns:
        df_target = df_target.copy()
        df_target["__row_id"] = df_target.index.astype(int)

    if "__row_id" not in df_hist.columns:
        df_hist = df_hist.copy()
        df_hist["__row_id"] = -1

    _require_cols(df_target, ["season", "Player", "Tm", "__row_id"], "build_target_matrix(df_target)")
    _require_cols(df_hist, ["season", "Player", "Tm"], "build_target_matrix(df_hist)")

    # concat hist + target (important for prev season lookup)
    df_all = pd.concat([df_hist, df_target], ignore_index=True, sort=False)
    df_all["season"] = _safe_num(df_all["season"]).astype("Int64")

    # choose base feature set on df_all (so numeric list is stable)
    if feature_set == "baseline":
        fs_base: FeatureSet = build_baseline_feature_set(df_all.columns)
    elif feature_set == "tree":
        fs_base = build_tree_feature_set(df_all.columns)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # compute prev_* on df_all BEFORE filtering target_year
    if add_prev:
        if prev_cols is None:
            # IMPORTANT for MIP: ensure MP / pct_MP are included if they exist,
            # because some filter_mip implementations require prev_MP or prev_pct_MP.
            prev_cols = fs_base.resolve_numeric(df_all.columns)
            for must in ("MP", "pct_MP"):
                if must in df_all.columns and must not in prev_cols:
                    prev_cols.append(must)

        df_all = add_prev_features(df_all, prev_cols=prev_cols)

    # isolate target season rows
    df_t = df_all[df_all["season"] == int(target_year)].copy()

    # eligibility filter (ROY rookies, SMOY bench proxy, etc.)
    df_t = apply_eligibility(df_t, award, rookies=rookies)

    # MIP: add deltas vs prev season and include them in X
    fs_final = fs_base
    if award.lower().strip() == "mip":
        num_cols_for_delta = fs_base.resolve_numeric(df_t.columns)
        df_t = add_delta_features(df_t, numeric_cols=num_cols_for_delta, prefix="delta_")

        delta_cols = [f"delta_{c}" for c in num_cols_for_delta if f"delta_{c}" in df_t.columns]
        fs_final = FeatureSet(
            name=f"{fs_base.name}_mip_delta",
            numeric=list(dict.fromkeys(num_cols_for_delta + delta_cols)),
            categorical=fs_base.resolve_categorical(df_t.columns),
        )

    # final column selection
    num_cols = fs_final.resolve_numeric(df_t.columns)
    cat_cols = fs_final.resolve_categorical(df_t.columns)

    X = df_t[num_cols + cat_cols].copy()
    if "Pos" in X.columns:
        X = _one_hot_pos(X, pos_col="Pos")

    meta_cols = ["season", "Player", "Tm", "__row_id"]
    meta_cols = [c for c in meta_cols if c in df_t.columns]
    meta = df_t[meta_cols].copy()

    return MatrixBundle(X=X, meta=meta, feature_set_name=fs_final.name)


def load_target_dataset(paths: TargetSnapshotPaths) -> pd.DataFrame:
    if paths.build_players_with_bio.exists():
        return pd.read_parquet(paths.build_players_with_bio)
    if paths.build_players_final.exists():
        return pd.read_parquet(paths.build_players_final)
    raise FileNotFoundError(
        f"Missing target dataset. Expected:\n"
        f"- {paths.build_players_with_bio}\n"
        f"- {paths.build_players_final}"
    )
