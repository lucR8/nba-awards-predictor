from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


AWARDS = ["mvp", "dpoy", "smoy", "roy", "mip"]


# ============================================================
# Helpers
# ============================================================

def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _bench_proxy(df: pd.DataFrame) -> pd.Series:
    out = df.copy()

    g = out["G"] if "G" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    gs = out["GS"] if "GS" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)

    g = pd.to_numeric(g, errors="coerce")
    gs = pd.to_numeric(gs, errors="coerce")

    ratio = (gs / g.replace(0, pd.NA)).fillna(0)
    return ratio < 0.5  # bench proxy


def _mpg_series(df: pd.DataFrame) -> pd.Series:
    """
    Robust MP per game:
    - prefer MP_per_g when present
    - else compute MP/G when both exist
    - else fallback to MP (assumed already per-game in some datasets)
    """
    if "MP_per_g" in df.columns:
        return pd.to_numeric(df["MP_per_g"], errors="coerce")

    mp = pd.to_numeric(df["MP"], errors="coerce") if "MP" in df.columns else pd.Series(np.nan, index=df.index)

    if "G" in df.columns:
        g = pd.to_numeric(df["G"], errors="coerce").replace(0, np.nan)
        mpg = mp / g

        # heuristic: if MP is already per-game, mp/g would be tiny; detect & fallback
        med_mpg = float(pd.Series(mpg).median(skipna=True)) if len(mpg) else np.nan
        med_mp = float(pd.Series(mp).median(skipna=True)) if len(mp) else np.nan
        if np.isfinite(med_mpg) and np.isfinite(med_mp) and (med_mpg < 3.0) and (5.0 <= med_mp <= 45.0):
            return mp
        return mpg

    return mp


# ============================================================
# Minutes / games thresholds
# ============================================================

MPG_MIN = {
    "mvp": 25,
    "dpoy": 20,
    "roy": 0,   
    "smoy": 0,
    "mip": 0,   
}

G_MIN = {
    "dpoy": 40,  # avoid low-sample DPOY outliers
}


def _apply_minutes_games_filters(df: pd.DataFrame, award: str) -> pd.DataFrame:
    out = df.copy()

    mpg_min = int(MPG_MIN.get(award, 0))
    if mpg_min > 0:
        mpg = _mpg_series(out)
        out = out[mpg.fillna(0) >= mpg_min]

    g_min = int(G_MIN.get(award, 0))
    if g_min > 0 and "G" in out.columns:
        g = pd.to_numeric(out["G"], errors="coerce")
        out = out[g.fillna(0) >= g_min]

    return out


# ============================================================
# Award filters
# ============================================================

def filter_mvp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    return _apply_minutes_games_filters(out, "mvp")


def filter_dpoy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    return _apply_minutes_games_filters(out, "dpoy")


def filter_smoy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[_bench_proxy(out)]
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    return _apply_minutes_games_filters(out, "smoy")


def filter_roy(df: pd.DataFrame, rookies: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = df.copy()

    # 1) Use is_rookie if it yields something
    if "is_rookie" in out.columns:
        m = pd.to_numeric(out["is_rookie"], errors="coerce").fillna(0).astype(int) == 1
        flagged = out[m].reset_index(drop=True)

        if len(flagged) > 0:
            return _apply_minutes_games_filters(flagged, "roy").reset_index(drop=True)

        # if flag exists but produces empty, fallback to rookies list if possible,
        # otherwise keep broad instead of wiping everything
        if rookies is None or rookies.empty or "Player" not in rookies.columns:
            broad = out.reset_index(drop=True)
            return _apply_minutes_games_filters(broad, "roy").reset_index(drop=True)

        rookie_names = set(rookies["Player"].astype(str).str.strip())
        m2 = out["Player"].astype(str).str.strip().isin(rookie_names)
        out2 = out[m2].reset_index(drop=True)
        return _apply_minutes_games_filters(out2, "roy").reset_index(drop=True)

    # 2) No flag -> use rookies list if available else keep broad
    if rookies is None or rookies.empty or "Player" not in rookies.columns:
        broad = out.reset_index(drop=True)
        return _apply_minutes_games_filters(broad, "roy").reset_index(drop=True)

    rookie_names = set(rookies["Player"].astype(str).str.strip())
    m = out["Player"].astype(str).str.strip().isin(rookie_names)
    out = out[m].reset_index(drop=True)
    return _apply_minutes_games_filters(out, "roy").reset_index(drop=True)


def filter_mip(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # no rookies
    if "is_rookie" in out.columns:
        out["is_rookie"] = pd.to_numeric(out["is_rookie"], errors="coerce").fillna(0).astype(int)
        out = out[out["is_rookie"] == 0]

    # min games
    if "G" in out.columns:
        out["G"] = pd.to_numeric(out["G"], errors="coerce")
        out = out[out["G"].fillna(0) >= 20]

    # min minutes per game
    mpg = _mpg_series(out)
    out = out[mpg.fillna(0) >= 15]

    # require season N-1 available (proxy: prev_MP or prev_pct_MP)
    prev_candidates = [c for c in ["prev_MP", "prev_pct_MP"] if c in out.columns]
    if prev_candidates:
        pc = prev_candidates[0]
        out[pc] = pd.to_numeric(out[pc], errors="coerce")
        out = out[out[pc].fillna(0) > 0]

    return out


def apply_eligibility(df: pd.DataFrame, award: str, *, rookies: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    award = award.lower().strip()
    if award == "mvp":
        return filter_mvp(df)
    if award == "dpoy":
        return filter_dpoy(df)
    if award == "smoy":
        return filter_smoy(df)
    if award == "roy":
        return filter_roy(df, rookies=rookies)
    if award == "mip":
        return filter_mip(df)
    raise ValueError(f"Unknown award: {award}")
