# src/awards_predictor/data/eligibility.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


AWARDS = ["mvp", "dpoy", "smoy", "roy", "mip"]


# ============================================================
# Helpers
# ============================================================

def _safe_upper(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().fillna("")


def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _bench_proxy(df: pd.DataFrame) -> pd.Series:
    out = df.copy()

    g = out["G"] if "G" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    gs = out["GS"] if "GS" in out.columns else pd.Series([pd.NA] * len(out), index=out.index)

    g = pd.to_numeric(g, errors="coerce")
    gs = pd.to_numeric(gs, errors="coerce")

    ratio = (gs / g.replace(0, pd.NA)).fillna(0)
    # “bench” proxy : moins de 50% de starts
    return ratio < 0.5


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
        # if MP is already per-game, mp/g would be tiny; detect & fallback
        # heuristic: if median(mpg) < 3 and median(mp) between 5..45, assume MP is per-game
        med_mpg = float(pd.Series(mpg).median(skipna=True)) if len(mpg) else np.nan
        med_mp = float(pd.Series(mp).median(skipna=True)) if len(mp) else np.nan
        if np.isfinite(med_mpg) and np.isfinite(med_mp) and (med_mpg < 3.0) and (5.0 <= med_mp <= 45.0):
            return mp
        return mpg

    return mp


# ============================================================
# Minutes / games thresholds (patch)
# NOTE: user request -> ROY set to 0
# ============================================================

MPG_MIN = {
    "mvp": 25,
    "dpoy": 20,
    "roy": 0,   
    "smoy": 0,
    "mip": 0,
}

G_MIN = {
    # help remove low-sample DPOY weirdness
    "dpoy": 40,
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
    """
    MVP: broad eligibility, but enforce meaningful minutes (MPG_MIN['mvp']) when possible.
    """
    out = df.copy()
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    out = _apply_minutes_games_filters(out, "mvp")
    return out


def filter_dpoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    DPOY: enforce meaningful minutes + minimum games (to avoid low-sample outliers).
    """
    out = df.copy()
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    out = _apply_minutes_games_filters(out, "dpoy")
    return out


def filter_smoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMOY: players mostly off the bench.
    Uses a proxy based on GS/G when available.
    No MPG threshold (by design).
    """
    out = df.copy()
    bench_mask = _bench_proxy(out)
    out = out[bench_mask]
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    # keep MPG_MIN['smoy']=0
    out = _apply_minutes_games_filters(out, "smoy")
    return out


def _norm_name(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().lower()
    for ch in [".", ",", "'", '"', "*", "`"]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s


def filter_roy(df: pd.DataFrame, rookies: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = df.copy()

    # 1) Source la plus fiable : flag déjà présent
    if "is_rookie" in out.columns:
        m = pd.to_numeric(out["is_rookie"], errors="coerce").fillna(0).astype(int) == 1
        out = out[m].reset_index(drop=True)
        # ROY MPG threshold requested = 0, so no extra filtering
        return _apply_minutes_games_filters(out, "roy").reset_index(drop=True)

    # 2) Fallback : rookies list (si dispo)
    if rookies is None or rookies.empty or "Player" not in rookies.columns:
        out = out.reset_index(drop=True)
        return _apply_minutes_games_filters(out, "roy").reset_index(drop=True)

    rookie_names = set(rookies["Player"].astype(str).str.strip())
    m = out["Player"].astype(str).str.strip().isin(rookie_names)
    out = out[m].reset_index(drop=True)
    return _apply_minutes_games_filters(out, "roy").reset_index(drop=True)


def filter_mip(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # pas rookies
    if "is_rookie" in out.columns:
        out["is_rookie"] = pd.to_numeric(out["is_rookie"], errors="coerce").fillna(0).astype(int)
        out = out[out["is_rookie"] == 0]

    # min games
    if "G" in out.columns:
        out["G"] = pd.to_numeric(out["G"], errors="coerce")
        out = out[out["G"].fillna(0) >= 20]

    # min minutes per game (dataset-dependent: MP may be per-game or total)
    mpg = _mpg_series(out)
    out = out[mpg.fillna(0) >= 15]

    # exiger saison N-1 dispo (proxy : prev_MP ou prev_pct_MP)
    prev_candidates = [c for c in ["prev_MP", "prev_pct_MP"] if c in out.columns]
    if prev_candidates:
        pc = prev_candidates[0]
        out[pc] = pd.to_numeric(out[pc], errors="coerce")
        out = out[out[pc].fillna(0) > 0]

    # keep MPG_MIN['mip']=0 for the global patch; MIP already has its own mpg>=15 rule above
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
