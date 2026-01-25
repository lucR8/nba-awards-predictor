from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


AWARDS = ["mvp", "dpoy", "smoy", "roy", "mip"]


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



def filter_mvp(df: pd.DataFrame) -> pd.DataFrame:
    """
    MVP: broad eligibility. Keep players with meaningful minutes/games when possible.
    """
    out = df.copy()
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    return out


def filter_dpoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    DPOY: similar to MVP but can be more permissive. Keep meaningful MP if available.
    """
    out = df.copy()
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    return out


def filter_smoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMOY: players mostly off the bench.
    Uses a proxy based on GS/G when available.
    """
    out = df.copy()
    bench_mask = _bench_proxy(out)
    out = out[bench_mask]
    if _has_col(out, "MP"):
        mp = pd.to_numeric(out["MP"], errors="coerce")
        out = out[mp.fillna(0) > 0]
    return out


def _norm_name(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().lower()
    # retire ponctuation simple + doubles espaces
    for ch in [".", ",", "'", '"', "*", "`"]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s


def filter_roy(df: pd.DataFrame, rookies: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = df.copy()

    # 1) Source la plus fiable : flag déjà présent
    if "is_rookie" in out.columns:
        m = pd.to_numeric(out["is_rookie"], errors="coerce").fillna(0).astype(int) == 1
        return out[m].reset_index(drop=True)

    # 2) Fallback : rookies list (si dispo)
    if rookies is None or rookies.empty or "Player" not in rookies.columns:
        return out.reset_index(drop=True)

    rookie_names = set(rookies["Player"].astype(str).str.strip())
    m = out["Player"].astype(str).str.strip().isin(rookie_names)
    return out[m].reset_index(drop=True)




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

    # min minutes per game (MP is per-game in your dataset)
    if "MP" in out.columns:
        out["MP"] = pd.to_numeric(out["MP"], errors="coerce")
        out = out[out["MP"].fillna(0) >= 15]

    # exiger saison N-1 dispo (proxy : prev_MP ou prev_pct_MP)
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
