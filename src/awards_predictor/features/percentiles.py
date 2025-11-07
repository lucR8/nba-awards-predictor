from __future__ import annotations
import pandas as pd
import numpy as np

def add_percentiles(df: pd.DataFrame, cols: list[str], groupby: str | None = None, suffix: str = "_pctile") -> pd.DataFrame:
    out = df.copy()
    if groupby is None:
        for c in cols:
            out[c + suffix] = out[c].rank(pct=True)
        return out
    # group-wise percentiles (e.g., by position)
    out_list = []
    for g, gdf in out.groupby(groupby):
        gdf = gdf.copy()
        for c in cols:
            gdf[c + suffix] = gdf[c].rank(pct=True)
        out_list.append(gdf)
    return pd.concat(out_list, axis=0, ignore_index=True)

def zscore_by_group(df: pd.DataFrame, cols: list[str], groupby: str) -> pd.DataFrame:
    out = df.copy()
    out_list = []
    for g, gdf in out.groupby(groupby):
        gdf = gdf.copy()
        for c in cols:
            mu = gdf[c].mean()
            sd = gdf[c].std(ddof=0) or 1.0
            gdf[c + "_z"] = (gdf[c] - mu) / sd
        out_list.append(gdf)
    return pd.concat(out_list, axis=0, ignore_index=True)
