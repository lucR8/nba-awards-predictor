from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_players_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "player_id","player_name","team","position","gp","mpg","pts","ast","reb",
        "ts_pct","usg_pct","ws","bpm","vorp","starter_flag"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in players CSV: {missing}")
    return df

def read_teams_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"team","win_pct","off_rating","def_rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in teams CSV: {missing}")
    return df
