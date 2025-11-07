from __future__ import annotations
import pandas as pd
from awards_predictor.data.collectors import read_players_csv, read_teams_csv
from awards_predictor.features.percentiles import add_percentiles, zscore_by_group

NUM_COLS = ["pts","ast","reb","ts_pct","usg_pct","ws","bpm","vorp","gp","mpg"]

def build_mvp_features(players_csv: str, teams_csv: str) -> pd.DataFrame:
    players = read_players_csv(players_csv)
    teams = read_teams_csv(teams_csv)

    df = players.merge(teams, on="team", how="left")
    # Percentiles global + par position
    df = add_percentiles(df, cols=["pts","ast","reb","ts_pct","ws","bpm","vorp"], groupby=None)
    df = add_percentiles(df, cols=["pts","ast","reb","ts_pct","ws","bpm","vorp"], groupby="position", suffix="_pos_pctile")
    # Z-scores par position
    df = zscore_by_group(df, cols=["pts","ast","reb","ts_pct","ws","bpm","vorp"], groupby="position")

    # Quelques features composites simples
    df["impact_off"] = df["ts_pct"] * (df["usg_pct"] / 30.0) * (1.0 + df["ast"]/10.0)
    df["impact_all"] = df["impact_off"] + (df["reb"]/10.0) + (df["bpm"]/5.0)

    # Garder un set de colonnes pour le modèle MVP (peut évoluer)
    feature_cols = [
        "gp","mpg","pts","ast","reb","ts_pct","usg_pct","ws","bpm","vorp",
        "win_pct","off_rating","def_rating","position","team","starter_flag",
        "pts_pctile","ast_pctile","reb_pctile","ts_pctile","ws_pctile","bpm_pctile","vorp_pctile",
        "pts_pos_pctile","ast_pos_pctile","reb_pos_pctile","ts_pos_pctile","ws_pos_pctile","bpm_pos_pctile","vorp_pos_pctile",
        "pts_z","ast_z","reb_z","ts_pct_z","ws_z","bpm_z","vorp_z",
        "impact_off","impact_all"
    ]
    return df[feature_cols + ["player_id","player_name"]]
