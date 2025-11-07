from __future__ import annotations
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

from awards_predictor.config import MVPConfig

def make_mvp_pipeline(cat_features: List[str], num_features: List[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def train_ranker_as_classifier(features: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:
    # TODO: remplacer par de vrais labels historiques
    # Ici on simule : top 3 par 'impact_all' sont positifs
    df = features.copy()
    df = df.sort_values("impact_all", ascending=False).reset_index(drop=True)
    df["label_mvp"] = 0
    df.loc[:2, "label_mvp"] = 1  # 3 meilleurs comme MVP "proxy"
    cfg = MVPConfig()
    X = df[cfg.categorical_features + cfg.numeric_features + [
        "pts_pctile","ast_pctile","reb_pctile","ts_pctile","ws_pctile","bpm_pctile","vorp_pctile",
        "pts_pos_pctile","ast_pos_pctile","reb_pos_pctile","ts_pos_pctile","ws_pos_pctile","bpm_pos_pctile","vorp_pos_pctile",
        "pts_z","ast_z","reb_z","ts_pct_z","ws_z","bpm_z","vorp_z",
        "impact_off","impact_all"
    ]].copy()
    y = df["label_mvp"].astype(int).values

    pipe = make_mvp_pipeline(cat_features=cfg.categorical_features, num_features=[c for c in X.columns if c not in cfg.categorical_features])
    pipe.fit(X, y)
    return pipe, df
