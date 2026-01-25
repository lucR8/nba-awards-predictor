from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from awards_predictor.config import AWARDS, HIST_ENRICHED_PARQUET
from awards_predictor.features.build_matrix import build_target_matrix


# ============================================================
# Label resolution
# ============================================================

def _find_label_col(df: pd.DataFrame, award: str) -> str:
    cands = [
        f"y_{award}",
        f"is_{award}",
        f"{award}_label",
        f"{award}_winner",
        f"winner_{award}",
        award,
    ]
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(
        f"Cannot find label column for award='{award}'. Tried {cands}. "
        f"Available cols: {list(df.columns)[:50]} ..."
    )


def _as_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    x = pd.to_numeric(s, errors="coerce")
    return (x.fillna(0) > 0).astype(int)


def _infer_year_end(season_val) -> Optional[int]:
    if season_val is None or (isinstance(season_val, float) and pd.isna(season_val)):
        return None
    if isinstance(season_val, (int, np.integer)):
        return int(season_val)

    s = str(season_val).strip()
    if s.isdigit():
        return int(s)

    if "-" in s:
        a, b = s.split("-", 1)
        a, b = a.strip(), b.strip()
        if len(b) == 2 and a.isdigit():
            return int(a[:2] + b)
        if b.isdigit():
            return int(b)

    return None


# ============================================================
# Models
# ============================================================

def _make_baseline_model():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _make_tree_model():
    try:
        import xgboost as xgb  # type: ignore

        return xgb.XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=0,
        )
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X)).reshape(-1)


# ============================================================
# Training table
# ============================================================

@dataclass(frozen=True)
class TrainingTable:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def _infer_rookies_for_year(block: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Return rookies dataframe with at least ['Player'] column.
    Only uses explicit rookie flags (robust, non-heuristic).
    """
    if "Player" not in block.columns:
        return None

    if "is_rookie" in block.columns:
        m = pd.to_numeric(block["is_rookie"], errors="coerce").fillna(0).astype(int) == 1
        r = block.loc[m, ["Player"]].dropna().drop_duplicates().reset_index(drop=True)
        return r if len(r) else None

    return None


def _build_training_table(df_hist: pd.DataFrame, award: str, feature_set: str) -> TrainingTable:
    """
    Internal builder used by both training and evaluation.
    Kept as a stable internal function because evaluate_models historically imported it.
    """
    label_col = _find_label_col(df_hist, award)
    df_hist = df_hist.copy()

    # normalize season/year
    if "season" not in df_hist.columns and "year" in df_hist.columns:
        df_hist["season"] = df_hist["year"]

    df_hist["year"] = df_hist["season"].apply(_infer_year_end)
    df_hist = df_hist[~pd.isna(df_hist["year"])].copy()
    df_hist["year"] = df_hist["year"].astype(int)

    # IMPORTANT: build_target_matrix expects end-year ints in "season"
    df_hist["season"] = df_hist["year"].astype(int)

    years = sorted(df_hist["year"].unique().tolist())
    if len(years) < 10:
        raise ValueError("Not enough seasons to train models.")

    Xs: list[pd.DataFrame] = []
    ys: list[pd.Series] = []
    metas: list[pd.DataFrame] = []

    for y in years:
        block = df_hist[df_hist["year"] == y].copy()
        block["__row_id"] = block.index.astype(int)
        block["season"] = int(y)

        hist_before = df_hist[df_hist["year"] < y].copy()
        hist_before["season"] = hist_before["year"].astype(int)

        # MIP needs history
        if award.lower() == "mip" and hist_before.empty:
            continue

        rookies = None
        if award.lower() == "roy":
            rookies = _infer_rookies_for_year(block)
            if rookies is None or len(rookies) == 0:
                continue

        bundle = build_target_matrix(
            df_hist=hist_before,
            df_target=block,
            target_year=int(y),
            award=award,
            feature_set=feature_set,
            rookies=rookies,
        )

        if bundle.X is None or len(bundle.X) == 0:
            continue

        if "__row_id" not in bundle.meta.columns:
            raise RuntimeError("build_target_matrix must propagate __row_id")

        row_ids = pd.to_numeric(bundle.meta["__row_id"], errors="coerce")
        if row_ids.isna().any():
            raise RuntimeError(f"Found NA in __row_id (award={award}, year={y})")

        row_ids = row_ids.astype(int).to_numpy()
        y_season = _as_binary(df_hist.loc[row_ids, label_col]).reset_index(drop=True)

        Xs.append(bundle.X.reset_index(drop=True))
        ys.append(y_season)

        meta = bundle.meta.reset_index(drop=True).copy()
        meta["year"] = int(y)
        metas.append(meta)

    if not Xs:
        raise ValueError(f"No usable seasons for award={award}")

    X_all = pd.concat(Xs, ignore_index=True)
    y_all = pd.concat(ys, ignore_index=True)
    meta_all = pd.concat(metas, ignore_index=True)

    return TrainingTable(X=X_all, y=y_all, meta=meta_all)


def build_training_table(df_hist: pd.DataFrame, award: str, feature_set: str) -> TrainingTable:
    """
    Public API wrapper (preferred import path).
    """
    return _build_training_table(df_hist, award=award, feature_set=feature_set)


# ============================================================
# Train loop
# ============================================================

def train_all(models_dir: Path, val_years: int = 2, test_years: int = 3) -> str:
    df_hist = pd.read_parquet(HIST_ENRICHED_PARQUET)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = models_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    families = [("baseline", "baseline"), ("tree", "tree")]

    from sklearn.metrics import roc_auc_score, log_loss

    for award in AWARDS:
        for family, feature_set in families:
            table = build_training_table(df_hist, award, feature_set)

            X, y, meta = table.X, table.y, table.meta
            years = sorted(meta["year"].unique().tolist())

            test_years_cut = years[-test_years:]
            val_years_cut = years[-(test_years + val_years):-test_years]

            is_test = meta["year"].isin(test_years_cut).to_numpy()
            is_val = meta["year"].isin(val_years_cut).to_numpy()
            is_train = ~(is_test | is_val)

            model = _make_baseline_model() if family == "baseline" else _make_tree_model()

            if family == "tree":
                try:
                    model.fit(X[is_train], y[is_train], eval_set=[(X[is_val], y[is_val])], verbose=False)
                except TypeError:
                    model.fit(X[is_train], y[is_train])
            else:
                model.fit(X[is_train], y[is_train])

            p_val = _predict_proba(model, X[is_val])
            p_test = _predict_proba(model, X[is_test])

            metrics = {
                "award": award,
                "family": family,
                "feature_set": feature_set,
                "n_train": int(is_train.sum()),
                "n_val": int(is_val.sum()),
                "n_test": int(is_test.sum()),
                "val_auc": float(roc_auc_score(y[is_val], p_val)),
                "test_auc": float(roc_auc_score(y[is_test], p_test)),
                "val_logloss": float(log_loss(y[is_val], np.clip(p_val, 1e-6, 1 - 1e-6))),
                "test_logloss": float(log_loss(y[is_test], np.clip(p_test, 1e-6, 1 - 1e-6))),
                "years": {
                    "val_years": [int(v) for v in val_years_cut],
                    "test_years": [int(t) for t in test_years_cut],
                },
            }

            out_dir = out_root / family / award
            out_dir.mkdir(parents=True, exist_ok=True)

            import joblib
            joblib.dump(model, out_dir / "model.joblib")
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return run_id


__all__ = [
    "TrainingTable",
    "train_all",
    "build_training_table",
    "_build_training_table",  # backward-compatible for evaluate_models.py
]


def main() -> int:
    train_all(Path("models"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
