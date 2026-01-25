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
# Label resolution (robust)
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
        f"Cannot find label column for award='{award}'. Tried: {cands}. "
        f"Available cols sample: {list(df.columns)[:60]} ..."
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
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
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
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1]
    p = model.predict(X)
    return np.asarray(p).reshape(-1)


# ============================================================
# Training table builder (season by season)
# ============================================================

@dataclass(frozen=True)
class TrainingTable:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # season, year, Player, Tm, __row_id
    
def _infer_rookies_for_year(block: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Return a rookies dataframe compatible with apply_eligibility(..., rookies=...)
    Expected at least a 'Player' column.
    Tries common conventions in your historical dataset.
    """
    if "Player" not in block.columns:
        return None

    # 1) Basketball-Reference convention: Rk == 1 for rookies (if present in your built dataset)
    if "Rk" in block.columns:
        m = pd.to_numeric(block["Rk"], errors="coerce") == 1
        r = block.loc[m, ["Player"]].dropna().drop_duplicates()
        return r if len(r) else None

    # 2) Experience/Exp == 0
    for col in ("Experience", "Exp", "EXP"):
        if col in block.columns:
            m = pd.to_numeric(block[col], errors="coerce") == 0
            r = block.loc[m, ["Player"]].dropna().drop_duplicates()
            return r if len(r) else None

    # 3) If you already have a boolean flag
    for col in ("is_rookie", "rookie", "rookie_flag"):
        if col in block.columns:
            m = block[col].astype(bool)
            r = block.loc[m, ["Player"]].dropna().drop_duplicates()
            return r if len(r) else None

    return None


def _should_skip_season_for_award(award: str, y: int, hist_before: pd.DataFrame) -> bool:
    # MIP requires prev season context; first year (or any year with empty history) cannot produce deltas meaningfully.
    if award.lower() == "mip" and (hist_before is None or len(hist_before) == 0):
        return True
    return False



def _should_skip_season_for_award(award: str, year: int, hist_before: pd.DataFrame) -> bool:
    """
    Skip seasons that cannot produce valid samples for a given award.
    - MIP needs at least one previous season to compute prev_/delta features.
    """
    a = award.lower().strip()
    if a == "mip":
        return hist_before.empty
    return False


def _infer_rookies_for_year(block: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Build a rookies dataframe (at least ['Player']) for ROY filtering.
    Priority:
      1) use 'is_rookie' flag if present
      2) otherwise return None (caller will skip that season)
    """
    if "Player" not in block.columns:
        return None

    if "is_rookie" in block.columns:
        m = pd.to_numeric(block["is_rookie"], errors="coerce").fillna(0).astype(int) == 1
        rook = block.loc[m, ["Player"]].dropna().drop_duplicates().reset_index(drop=True)
        return rook if len(rook) else None

    return None


def _build_training_table(df_hist: pd.DataFrame, award: str, feature_set: str) -> TrainingTable:
    label_col = _find_label_col(df_hist, award)

    # ---------------------------
    # Normalize season/year
    # ---------------------------
    df_hist = df_hist.copy()

    if "season" not in df_hist.columns and "year" in df_hist.columns:
        df_hist["season"] = df_hist["year"]

    # Compute end-year int
    df_hist["year"] = df_hist["season"].apply(_infer_year_end)
    df_hist = df_hist[~pd.isna(df_hist["year"])].copy()
    df_hist["year"] = df_hist["year"].astype(int)

    # >>> CRITICAL: make season match build_target_matrix expectations
    # build_target_matrix filters on "season" == target_year
    df_hist["season"] = df_hist["year"].astype(int)

    years = sorted(df_hist["year"].unique().tolist())
    if len(years) < 10:
        raise ValueError(f"Not enough seasons in history dataset: {len(years)}")

    X_blocks: list[pd.DataFrame] = []
    y_blocks: list[pd.Series] = []
    m_blocks: list[pd.DataFrame] = []

    reasons = {"skip_rule": 0, "no_rookies": 0, "empty_X": 0, "ok": 0}

    for y in years:
        block = df_hist[df_hist["year"] == y].copy()

        # stable row id for label recovery after eligibility filtering
        block["__row_id"] = block.index.astype(int)

        # ensure block season is end-year int
        block["season"] = int(y)

        hist_before = df_hist[df_hist["year"] < y].copy()
        hist_before["season"] = hist_before["year"].astype(int)

        # ---- Skip rules (MIP needs at least one previous season in hist_before) ----
        if award.lower() == "mip":
            if hist_before.empty:
                reasons["skip_rule"] += 1
                continue

        # ---- ROY rookies ----
        rookies = None
        if award.lower() == "roy":
            rookies = _infer_rookies_for_year(block)  # must return df with at least Player col (or whatever your eligibility expects)
            if rookies is None or len(rookies) == 0:
                reasons["no_rookies"] += 1
                continue

        bundle = build_target_matrix(
            df_hist=hist_before,
            df_target=block,
            target_year=int(y),
            award=award,
            feature_set=feature_set,
            rookies=rookies,
        )

        # If eligibility / delta kills all rows, skip season
        if bundle.X is None or len(bundle.X) == 0:
            reasons["empty_X"] += 1
            continue

        # REQUIRE __row_id in meta to recover labels
        if "__row_id" not in bundle.meta.columns:
            raise RuntimeError(
                "build_target_matrix() must include '__row_id' in meta when present in df_target."
            )

        row_ids = pd.to_numeric(bundle.meta["__row_id"], errors="coerce")
        if row_ids.isna().any():
            raise RuntimeError(f"Found NA in __row_id for award={award}, year={y}")

        row_ids = row_ids.astype(int).to_numpy()
        y_season = _as_binary(df_hist.loc[row_ids, label_col]).reset_index(drop=True)

        X_season = bundle.X.reset_index(drop=True)
        meta_season = bundle.meta.reset_index(drop=True).copy()
        meta_season["year"] = int(y)

        if len(X_season) != len(y_season) or len(X_season) != len(meta_season):
            raise RuntimeError(
                f"Alignment error for award={award}, year={y}: "
                f"X={len(X_season)} y={len(y_season)} meta={len(meta_season)}"
            )

        X_blocks.append(X_season)
        y_blocks.append(y_season)
        m_blocks.append(meta_season)
        reasons["ok"] += 1

    if len(X_blocks) == 0:
        raise ValueError(
            f"No training seasons produced any samples for award='{award}'. Reasons: {reasons}. "
            "Eligibility may be too strict, rookies inference missing (ROY), or delta/prev requirements not met (MIP)."
        )

    X_all = pd.concat(X_blocks, axis=0, ignore_index=True)
    y_all = pd.concat(y_blocks, axis=0, ignore_index=True)
    meta_all = pd.concat(m_blocks, axis=0, ignore_index=True)

    y_all = _as_binary(y_all)

    meta_all["year"] = pd.to_numeric(meta_all["year"], errors="coerce")
    keep = ~pd.isna(meta_all["year"])
    X_all = X_all.loc[keep].reset_index(drop=True)
    y_all = y_all.loc[keep].reset_index(drop=True)
    meta_all = meta_all.loc[keep].reset_index(drop=True)
    meta_all["year"] = meta_all["year"].astype(int)

    return TrainingTable(X=X_all, y=y_all, meta=meta_all)



# ============================================================
# Train loop (time split)
# ============================================================

def train_all(models_dir: Path, val_years: int = 2, test_years: int = 3) -> str:
    df_hist = pd.read_parquet(HIST_ENRICHED_PARQUET)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = models_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    families = [("baseline", "baseline"), ("tree", "tree")]

    print("\n==============================")
    print("🏋️  TRAIN AWARDS MODELS")
    print("==============================")
    print("Data:  ", str(HIST_ENRICHED_PARQUET))
    print("Output:", str(models_dir.resolve()))
    print("Run id:", run_id)
    print(f"Split:  val_years={val_years}, test_years={test_years}")
    print("Awards:", AWARDS)
    print("Families:", [f for f, _ in families])
    print()

    from sklearn.metrics import roc_auc_score, log_loss

    for award in AWARDS:
        for family, feature_set in families:
            print(f"--- Training {award} ({family}) ---")

            table = _build_training_table(df_hist, award=award, feature_set=feature_set)
            X_all, y_all, meta = table.X, table.y, table.meta

            years = sorted(meta["year"].unique().tolist())
            test_cut = years[-test_years:]
            val_cut = years[-(test_years + val_years):-test_years]

            is_test = meta["year"].isin(test_cut).to_numpy()
            is_val = meta["year"].isin(val_cut).to_numpy()
            is_train = ~(is_test | is_val)

            if len(is_train) != len(X_all) or len(y_all) != len(X_all):
                raise RuntimeError(
                    f"Mask/array mismatch: X={len(X_all)}, y={len(y_all)}, mask={len(is_train)} "
                    f"(award={award}, family={family})"
                )

            X_tr, y_tr = X_all.loc[is_train], y_all.loc[is_train]
            X_va, y_va = X_all.loc[is_val], y_all.loc[is_val]
            X_te, y_te = X_all.loc[is_test], y_all.loc[is_test]

            model = _make_baseline_model() if family == "baseline" else _make_tree_model()

            if family == "tree":
                try:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                except TypeError:
                    model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr)

            p_va = _predict_proba(model, X_va)
            p_te = _predict_proba(model, X_te)

            def _safe_auc(y, p):
                try:
                    return float(roc_auc_score(y, p))
                except Exception:
                    return float("nan")

            metrics = {
                "award": award,
                "family": family,
                "feature_set": feature_set,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
                "val_auc": _safe_auc(y_va, p_va),
                "test_auc": _safe_auc(y_te, p_te),
                "val_logloss": float(log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6))),
                "test_logloss": float(log_loss(y_te, np.clip(p_te, 1e-6, 1 - 1e-6))),
                "years": {
                    "train_min": int(min(years)),
                    "train_max": int(max(years)),
                    "val_years": [int(x) for x in val_cut],
                    "test_years": [int(x) for x in test_cut],
                },
            }

            out_dir = out_root / family / award
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                import joblib  # type: ignore
                joblib.dump(model, out_dir / "model.joblib")
            except Exception as e:
                raise RuntimeError(f"Failed to save sklearn model: {e}")

            try:
                import xgboost as xgb  # type: ignore
                if isinstance(model, xgb.XGBClassifier):
                    model.save_model(str(out_dir / "xgb_model.json"))
            except Exception:
                pass

            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"   ✓ saved to {out_dir} | val_auc={metrics['val_auc']:.3f} test_auc={metrics['test_auc']:.3f}")

    print("\n✅ Training completed. Run:", run_id)
    return run_id


def main() -> int:
    train_all(models_dir=Path("models"), val_years=2, test_years=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
