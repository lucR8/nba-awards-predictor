from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import *  # noqa: F401,F403  (adds src/ to sys.path)

import warnings

# Silence sklearn 1.8+ FutureWarning about LogisticRegression(penalty=...)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*'penalty' was deprecated in version 1\.8.*",
)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

from awards_predictor.config import HIST_ENRICHED_PARQUET
from awards_predictor.train.train_awards import _build_training_table


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} in df. Available={list(df.columns)}")


def _ensure_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Force all columns to numeric. Non-numeric becomes NaN -> then imputer handles it.
    Prevents pandas/sklearn transform crashes.
    """
    if not isinstance(X, pd.DataFrame):
        return X
    Xn = X.copy()
    for c in Xn.columns:
        if not pd.api.types.is_numeric_dtype(Xn[c]):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn


def temporal_split(meta: pd.DataFrame, *, val_years: int, test_years: int) -> pd.DataFrame:
    years = sorted(pd.Series(meta["year"]).dropna().astype(int).unique().tolist())
    if len(years) < (val_years + test_years + 1):
        raise ValueError(
            f"Not enough seasons to split: have {len(years)} years, "
            f"need at least {val_years + test_years + 1}."
        )

    test_cut = years[-test_years:]
    val_cut = years[-(test_years + val_years) : -test_years]

    meta = meta.copy()
    meta["split"] = "train"
    meta.loc[meta["year"].isin(val_cut), "split"] = "val"
    meta.loc[meta["year"].isin(test_cut), "split"] = "test"
    return meta


def make_model(model_name: str, seed: int = 42):
    imputer = SimpleImputer(strategy="median")

    if model_name == "logreg_l2":
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler(with_mean=False)), ("model", clf)]
        )

    if model_name == "logreg_elasticnet":
        clf = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            C=1.0,
            solver="saga",
            max_iter=4000,
            tol=1e-3,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler(with_mean=False)), ("model", clf)]
        )
        
    if model_name == "logreg_l1":
        # L1 via elasticnet API (sklearn >= 1.8 friendly)
        clf = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=1.0,          # == L1
            C=1.0,
            solver="saga",
            max_iter=1500,         # baisse pour éviter les runs interminables
            tol=1e-2,              # tol plus large = converge plus vite
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler(with_mean=False)), ("model", clf)]
        )       



    if model_name == "linear_svm_calibrated":
        base = LinearSVC(C=1.0, random_state=seed)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler(with_mean=False)), ("model", clf)]
        )

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("model", clf)])

    raise ValueError(f"Unknown model_name={model_name}")


def _pick_time_col(meta: pd.DataFrame) -> str:
    for c in ["year", "season_end_year", "season", "season_end"]:
        if c in meta.columns:
            return c
    raise ValueError(
        "No time column found in meta. Expected one of: year, season_end_year, season, season_end"
    )


def _as_pandas(X, y, meta: pd.DataFrame):
    """Force X/y to be pandas aligned with meta index."""
    meta = meta.reset_index(drop=True)

    if isinstance(X, pd.DataFrame):
        Xp = X.reset_index(drop=True)
    else:
        Xp = pd.DataFrame(np.asarray(X))

    if isinstance(y, pd.Series):
        yp = y.reset_index(drop=True)
    else:
        yp = pd.Series(np.asarray(y).ravel())

    n = min(len(meta), len(Xp), len(yp))
    meta = meta.iloc[:n].reset_index(drop=True)
    Xp = Xp.iloc[:n].reset_index(drop=True)
    yp = yp.iloc[:n].reset_index(drop=True)

    return Xp, yp, meta


def _score(model, X_: pd.DataFrame) -> np.ndarray:
    # Works with Pipeline too.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_)
    return model.predict(X_)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--award", required=True, type=str)
    ap.add_argument(
        "--model",
        required=True,
        choices=["logreg_l2", "logreg_l1", "logreg_elasticnet", "linear_svm_calibrated", "rf"]
    )
    ap.add_argument("--feature-set", default="baseline", type=str)
    ap.add_argument("--out-dir", default="reports/model_eval/learning_curves", type=str)
    ap.add_argument("--fracs", default="0.2,0.4,0.6,0.8,1.0", type=str)
    ap.add_argument("--seed", default=42, type=int)

    ap.add_argument("--val-years", type=int, default=2)
    ap.add_argument("--test-years", type=int, default=3)

    ap.add_argument(
        "--min-pos",
        type=int,
        default=5,
        help="Minimum number of positives required in a train subset.",
    )

    # ✅ NEW: multi-run averaging for report-quality curves
    ap.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of random seeds per learning-curve point (mean ± std).",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(HIST_ENRICHED_PARQUET)

    table = _build_training_table(df, award=args.award, feature_set=args.feature_set)
    X, y, meta = table.X, table.y, table.meta

    if not isinstance(meta, pd.DataFrame):
        raise ValueError("Expected meta to be a pandas DataFrame.")

    X, y, meta = _as_pandas(X, y, meta)

    # Ensure we have meta["year"] and meta["split"]
    if "split" not in meta.columns:
        year_col = _pick_col(meta, ["year", "season_end_year", "end_year"])
        meta["year"] = pd.to_numeric(meta[year_col], errors="coerce")
        meta = meta.dropna(subset=["year"]).copy()
        meta["year"] = meta["year"].astype(int).reset_index(drop=True)

        # Re-align X/y after dropping NA years
        X = X.loc[meta.index].reset_index(drop=True)
        y = y.loc[meta.index].reset_index(drop=True)

        meta = temporal_split(meta, val_years=args.val_years, test_years=args.test_years)

    time_col = _pick_time_col(meta)

    m_train = meta["split"].astype(str).str.lower().eq("train")
    m_val = meta["split"].astype(str).str.lower().eq("val")

    X_train_all = X.loc[m_train].reset_index(drop=True)
    y_train_all = y.loc[m_train].reset_index(drop=True)
    meta_train = meta.loc[m_train].reset_index(drop=True)

    X_val = X.loc[m_val].reset_index(drop=True)
    y_val = y.loc[m_val].reset_index(drop=True)

    seasons = np.array(sorted(pd.Series(meta_train[time_col]).dropna().astype(int).unique()))
    if len(seasons) < 3:
        raise ValueError(f"Not enough TRAIN seasons to build learning curve (found {len(seasons)}).")

    fracs = [float(x) for x in args.fracs.split(",")]
    rows: list[dict] = []

    # pre-coerce val once (same columns/order)
    X_val_num = _ensure_numeric_df(X_val)

    for frac in fracs:
        k = max(1, int(np.ceil(frac * len(seasons))))
        sel_seasons = seasons[:k]
        m_sel = pd.Series(meta_train[time_col]).astype(int).isin(sel_seasons).to_numpy()

        # build subset
        X_tr = X_train_all.loc[m_sel].reset_index(drop=True)
        y_tr = y_train_all.loc[m_sel].reset_index(drop=True)

        # numeric coercion (strings -> NaN) then imputer in pipeline
        X_tr = _ensure_numeric_df(X_tr)

        # guards: need both classes + enough positives
        y_arr = np.asarray(y_tr).ravel()
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())

        base_row = dict(
            award=args.award,
            model=args.model,
            feature_set=args.feature_set,
            frac=float(frac),
            n_seasons=int(k),
            seasons_max=int(sel_seasons.max()),
            n_train_rows=int(X_tr.shape[0]),
            n_pos=n_pos,
            n_neg=n_neg,
        )

        if n_pos == 0 or n_neg == 0:
            print(
                f"[SKIP] award={args.award} frac={frac:.2f} seasons<= {int(sel_seasons.max())} "
                f"-> single class (pos={n_pos}, neg={n_neg})"
            )
            rows.append(
                dict(
                    **base_row,
                    train_auc=np.nan,
                    val_auc=np.nan,
                    train_auc_mean=np.nan,
                    train_auc_std=np.nan,
                    val_auc_mean=np.nan,
                    val_auc_std=np.nan,
                    n_runs=int(args.n_runs),
                    status="skip_single_class",
                )
            )
            continue

        if n_pos < int(args.min_pos):
            print(
                f"[SKIP] award={args.award} frac={frac:.2f} seasons<= {int(sel_seasons.max())} "
                f"-> too few positives (pos={n_pos}, neg={n_neg})"
            )
            rows.append(
                dict(
                    **base_row,
                    train_auc=np.nan,
                    val_auc=np.nan,
                    train_auc_mean=np.nan,
                    train_auc_std=np.nan,
                    val_auc_mean=np.nan,
                    val_auc_std=np.nan,
                    n_runs=int(args.n_runs),
                    status="skip_too_few_pos",
                )
            )
            continue

        train_aucs: list[float] = []
        val_aucs: list[float] = []

        try:
            for r in range(int(args.n_runs)):
                seed_r = int(args.seed) + r
                model = make_model(args.model, seed=seed_r)
                model.fit(X_tr, y_tr)

                s_tr = _score(model, X_tr)
                s_val = _score(model, X_val_num)

                train_aucs.append(float(roc_auc_score(y_tr, s_tr)))
                val_aucs.append(float(roc_auc_score(y_val, s_val)))

            train_mean = float(np.mean(train_aucs))
            val_mean = float(np.mean(val_aucs))
            train_std = float(np.std(train_aucs, ddof=1)) if len(train_aucs) > 1 else 0.0
            val_std = float(np.std(val_aucs, ddof=1)) if len(val_aucs) > 1 else 0.0

            rows.append(
                dict(
                    **base_row,
                    # legacy cols (so old plotters still work)
                    train_auc=train_mean,
                    val_auc=val_mean,
                    # report-ready stats
                    train_auc_mean=train_mean,
                    train_auc_std=train_std,
                    val_auc_mean=val_mean,
                    val_auc_std=val_std,
                    n_runs=int(args.n_runs),
                    status="ok",
                )
            )

        except Exception as e:
            print(
                f"[FAIL] award={args.award} frac={frac:.2f} seasons<= {int(sel_seasons.max())} "
                f"failed: {type(e).__name__}: {e}"
            )
            rows.append(
                dict(
                    **base_row,
                    train_auc=np.nan,
                    val_auc=np.nan,
                    train_auc_mean=np.nan,
                    train_auc_std=np.nan,
                    val_auc_mean=np.nan,
                    val_auc_std=np.nan,
                    n_runs=int(args.n_runs),
                    status=f"fail_{type(e).__name__}",
                )
            )
            continue

    out_csv = out_dir / f"learning_curve_{args.award}_{args.model}_{args.feature_set}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
