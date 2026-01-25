from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from awards_predictor.config import AWARDS, HIST_ENRICHED_PARQUET
from awards_predictor.models.registry import get_model_specs
from awards_predictor.evaluation.ranking_metrics import season_ranking_report
from awards_predictor.evaluation.explain import (
    explain_linear_coefficients,
    explain_permutation_importance,
)

from awards_predictor.train.train_awards import _build_training_table


# -----------------------------
# Speed knobs (defendable)
# -----------------------------
PERM_REPEATS = 2
VAL_SUBSAMPLE = 3000
TOP_N_FEATURES = 25


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Available={list(df.columns)}")


def _get_feature_names(table, X) -> list[str]:
    for attr in ("feature_names", "feature_cols", "features", "columns"):
        v = getattr(table, attr, None)
        if v is not None:
            try:
                v_list = list(v)
                if len(v_list) > 0:
                    return v_list
            except Exception:
                pass
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    if hasattr(X, "shape") and len(X.shape) == 2:
        return [f"f{i}" for i in range(int(X.shape[1]))]
    return []


def _replace_inf_with_nan(X):
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf], np.nan)
    arr = np.asarray(X).copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _needs_scaling(estimator) -> bool:
    name = estimator.__class__.__name__.lower()
    return ("logistic" in name) or ("linear" in name) or ("svm" in name)


def _wrap_with_preprocess(estimator) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if _needs_scaling(estimator):
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _tune_for_eval(estimator):
    cls = estimator.__class__.__name__.lower()
    if "logistic" in cls:
        try:
            params = estimator.get_params()
            if "max_iter" in params:
                estimator.set_params(max_iter=min(int(params["max_iter"]), 1500))
        except Exception:
            pass
    return estimator


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


def _safe_auc(y_true, scores) -> float:
    y_s = pd.Series(y_true).dropna().astype(int)
    if y_s.nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_s.values, np.asarray(scores)))


def _subsample_for_perm(X, y, max_n: int, seed: int = 42):
    n = int(getattr(X, "shape")[0])
    if n <= max_n:
        return X, y

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_n, replace=False)

    if isinstance(X, pd.DataFrame):
        X2 = X.iloc[idx]
    else:
        X2 = np.asarray(X)[idx]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y2 = y.iloc[idx]
    else:
        y2 = np.asarray(y)[idx]

    return X2, y2


def _build_table_for_award(df: pd.DataFrame, award: str):
    """
    IMPORTANT:
    - MIP doit utiliser le feature_set qui inclut pct + deltas (si votre pipeline le prévoit).
    - fallback safe sur baseline si "mip" n'existe pas.
    """
    if award == "mip":
        try:
            return _build_training_table(df, award=award, feature_set="mip")
        except TypeError:
            # si signature différente
            return _build_training_table(df, award=award, feature_set="baseline")
        except Exception:
            return _build_training_table(df, award=award, feature_set="baseline")

    return _build_training_table(df, award=award, feature_set="baseline")


def main(out_dir: str = "reports/model_eval", val_years: int = 2, test_years: int = 3) -> int:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[EVAL] loading data: {HIST_ENRICHED_PARQUET}")
    df = pd.read_parquet(HIST_ENRICHED_PARQUET)
    specs = get_model_specs()

    rows: list[dict] = []

    for award in AWARDS:
        print(f"\n[EVAL] ===== award={award} =====")
        table = _build_table_for_award(df, award)

        X = table.X
        y = table.y
        meta = table.meta.copy()

        # standardize metadata
        year_col = _pick_col(meta, ["year", "season_end_year", "end_year"])
        season_col = _pick_col(meta, ["season", "season_str", "season_id", "season_year"])

        meta["year"] = pd.to_numeric(meta[year_col], errors="coerce").astype("Int64")
        meta["season"] = meta[season_col].astype(str)

        # drop bad meta rows
        meta = meta.dropna(subset=["year", "season"]).copy()
        meta["year"] = meta["year"].astype(int)

        # Align X/y to meta rows by INDEX (not by reset blindly)
        if isinstance(X, pd.DataFrame):
            X = X.loc[meta.index]
        else:
            X = np.asarray(X)[meta.index.to_numpy()]

        if isinstance(y, pd.Series):
            y = y.loc[meta.index]
        else:
            y = np.asarray(y)[meta.index.to_numpy()]

        # Now reset everything to 0..n-1 to simplify slicing
        meta = meta.reset_index(drop=True)
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
        else:
            X = np.asarray(X)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)
        else:
            y = np.asarray(y)

        feat_names = _get_feature_names(table, X)

        meta = temporal_split(meta, val_years=val_years, test_years=test_years)
        m_train = meta["split"] == "train"
        m_val = meta["split"] == "val"
        m_test = meta["split"] == "test"

        # slice
        if isinstance(X, pd.DataFrame):
            X_train, X_val, X_test = X.loc[m_train], X.loc[m_val], X.loc[m_test]
        else:
            X_train, X_val, X_test = X[m_train.values], X[m_val.values], X[m_test.values]

        if isinstance(y, pd.Series):
            y_train, y_val, y_test = y.loc[m_train], y.loc[m_val], y.loc[m_test]
        else:
            y_train, y_val, y_test = y[m_train.values], y[m_val.values], y[m_test.values]

        X_train_ = _replace_inf_with_nan(X_train)
        X_val_ = _replace_inf_with_nan(X_val)
        X_test_ = _replace_inf_with_nan(X_test)

        print(f"[EVAL] shapes: train={getattr(X_train_, 'shape', None)} val={getattr(X_val_, 'shape', None)} test={getattr(X_test_, 'shape', None)}")

        for name, spec in specs.items():
            print(f"[EVAL] -> model={name} (family={spec.family}) fit...")
            base = _tune_for_eval(spec.builder())
            model = _wrap_with_preprocess(base)

            model.fit(X_train_, y_train)

            print(f"[EVAL] -> model={name} scoring...")
            if hasattr(model, "predict_proba"):
                s_val = model.predict_proba(X_val_)[:, 1]
                s_test = model.predict_proba(X_test_)[:, 1]
            else:
                s_val = model.decision_function(X_val_)
                s_test = model.decision_function(X_test_)

            auc_val = _safe_auc(y_val, s_val)
            auc_test = _safe_auc(y_test, s_test)

            df_val = meta.loc[m_val, ["season"]].copy()
            df_val["score"] = s_val
            df_val["y"] = pd.Series(y_val).to_numpy()
            r_val = season_ranking_report(df_val, season_col="season", score_col="score", label_col="y")

            df_test = meta.loc[m_test, ["season"]].copy()
            df_test["score"] = s_test
            df_test["y"] = pd.Series(y_test).to_numpy()
            r_test = season_ranking_report(df_test, season_col="season", score_col="score", label_col="y")

            rows.append(
                {
                    "award": award,
                    "model": name,
                    "family": spec.family,
                    "val_auc": auc_val,
                    "test_auc": auc_test,
                    "val_hit@1": r_val.hit_at_1,
                    "val_hit@5": r_val.hit_at_5,
                    "val_mrr": r_val.mrr,
                    "test_hit@1": r_test.hit_at_1,
                    "test_hit@5": r_test.hit_at_5,
                    "test_mrr": r_test.mrr,
                    "test_mean_winner_rank": r_test.mean_winner_rank,
                    "n_test_seasons": r_test.n_seasons,
                }
            )

            # explain (heavy)
            try:
                print(f"[EVAL] -> model={name} explain...")
                inner = model.named_steps.get("model", model)

                if hasattr(inner, "coef_"):
                    df_imp = explain_linear_coefficients(inner, feat_names, top_n=TOP_N_FEATURES)
                    df_imp.to_csv(out / f"top_features_{award}_{name}.csv", index=False)
                else:
                    X_pi, y_pi = _subsample_for_perm(X_val_, y_val, max_n=VAL_SUBSAMPLE, seed=42)
                    df_imp = explain_permutation_importance(
                        model,
                        X_pi,
                        y_pi,
                        feat_names,
                        top_n=TOP_N_FEATURES,
                        n_repeats=PERM_REPEATS,
                    )
                    df_imp.to_csv(out / f"top_features_{award}_{name}.csv", index=False)

            except Exception as e:
                print(f"[EVAL]    (skip explain) model={name} reason={type(e).__name__}: {e}")

    df_metrics = pd.DataFrame(rows).sort_values(["award", "test_mrr"], ascending=[True, False])
    out_file = out / "metrics_by_award.csv"
    df_metrics.to_csv(out_file, index=False)
    print(f"\n[OK] wrote {out_file}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="reports/model_eval")
    parser.add_argument("--val-years", type=int, default=2)
    parser.add_argument("--test-years", type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(main(out_dir=args.out_dir, val_years=args.val_years, test_years=args.test_years))
