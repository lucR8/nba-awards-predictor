from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import *  # noqa: F401,F403  (adds src/ to sys.path)

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from awards_predictor.config import HIST_ENRICHED_PARQUET
from awards_predictor.train.train_awards import _build_training_table


# -----------------------------
# Helpers
# -----------------------------
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} in df. Available={list(df.columns)}")


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


def _as_pandas(X, y, meta: pd.DataFrame):
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
    return (
        Xp.iloc[:n].reset_index(drop=True),
        yp.iloc[:n].reset_index(drop=True),
        meta.iloc[:n].reset_index(drop=True),
    )


def _ensure_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        return X
    Xn = X.copy()
    for c in Xn.columns:
        if not pd.api.types.is_numeric_dtype(Xn[c]):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn


def _score(model, X_: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_)[:, 1]


def make_logreg_pipeline(
    *,
    C: float,
    penalty: str,
    l1_ratio: float | None,
    seed: int,
    max_iter: int,
    tol: float,
) -> Pipeline:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=False)

    if penalty == "l2":
        # NOTE: sklearn 1.8 warns about penalty; we avoid passing it.
        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            tol=tol,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("scaler", scaler), ("model", clf)])

    if penalty == "l1":
        # Faster & stable for L1 in many tabular cases
        clf = LogisticRegression(
            C=C,
            penalty="l1",
            solver="liblinear",
            max_iter=max_iter,
            tol=tol,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("scaler", scaler), ("model", clf)])

    if penalty == "elasticnet":
        if l1_ratio is None:
            raise ValueError("penalty=elasticnet requires --l1-ratio")

        clf = LogisticRegression(
            C=C,
            penalty="elasticnet",
            l1_ratio=float(l1_ratio),
            solver="saga",
            max_iter=max_iter,
            tol=tol,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("scaler", scaler), ("model", clf)])

    raise ValueError(f"Unknown penalty={penalty}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--award", required=True, type=str)
    ap.add_argument("--feature-set", default="baseline", type=str)

    ap.add_argument("--penalty", choices=["l2", "l1", "elasticnet"], default="l2")

    # ONE single definition (no conflict)
    ap.add_argument(
        "--l1-ratio",
        type=float,
        default=None,
        help="Only used if --penalty elasticnet (e.g. 0.5).",
    )

    ap.add_argument("--Cs", default="0.01,0.1,1,10", type=str, help="Comma-separated C values")

    ap.add_argument("--val-years", type=int, default=2)
    ap.add_argument("--test-years", type=int, default=3)

    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-iter", type=int, default=800)
    ap.add_argument("--tol", type=float, default=1e-2)

    ap.add_argument("--out-dir", default="reports/model_eval/hparam_sweep", type=str)
    ap.add_argument("--out-name", default="", type=str, help="Optional custom prefix")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(HIST_ENRICHED_PARQUET)
    table = _build_training_table(df, award=args.award, feature_set=args.feature_set)
    X, y, meta = table.X, table.y, table.meta

    if not isinstance(meta, pd.DataFrame):
        raise ValueError("Expected meta to be a pandas DataFrame.")

    X, y, meta = _as_pandas(X, y, meta)

    if "split" not in meta.columns:
        year_col = _pick_col(meta, ["year", "season_end_year", "end_year"])
        meta["year"] = pd.to_numeric(meta[year_col], errors="coerce")
        meta = meta.dropna(subset=["year"]).copy()
        meta["year"] = meta["year"].astype(int).reset_index(drop=True)

        X = X.loc[meta.index].reset_index(drop=True)
        y = y.loc[meta.index].reset_index(drop=True)

        meta = temporal_split(meta, val_years=args.val_years, test_years=args.test_years)

    m_train = meta["split"].astype(str).str.lower().eq("train")
    m_val = meta["split"].astype(str).str.lower().eq("val")

    X_train = _ensure_numeric_df(X.loc[m_train].reset_index(drop=True))
    y_train = y.loc[m_train].reset_index(drop=True)

    X_val = _ensure_numeric_df(X.loc[m_val].reset_index(drop=True))
    y_val = y.loc[m_val].reset_index(drop=True)

    y_arr = np.asarray(y_train).ravel()
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())

    if n_neg == 0:
        raise ValueError(
            f"Train split has a single class (pos={n_pos}, neg={n_neg}). "
            f"Increase training seasons (val/test years) or check labels."
        )
    if n_pos < int(args.min_pos):
        raise ValueError(
            f"Not enough positives in TRAIN (pos={n_pos}, neg={n_neg}, min_pos={args.min_pos}). "
            f"Lower --min-pos or expand training seasons."
        )

    Cs = [float(x) for x in args.Cs.split(",") if x.strip()]

    # IMPORTANT: prevent l1_ratio from leaking into l1/l2
    l1_ratio_arg = args.l1_ratio if args.penalty == "elasticnet" else None

    rows: list[dict] = []
    for C in Cs:
        train_aucs: list[float] = []
        val_aucs: list[float] = []

        for r in range(int(args.n_runs)):
            seed_r = int(args.seed) + r

            pipe = make_logreg_pipeline(
                C=C,
                penalty=args.penalty,
                l1_ratio=l1_ratio_arg,
                seed=seed_r,
                max_iter=args.max_iter,
                tol=args.tol,
            )

            pipe.fit(X_train, y_train)
            s_tr = _score(pipe, X_train)
            s_va = _score(pipe, X_val)

            train_aucs.append(float(roc_auc_score(y_train, s_tr)))
            val_aucs.append(float(roc_auc_score(y_val, s_va)))

        rows.append(
            dict(
                award=args.award,
                feature_set=args.feature_set,
                penalty=args.penalty,
                l1_ratio=(l1_ratio_arg if args.penalty == "elasticnet" else np.nan),
                C=C,
                n_runs=int(args.n_runs),
                train_auc_mean=float(np.mean(train_aucs)),
                train_auc_std=float(np.std(train_aucs, ddof=1)) if len(train_aucs) > 1 else 0.0,
                val_auc_mean=float(np.mean(val_aucs)),
                val_auc_std=float(np.std(val_aucs, ddof=1)) if len(val_aucs) > 1 else 0.0,
            )
        )

    out_prefix = args.out_name.strip() or f"{args.award}_{args.feature_set}_{args.penalty}"
    out_csv = out_dir / f"hparam_sweep_logreg_{out_prefix}.csv"
    pd.DataFrame(rows).sort_values("C").to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
