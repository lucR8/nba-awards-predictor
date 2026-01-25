from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import *  # noqa: F401,F403  (adds src/ to sys.path)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from awards_predictor.config import HIST_ENRICHED_PARQUET
from awards_predictor.train.train_awards import _build_training_table


# -----------------------------
# Split helpers (same spirit as learning_curves)
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


def _ensure_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """Force all columns to numeric; non-numeric -> NaN (imputer handles it)."""
    if not isinstance(X, pd.DataFrame):
        return X
    Xn = X.copy()
    for c in Xn.columns:
        if not pd.api.types.is_numeric_dtype(Xn[c]):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn


def _score(model, X_: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_)
    return model.predict(X_)


# -----------------------------
# Models (same API as learning_curves)
# -----------------------------
def make_model(model_name: str, seed: int = 42) -> Pipeline:
    imputer = SimpleImputer(strategy="median")

    if model_name == "logreg_l2":
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("scaler", StandardScaler(with_mean=False)), ("model", clf)])

    if model_name == "linear_svm_calibrated":
        base = LinearSVC(C=1.0, random_state=seed)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        return Pipeline([("imputer", imputer), ("scaler", StandardScaler(with_mean=False)), ("model", clf)])

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


# -----------------------------
# Feature groups (heuristics + audit)
# -----------------------------
@dataclass(frozen=True)
class GroupSpec:
    name: str
    patterns: list[str]  # regex patterns


DEFAULT_GROUPS: list[GroupSpec] = [
    GroupSpec(
        "bio",
        [
            r"(^|_)age($|_)",
            r"height",
            r"weight",
            r"birth",
            r"country",
            r"national",
            r"draft",
            r"experience",
            r"position",
        ],
    ),
    GroupSpec(
        "team_context",
        [
            r"team",
            r"tm_",
            r"seed",
            r"wins?",
            r"loss",
            r"conf",
            r"division",
            r"net_rtg",
            r"off_rtg",
            r"def_rtg",
        ],
    ),
    GroupSpec(
        "external_metrics",
        [
            r"raptor",
            r"lebron",
            r"mamba",
            r"epm",
            r"dpm",
            r"bpm",
        ],
    ),
    GroupSpec(
        "percentiles",
        [
            r"pctile",
            r"percentile",
            r"_pct$",
            r"_pctl$",
        ],
    ),
]


def match_group_columns(columns: list[str], spec: GroupSpec) -> list[str]:
    cols = []
    for col in columns:
        for pat in spec.patterns:
            if re.search(pat, col, flags=re.IGNORECASE):
                cols.append(col)
                break
    return sorted(set(cols))


def build_group_map(X: pd.DataFrame) -> dict[str, list[str]]:
    col_list = list(X.columns)
    out: dict[str, list[str]] = {}
    for spec in DEFAULT_GROUPS:
        out[spec.name] = match_group_columns(col_list, spec)
    return out


def drop_columns(X: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    if not cols_to_drop:
        return X
    existing = [c for c in cols_to_drop if c in X.columns]
    if not existing:
        return X
    return X.drop(columns=existing)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--award", required=True, type=str)
    ap.add_argument("--feature-set", default="baseline", type=str)

    ap.add_argument("--model", choices=["logreg_l2", "linear_svm_calibrated", "rf"], default="logreg_l2")

    ap.add_argument("--val-years", type=int, default=2)
    ap.add_argument("--test-years", type=int, default=3)

    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--ablations",
        default="bio,team_context,external_metrics,percentiles",
        type=str,
        help="Comma-separated group names to ablate (drop). Use 'none' to only run baseline.",
    )

    ap.add_argument("--out-dir", default="reports/model_eval/feature_ablation", type=str)
    ap.add_argument("--plot", action="store_true", help="Also write a barplot PNG (val AUC).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(HIST_ENRICHED_PARQUET)
    table = _build_training_table(df, award=args.award, feature_set=args.feature_set)
    X, y, meta = table.X, table.y, table.meta

    if not isinstance(meta, pd.DataFrame):
        raise ValueError("Expected meta to be a pandas DataFrame.")

    X, y, meta = _as_pandas(X, y, meta)
    X = _ensure_numeric_df(X)

    # Build split if missing
    if "split" not in meta.columns:
        year_col = _pick_col(meta, ["year", "season_end_year", "end_year"])
        meta["year"] = pd.to_numeric(meta[year_col], errors="coerce")
        meta = meta.dropna(subset=["year"]).copy()
        meta["year"] = meta["year"].astype(int).reset_index(drop=True)

        # Re-align X/y
        X = X.loc[meta.index].reset_index(drop=True)
        y = y.loc[meta.index].reset_index(drop=True)

        meta = temporal_split(meta, val_years=args.val_years, test_years=args.test_years)

    m_train = meta["split"].astype(str).str.lower().eq("train")
    m_val = meta["split"].astype(str).str.lower().eq("val")
    m_test = meta["split"].astype(str).str.lower().eq("test")

    X_train = X.loc[m_train].reset_index(drop=True)
    y_train = y.loc[m_train].reset_index(drop=True)

    X_val = X.loc[m_val].reset_index(drop=True)
    y_val = y.loc[m_val].reset_index(drop=True)

    X_test = X.loc[m_test].reset_index(drop=True)
    y_test = y.loc[m_test].reset_index(drop=True)

    # Guards
    y_arr = np.asarray(y_train).ravel()
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_neg == 0 or n_pos < int(args.min_pos):
        raise ValueError(f"Train split invalid (pos={n_pos}, neg={n_neg}, min_pos={args.min_pos}).")

    group_map = build_group_map(X_train)

    # Parse ablations
    ablations_raw = [s.strip() for s in args.ablations.split(",") if s.strip()]
    if len(ablations_raw) == 1 and ablations_raw[0].lower() == "none":
        ablation_names: list[str] = []
    else:
        ablation_names = ablations_raw

    # Always include baseline
    variants: list[tuple[str, list[str]]] = [("baseline", [])]

    # Add each ablation as "no_<group>"
    for g in ablation_names:
        if g not in group_map:
            available = ", ".join(sorted(group_map.keys()))
            raise ValueError(f"Unknown group '{g}'. Available: {available}")
        variants.append((f"no_{g}", group_map[g]))

    rows: list[dict] = []

    # Audit print
    print("[INFO] Feature groups (matched columns):")
    for g, cols in group_map.items():
        print(f"  - {g:15s}: {len(cols):4d}")

    for variant_name, drop_cols in variants:
        Xtr = drop_columns(X_train, drop_cols)
        Xva = drop_columns(X_val, drop_cols)
        Xte = drop_columns(X_test, drop_cols)

        train_aucs: list[float] = []
        val_aucs: list[float] = []
        test_aucs: list[float] = []

        for r in range(int(args.n_runs)):
            seed_r = int(args.seed) + r
            model = make_model(args.model, seed=seed_r)
            model.fit(Xtr, y_train)

            s_tr = _score(model, Xtr)
            s_va = _score(model, Xva)
            s_te = _score(model, Xte)

            train_aucs.append(float(roc_auc_score(y_train, s_tr)))
            val_aucs.append(float(roc_auc_score(y_val, s_va)))
            test_aucs.append(float(roc_auc_score(y_test, s_te)))

        def _mean_std(xs: list[float]) -> tuple[float, float]:
            m = float(np.mean(xs))
            s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
            return m, s

        tr_m, tr_s = _mean_std(train_aucs)
        va_m, va_s = _mean_std(val_aucs)
        te_m, te_s = _mean_std(test_aucs)

        rows.append(
            dict(
                award=args.award,
                feature_set=args.feature_set,
                model=args.model,
                variant=variant_name,
                dropped_cols=len(drop_cols),
                n_runs=int(args.n_runs),
                train_auc_mean=tr_m,
                train_auc_std=tr_s,
                val_auc_mean=va_m,
                val_auc_std=va_s,
                test_auc_mean=te_m,
                test_auc_std=te_s,
            )
        )

        print(f"[OK] {variant_name:18s}  val_auc={va_m:.4f}  test_auc={te_m:.4f}  (drop={len(drop_cols)})")

    out_csv = out_dir / f"feature_ablation_{args.award}_{args.model}_{args.feature_set}.csv"
    res = pd.DataFrame(rows).sort_values(["val_auc_mean"], ascending=False)
    res.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    if args.plot:
        # Barplot val AUC (with std if n_runs>1)
        import matplotlib.pyplot as plt

        # Sort by val
        res_plot = res.sort_values("val_auc_mean", ascending=True).reset_index(drop=True)
        ylabels = res_plot["variant"].tolist()
        vals = res_plot["val_auc_mean"].to_numpy()
        errs = res_plot["val_auc_std"].to_numpy()

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.barh(ylabels, vals, xerr=errs if np.any(errs > 0) else None)
        ax.set_xlabel("Validation AUC")
        ax.set_title(f"{args.award.upper()} — feature ablation ({args.model}, {args.feature_set})")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        out_png = out_dir / f"feature_ablation_{args.award}_{args.model}_{args.feature_set}.png"
        plt.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"[OK] wrote {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
