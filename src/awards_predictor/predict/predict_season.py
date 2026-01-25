# src/awards_predictor/predict/predict_season.py
from __future__ import annotations

# ============================================================
# Bootstrap
# ============================================================
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_DIR = _THIS_FILE.parents[2]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ============================================================

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import pandas as pd

from awards_predictor.config import AWARDS, HIST_ENRICHED_PARQUET
from awards_predictor.io.paths import target_paths, season_str
from awards_predictor.features.build_matrix import load_target_dataset, build_target_matrix


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    model_type: str
    feature_names: Optional[list[str]]


def _try_load_sklearn(run_dir: Path) -> Optional[LoadedModel]:
    try:
        import joblib  # type: ignore
    except Exception:
        return None
    p = run_dir / "model.joblib"
    if not p.exists():
        return None
    m = joblib.load(p)
    feat = getattr(m, "feature_names_in_", None)
    return LoadedModel(m, "sklearn", list(feat) if feat is not None else None)


def _try_load_xgboost(run_dir: Path) -> Optional[LoadedModel]:
    p = run_dir / "xgb_model.json"
    if not p.exists():
        return None
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        return None
    booster = xgb.Booster()
    booster.load_model(str(p))
    feat = booster.feature_names
    return LoadedModel(booster, "xgboost", list(feat) if feat else None)


def load_model(run_dir: Path) -> LoadedModel:
    for loader in (_try_load_sklearn, _try_load_xgboost):
        m = loader(run_dir)
        if m is not None:
            return m
    raise RuntimeError(f"No supported model file in: {run_dir}")


def align_X(X: pd.DataFrame, model: LoadedModel) -> pd.DataFrame:
    if not model.feature_names:
        return X
    cols = model.feature_names
    X2 = X.copy()
    for c in cols:
        if c not in X2.columns:
            X2[c] = 0.0
    return X2[cols]


def predict_proba(model: LoadedModel, X: pd.DataFrame) -> pd.Series:
    if model.model_type == "sklearn":
        proba = model.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index)

    if model.model_type == "xgboost":
        import xgboost as xgb  # type: ignore
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
        p = model.model.predict(dmat)
        return pd.Series(p, index=X.index)

    raise RuntimeError(f"Unsupported model_type={model.model_type}")


def latest_run_id(models_dir: Path) -> str:
    runs = [p for p in models_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found in {models_dir}")
    return sorted(runs, key=lambda p: p.name)[-1].name


def run_predict_season(year: int, topk: int = 10, models_dir: Path = Path("models")) -> Path:
    models_dir = models_dir if models_dir.is_absolute() else (REPO_ROOT / models_dir)
    run_id = latest_run_id(models_dir)

    # load data
    paths = target_paths(year=year, snapshot=None)
    df_target = load_target_dataset(paths)

    if not HIST_ENRICHED_PARQUET.exists():
        raise FileNotFoundError(f"Historical parquet not found: {HIST_ENRICHED_PARQUET}")
    df_hist = pd.read_parquet(HIST_ENRICHED_PARQUET)

    rookies_df = pd.read_csv(paths.raw_rookies) if paths.raw_rookies.exists() else None

    out_dir = paths.predictions
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    chosen = []

    for award in AWARDS:
        candidates = []
        for family, feature_set in (("baseline", "baseline"), ("tree", "tree")):
            run_dir = models_dir / run_id / family / award
            if not run_dir.exists():
                continue
            try:
                m = load_model(run_dir)
            except Exception:
                continue

            bundle = build_target_matrix(
                df_hist=df_hist,
                df_target=df_target,
                target_year=year,
                award=award,
                rookies=rookies_df,
                feature_set=feature_set,
                add_prev=True,
            )
            X = align_X(bundle.X, m)
            scores = predict_proba(m, X)

            res = bundle.meta.copy()
            res["award"] = award
            res["family"] = family
            res["score"] = scores.values
            res["rank"] = res["score"].rank(ascending=False, method="first").astype(int)
            res = res.sort_values("score", ascending=False).reset_index(drop=True)

            top1 = float(res.loc[0, "score"]) if len(res) else float("nan")
            candidates.append((top1, family, res))

        if not candidates:
            print(f"[WARN] no model available for {award} in run {run_id}")
            continue

        # pick the most confident model: highest top1 probability
        candidates.sort(key=lambda t: t[0], reverse=True)
        best_top1, best_family, best_res = candidates[0]
        chosen.append({"award": award, "family": best_family, "top1_score": best_top1})

        top = best_res.head(topk).copy()
        per_award_path = out_dir / f"{award}_top{topk}.csv"
        top.to_csv(per_award_path, index=False)

        all_rows.append(best_res)

        print(f"[OK] {award}: chose {best_family} (top1={best_top1:.4f})")

    if not all_rows:
        raise RuntimeError(f"No predictions produced (run_id={run_id}, models_dir={models_dir}).")

    all_df = pd.concat(all_rows, ignore_index=True)
    all_path = out_dir / f"all_awards_top{topk}.parquet"
    all_df.to_parquet(all_path, index=False)

    meta = {
        "year": year,
        "season": season_str(year),
        "snapshot": paths.root.name,
        "asof": paths.root.name.replace("asof_", ""),
        "built_on": str(date.today()),
        "models_dir": str(models_dir),
        "run_id": run_id,
        "topk": topk,
        "chosen": chosen,
        "exports": {
            "all_awards_parquet": str(all_path),
        },
    }
    (out_dir / "prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Prediction completed")
    print("Run:", run_id)
    print("Outputs:", out_dir)
    print("All awards:", all_path)
    return out_dir


def main() -> int:
    # zero mandatory args: default to 2026 / top5
    run_predict_season(year=2026, topk=5, models_dir=Path("models"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
