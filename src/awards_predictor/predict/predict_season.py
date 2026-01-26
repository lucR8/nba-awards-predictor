# src/awards_predictor/predict/predict_season.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from awards_predictor.config import AWARDS, HIST_ENRICHED_PARQUET
from awards_predictor.features.build_matrix import build_target_matrix, load_target_dataset
from awards_predictor.io.paths import season_str, target_paths


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    model_type: str
    feature_names: Optional[list[str]]


@dataclass(frozen=True)
class PretrainedRef:
    award: str
    family: str
    feature_set: str
    format: str
    model_path: Path


def load_pretrained_manifest(manifest_path: Path) -> dict[str, PretrainedRef]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, PretrainedRef] = {}
    for m in data.get("models", []):
        award = str(m["award"]).lower()
        out[award] = PretrainedRef(
            award=award,
            family=str(m.get("family", "baseline")).lower(),
            feature_set=str(m.get("feature_set", "baseline")).lower(),
            format=str(m.get("format", "joblib")).lower(),
            model_path=Path(m["model_path"]),
        )
    return out


def _try_load_sklearn(model_path: Path) -> Optional[LoadedModel]:
    try:
        import joblib  # type: ignore
    except Exception:
        return None
    if not model_path.exists():
        return None
    m = joblib.load(model_path)
    feat = getattr(m, "feature_names_in_", None)
    return LoadedModel(m, "sklearn", list(feat) if feat is not None else None)


def _try_load_xgboost(model_path: Path) -> Optional[LoadedModel]:
    if not model_path.exists():
        return None
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        return None
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    feat = booster.feature_names
    return LoadedModel(booster, "xgboost", list(feat) if feat else None)


def load_model_from_path(model_path: Path) -> LoadedModel:
    # support .joblib (sklearn pipeline) or xgb json
    if model_path.suffix.lower() == ".joblib":
        m = _try_load_sklearn(model_path)
        if m is None:
            raise RuntimeError(f"Failed to load sklearn joblib: {model_path}")
        return m

    if model_path.suffix.lower() == ".json":
        m = _try_load_xgboost(model_path)
        if m is None:
            raise RuntimeError(f"Failed to load xgboost json: {model_path}")
        return m

    raise RuntimeError(f"Unsupported model format: {model_path}")


def align_X(X: pd.DataFrame, model: LoadedModel) -> pd.DataFrame:
    if not model.feature_names:
        return X
    cols = model.feature_names
    X2 = X.copy()
    for c in cols:
        if c not in X2.columns:
            X2[c] = 0.0
    # drop extra cols to match expected order
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


def _resolve_snapshot_paths(year: int, snapshot_dir: Optional[Path]) -> Any:
    # target_paths(year, snapshot=None) already picks latest snapshot internally in your project
    if snapshot_dir is None:
        return target_paths(year=year, snapshot=None)
    snapshot_dir = snapshot_dir if snapshot_dir.is_absolute() else (REPO_ROOT / snapshot_dir)
    return target_paths(year=year, snapshot=snapshot_dir.name)


def run_predict_season(
    year: int,
    topk: int = 5,
    models_dir: Path = Path("models"),
    pretrained_manifest: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
) -> Path:
    models_dir = models_dir if models_dir.is_absolute() else (REPO_ROOT / models_dir)

    # snapshot
    paths = _resolve_snapshot_paths(year=year, snapshot_dir=snapshot_dir)

    # load target + hist
    df_target = load_target_dataset(paths)

    if not HIST_ENRICHED_PARQUET.exists():
        raise FileNotFoundError(f"Historical parquet not found: {HIST_ENRICHED_PARQUET}")
    df_hist = pd.read_parquet(HIST_ENRICHED_PARQUET)

    rookies_df = pd.read_csv(paths.raw_rookies) if paths.raw_rookies.exists() else None

    out_dir = paths.predictions
    out_dir.mkdir(parents=True, exist_ok=True)

    # model resolution
    pretrained: Optional[dict[str, PretrainedRef]] = None
    if pretrained_manifest is not None:
        pretrained_manifest = pretrained_manifest if pretrained_manifest.is_absolute() else (REPO_ROOT / pretrained_manifest)
        pretrained = load_pretrained_manifest(pretrained_manifest)

    run_id = None
    if pretrained is None:
        run_id = latest_run_id(models_dir)

    all_rows: list[pd.DataFrame] = []
    chosen: list[dict[str, Any]] = []

    for award in AWARDS:
        award_l = award.lower().strip()

        if pretrained is not None:
            if award_l not in pretrained:
                print(f"[WARN] no pretrained entry for award={award_l}")
                continue

            ref = pretrained[award_l]
            model_path = ref.model_path if ref.model_path.is_absolute() else (REPO_ROOT / ref.model_path)
            m = load_model_from_path(model_path)

            bundle = build_target_matrix(
                df_hist=df_hist,
                df_target=df_target,
                target_year=year,
                award=award,
                rookies=rookies_df,
                feature_set=ref.feature_set,
                add_prev=True,
            )
            X = align_X(bundle.X, m)
            scores = predict_proba(m, X)

            res = bundle.meta.copy()
            res["award"] = award
            res["family"] = ref.family
            res["score"] = scores.values
            res["rank"] = res["score"].rank(ascending=False, method="first").astype(int)
            res = res.sort_values("score", ascending=False).reset_index(drop=True)

            top1 = float(res.loc[0, "score"]) if len(res) else float("nan")

            chosen.append(
                {
                    "award": award,
                    "family": ref.family,
                    "feature_set": ref.feature_set,
                    "format": ref.format,
                    "model_path": str(ref.model_path.as_posix()),
                    "top1_score": top1,
                }
            )

            res.head(topk).to_csv(out_dir / f"{award}_top{topk}.csv", index=False)
            all_rows.append(res)
            print(f"[OK] {award}: pretrained {ref.family} (top1={top1:.4f})")
            continue

        # ---- non-pretrained path: choose best family by top1 ----
        assert run_id is not None

        candidates: list[tuple[float, str, str, LoadedModel, pd.DataFrame]] = []
        for family, feature_set in (("baseline", "baseline"), ("tree", "tree")):
            run_dir = models_dir / run_id / family / award
            if not run_dir.exists():
                continue

            # resolve model file
            joblib_path = run_dir / "model.joblib"
            xgb_path = run_dir / "xgb_model.json"
            if xgb_path.exists():
                model_path = xgb_path
                fmt = "xgb_json"
            else:
                model_path = joblib_path
                fmt = "joblib"

            try:
                m = load_model_from_path(model_path)
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
            candidates.append((top1, family, fmt, m, res))

        if not candidates:
            print(f"[WARN] no model available for {award} in run {run_id}")
            continue

        candidates.sort(key=lambda t: t[0], reverse=True)
        best_top1, best_family, best_fmt, _, best_res = candidates[0]

        # correct model_path for meta
        run_dir = models_dir / run_id / best_family / award
        best_model_path = (run_dir / "xgb_model.json") if best_fmt == "xgb_json" else (run_dir / "model.joblib")

        chosen.append(
            {
                "award": award,
                "family": best_family,
                "feature_set": best_family,  # baseline/tree
                "format": best_fmt,
                "model_path": best_model_path.as_posix(),
                "top1_score": best_top1,
            }
        )

        best_res.head(topk).to_csv(out_dir / f"{award}_top{topk}.csv", index=False)
        all_rows.append(best_res)
        print(f"[OK] {award}: chose {best_family} (top1={best_top1:.4f})")

    if not all_rows:
        raise RuntimeError("No predictions produced.")

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
        "run_id": run_id if run_id is not None else "pretrained",
        "topk": topk,
        "chosen": chosen,
        "exports": {"all_awards_parquet": str(all_path)},
    }
    (out_dir / "prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Prediction completed")
    print("Outputs:", out_dir)
    print("All awards:", all_path)
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--pretrained", type=str, default=None, help="Path to pretrained manifest JSON.")
    p.add_argument("--snapshot-dir", type=str, default=None, help="Explicit snapshot dir (data/target/.../asof_...).")
    args = p.parse_args()

    run_predict_season(
        year=args.year,
        topk=args.topk,
        models_dir=Path(args.models_dir),
        pretrained_manifest=Path(args.pretrained) if args.pretrained else None,
        snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
