# scripts/export_pretrained_models.py
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SelectedModel:
    award: str
    family: str
    feature_set: str
    fmt: str
    src_path: Path
    dst_path: Path
    top1_score: float | None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    # scripts/ is at repo_root/scripts/
    return Path(__file__).resolve().parents[1]


def export_from_prediction_meta(
    prediction_meta: Path,
    out_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    repo = _repo_root()
    data = _load_json(prediction_meta)

    chosen = data.get("chosen", [])
    if not chosen:
        raise ValueError(f"No 'chosen' models found in {prediction_meta}")

    out_root = out_root if out_root.is_absolute() else (repo / out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected: list[SelectedModel] = []

    for entry in chosen:
        award = str(entry["award"]).lower()
        family = str(entry.get("family", "")).lower()
        feature_set = str(entry.get("feature_set", family)).lower()
        fmt = str(entry.get("format", "joblib")).lower()

        src_raw = str(entry["model_path"])
        src_path = Path(src_raw)

        # if meta contains absolute windows path -> keep it
        # if meta contains relative -> resolve from repo root
        if not src_path.is_absolute():
            src_path = repo / src_path

        if not src_path.exists():
            raise FileNotFoundError(f"Missing model file referenced in prediction_meta: {src_path}")

        # keep filename (model.joblib or xgb_model.json)
        dst_path = out_root / family / award / src_path.name

        selected.append(
            SelectedModel(
                award=award,
                family=family,
                feature_set=feature_set,
                fmt=fmt,
                src_path=src_path,
                dst_path=dst_path,
                top1_score=float(entry["top1_score"]) if "top1_score" in entry else None,
            )
        )

    manifest = {
        "version": out_root.name,  # e.g. v1
        "source": {
            "prediction_meta": str(prediction_meta.as_posix()),
            "run_id": data.get("run_id"),
            "asof": data.get("asof"),
            "year": data.get("year"),
        },
        "models": [],
    }

    for s in selected:
        s.dst_path.parent.mkdir(parents=True, exist_ok=True)
        if s.dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {s.dst_path} (use --overwrite)")

        shutil.copy2(s.src_path, s.dst_path)

        # store a repo-friendly relative path
        rel = s.dst_path.relative_to(repo).as_posix()

        payload = {
            "award": s.award,
            "family": s.family,          # baseline/tree
            "feature_set": s.feature_set,
            "format": s.fmt,             # joblib or xgb_json
            "model_path": rel,
        }
        if s.top1_score is not None:
            payload["top1_score"] = s.top1_score

        manifest["models"].append(payload)

    manifest_path = out_root / "models_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prediction-meta",
        type=str,
        required=True,
        help="Path to prediction_meta.json produced by predict_season.py",
    )
    p.add_argument("--out", type=str, default="models/pretrained/v1")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    repo = _repo_root()
    prediction_meta = Path(args.prediction_meta)
    if not prediction_meta.is_absolute():
        prediction_meta = repo / prediction_meta

    out_root = Path(args.out)

    manifest_path = export_from_prediction_meta(
        prediction_meta=prediction_meta,
        out_root=out_root,
        overwrite=args.overwrite,
    )
    print(f"[OK] wrote manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
