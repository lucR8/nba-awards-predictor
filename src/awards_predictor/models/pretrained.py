from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
            family=str(m.get("family", "")).lower(),
            feature_set=str(m.get("feature_set", "baseline")).lower(),
            format=str(m.get("format", "joblib")).lower(),
            model_path=Path(m["model_path"]),
        )
    return out
