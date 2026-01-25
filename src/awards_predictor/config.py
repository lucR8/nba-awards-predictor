from __future__ import annotations

from pathlib import Path

from awards_predictor.io.paths import find_project_root


PROJECT_ROOT = find_project_root()

# ✅ Historique 1996–2025 (relatif au repo)
HIST_ENRICHED_PARQUET: Path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players"
    / "final"
    / "all_years_enriched.parquet"
)

# ✅ Racines de runs (à adapter si votre arbo diffère)
# (vous pouvez aussi ne pas les mettre ici et les passer en CLI)
RUNS_BASELINE_DIR: Path = PROJECT_ROOT / "data" / "processed" / "runs" / "baseline"
RUNS_TREE_DIR: Path = PROJECT_ROOT / "data" / "processed" / "runs" / "tree"

AWARDS = ["mvp", "dpoy", "smoy", "roy", "mip"]
