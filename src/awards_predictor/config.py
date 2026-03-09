from __future__ import annotations

from pathlib import Path

from awards_predictor.io.paths import find_project_root


PROJECT_ROOT = find_project_root()

# Historique 1996–2025 
HIST_ENRICHED_PARQUET: Path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players"
    / "final"
    / "all_years_enriched.parquet"
)

RUNS_BASELINE_DIR: Path = PROJECT_ROOT / "data" / "processed" / "runs" / "baseline"
RUNS_TREE_DIR: Path = PROJECT_ROOT / "data" / "processed" / "runs" / "tree"

AWARDS = ["mvp", "dpoy", "smoy", "roy", "mip"]
