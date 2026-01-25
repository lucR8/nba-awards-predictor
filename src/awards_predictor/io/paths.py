from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find repo root by walking up parents until we find:
      - data/ directory
      - scripts/ directory
    """
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
    raise RuntimeError(
        "Could not find project root (expected folders: data/ and scripts/). "
        f"Start was: {start}"
    )


def season_str(end_year: int) -> str:
    """2026 -> '2025-26'"""
    start = end_year - 1
    return f"{start}-{str(end_year)[-2:]}"


@dataclass(frozen=True)
class TargetSnapshotPaths:
    """
    Canonical paths for a target snapshot.
    """
    root: Path                 # .../data/target/<year>/asof_<date>
    raw: Path                  # .../raw
    build: Path                # .../build
    predictions: Path          # .../predictions

    raw_players_regular: Path  # .../raw/players/regular/<year>/...
    raw_teams_regular: Path    # .../raw/teams/regular/<year>/...
    raw_rookies: Path          # .../raw/rookies/<year>_rookies.csv
    raw_bio_dir: Path          # .../raw/nba_bio/
    raw_bio_csv: Path          # .../raw/nba_bio/<season>.csv

    build_players_final_dir: Path
    build_players_final: Path          # all_years_final.parquet
    build_players_with_bio: Path       # all_years_with_bio.parquet

    fetch_meta: Path           # .../meta.json or fetch_meta.json (depending on your fetch script)
    build_meta: Path           # .../build/meta.json


def get_target_year_dir(project_root: Path, year: int) -> Path:
    return project_root / "data" / "target" / str(year)


def list_snapshots(project_root: Path, year: int) -> list[Path]:
    year_dir = get_target_year_dir(project_root, year)
    if not year_dir.exists():
        return []
    snaps = [p for p in year_dir.iterdir() if p.is_dir() and p.name.startswith("asof_")]
    return sorted(snaps, key=lambda p: p.name)


def latest_snapshot(project_root: Path, year: int) -> Path:
    snaps = list_snapshots(project_root, year)
    if not snaps:
        raise FileNotFoundError(
            f"No snapshot found in data/target/{year}/ (expected asof_YYYY-MM-DD folders)"
        )
    return snaps[-1]


def resolve_snapshot_dir(
    project_root: Path,
    year: int,
    snapshot: Optional[str] = None,
) -> Path:
    """
    snapshot=None -> latest asof_ folder
    snapshot='YYYY-MM-DD' -> data/target/<year>/asof_<snapshot>
    """
    if snapshot is None:
        return latest_snapshot(project_root, year)
    p = get_target_year_dir(project_root, year) / f"asof_{snapshot}"
    if not p.exists():
        raise FileNotFoundError(f"Snapshot not found: {p}")
    return p


def target_paths(
    year: int,
    snapshot: Optional[str] = None,
    *,
    project_root: Optional[Path] = None,
) -> TargetSnapshotPaths:
    """
    Build a consistent set of paths for a given target snapshot.
    """
    root = project_root or find_project_root()
    snap_dir = resolve_snapshot_dir(root, year, snapshot)

    raw = snap_dir / "raw"
    build = snap_dir / "build"
    predictions = snap_dir / "predictions"

    season = season_str(year)

    return TargetSnapshotPaths(
        root=snap_dir,
        raw=raw,
        build=build,
        predictions=predictions,

        raw_players_regular=raw / "players" / "regular" / str(year),
        raw_teams_regular=raw / "teams" / "regular" / str(year),
        raw_rookies=raw / "rookies" / f"{year}_rookies.csv",
        raw_bio_dir=raw / "nba_bio",
        raw_bio_csv=raw / "nba_bio" / f"{season}.csv",

        build_players_final_dir=build / "players" / "final",
        build_players_final=build / "players" / "final" / "all_years_final.parquet",
        build_players_with_bio=build / "players" / "final" / "all_years_with_bio.parquet",

        # depending on your fetch script naming (meta.json or fetch_meta.json)
        fetch_meta=(snap_dir / "meta.json") if (snap_dir / "meta.json").exists() else (snap_dir / "fetch_meta.json"),
        build_meta=build / "meta.json",
    )
