"""Repository health check (structure + conventions).

Helper script (not a test) to quickly sanity-check the repository structure
from any working directory.

Run:
  python repo_health_check_fixed.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking up until either .git exists OR README.md+src/ exist."""
    p = (start or Path.cwd()).resolve()
    for cur in [p, *p.parents]:
        if (cur / ".git").exists():
            return cur
        if (cur / "README.md").exists() and (cur / "src").exists():
            return cur
    return p


REQUIRED_DIRS = [
    "src/awards_predictor",
    "scripts",
    "tests",
    ".github/workflows",
]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore",
]

# Generated artifacts: expected after running evaluation, should NOT be required in git
GENERATED_DIRS = [
    "reports",
]

OPTIONAL_BUT_RECOMMENDED = [
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
]


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def _check_many(root: Path, paths: Iterable[str]) -> tuple[list[str], list[str]]:
    ok, missing = [], []
    for rel in paths:
        if _exists(root, rel):
            ok.append(rel)
        else:
            missing.append(rel)
    return ok, missing


def main() -> int:
    root = find_repo_root()

    ok_dirs, missing_dirs = _check_many(root, REQUIRED_DIRS)
    ok_files, missing_files = _check_many(root, REQUIRED_FILES)
    gen_ok, gen_missing = _check_many(root, GENERATED_DIRS)
    opt_ok, opt_missing = _check_many(root, OPTIONAL_BUT_RECOMMENDED)

    report = {
        "repo_root": str(root),
        "required_ok": ok_dirs + ok_files,
        "required_missing": missing_dirs + missing_files,
        "generated_ok": gen_ok,
        "generated_missing": gen_missing,
        "optional_ok": opt_ok,
        "optional_missing": opt_missing,
    }

    print("\n🔍 REPO HEALTH CHECK\n")
    print(json.dumps(report, indent=2))

    missing_required = report["required_missing"]
    if missing_required:
        print("\n❌ Missing REQUIRED items:")
        for m in missing_required:
            print(f"  - {m}")
        return 1

    print("\n✅ Required repository structure looks good.")

    if opt_missing:
        print("\nℹ️ Optional items missing (fine for an academic project):")
        for m in opt_missing:
            print(f"  - {m}")

    if gen_missing:
        print("\nℹ️ Generated dirs not present yet (expected until you run evaluation):")
        for m in gen_missing:
            print(f"  - {m}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
