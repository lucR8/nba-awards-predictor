"""
Quick repo structure validator for nba-awards-predictor
Ensures the project respects recommended GitHub & Python standards.
"""

import os
import json
from pathlib import Path

REQUIRED_DIRS = [
    "src/awards_predictor",
    "tests",
    "notebooks",
    "models",
    "data/raw",
    "data/processed",
    ".github/workflows",
]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore",
]

OPTIONAL_BUT_RECOMMENDED = [
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
]

def check_exists(path):
    return Path(path).exists()

def main():
    report = {"missing": [], "ok": []}

    for d in REQUIRED_DIRS:
        if check_exists(d):
            report["ok"].append(d)
        else:
            report["missing"].append(d)

    for f in REQUIRED_FILES:
        if check_exists(f):
            report["ok"].append(f)
        else:
            report["missing"].append(f)

    print("\n🔍 REPO STRUCTURE CHECK\n")
    print(json.dumps(report, indent=2))

    if report["missing"]:
        print("\n⚠️ Missing important files/directories!")
        for m in report["missing"]:
            print(f"  → {m}")
    else:
        print("\n✅ Repository structure is compliant.")

if __name__ == "__main__":
    main()
