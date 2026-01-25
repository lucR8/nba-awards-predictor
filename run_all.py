# run_all.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


import os

def run(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full NBA Awards pipeline (fetch → build → train → predict)")
    parser.add_argument("--year", type=int, default=2026, help="BRef season end year (default: 2026)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K predictions per award (default: 5)")
    args = parser.parse_args()

    year = args.year
    topk = args.topk

    print("\n==============================")
    print(f"🏀 NBA AWARDS RUN ALL — {year}")
    print("==============================")

    # 1) FETCH target season
    run([PYTHON, "-m", "scripts.fetch.run_target_season", "--year", str(year)])

    # 2) BUILD target dataset
    run([PYTHON, "-m", "scripts.build.run_target_season", "--year", str(year)])

    # 3) TRAIN all models (baseline + tree) -> models/<run_id>/...
    run([PYTHON, "-m", "awards_predictor.train.train_awards"])

    # 4) PREDICT using latest run in models/
    run([PYTHON, "-m", "awards_predictor.predict.predict_season"])

    print("\n✅ RUN ALL COMPLETED")
    print(f"📂 Outputs → data/target/{year}/asof_*/predictions/")


if __name__ == "__main__":
    main()
