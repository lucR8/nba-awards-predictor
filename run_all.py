# run_all.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

# Make src importable for THIS process too (not only subprocesses)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PYTHON = sys.executable


def _build_env() -> dict:
    """Environment with src/ on PYTHONPATH so `python -m awards_predictor...` works without pip install."""
    env = os.environ.copy()
    src = str(SRC_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    return env


def run(cmd: List[str]) -> None:
    """Run a command from repository root with a consistent environment."""
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=_build_env())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full NBA Awards pipeline (fetch → build → train → leakage-check → predict → evaluate → plot)."
    )
    parser.add_argument("--year", type=int, default=2026, help="BRef season end year (default: 2026)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K predictions per award (default: 5)")

    # Steps toggles
    parser.add_argument("--skip-fetch", action="store_true", help="Skip target-season fetch step.")
    parser.add_argument("--skip-build", action="store_true", help="Skip target-season build step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction step.")

    # Leakage check (after train, before predict)
    parser.add_argument("--skip-leakage-check", action="store_true", help="Skip leakage check.")
    parser.add_argument("--leakage-strict", action="store_true", help="Force strict leakage mode.")
    parser.add_argument("--no-leakage-strict", action="store_true", help="Only warn on leakage suspicion.")
    parser.add_argument("--leakage-corr-threshold", type=float, default=0.95)

    # Offline evaluation
    parser.add_argument("--evaluate", action="store_true", help="Run offline evaluation for report.")
    parser.add_argument("--eval-out", type=str, default="reports/model_eval", help="Eval output dir.")
    parser.add_argument("--val-years", type=int, default=2, help="Number of validation seasons (default: 2)")
    parser.add_argument("--test-years", type=int, default=3, help="Number of test seasons (default: 3)")

    # Plots (optional)
    parser.add_argument("--plot", action="store_true", help="Generate plots from evaluation metrics (requires --evaluate).")
    parser.add_argument("--plot-out", type=str, default="", help="Plot output dir (default: <eval-out>/plots).")

    args = parser.parse_args()

    year = args.year
    topk = args.topk

    print("\n==============================")
    print(f"🏀 NBA AWARDS RUN ALL — {year}")
    print("==============================")

    # 1) FETCH
    if not args.skip_fetch:
        run([PYTHON, "-m", "scripts.fetch.run_target_season", "--year", str(year)])

    # 2) BUILD
    if not args.skip_build:
        run([PYTHON, "-m", "scripts.build.run_target_season", "--year", str(year)])

    # 3) TRAIN
    if not args.skip_train:
        run([PYTHON, "-m", "awards_predictor.train.train_awards",
             "--val-years", str(args.val_years), "--test-years", str(args.test_years)])

    # 3.5) LEAKAGE CHECK (only if we will predict)
    do_predict = not args.skip_predict
    do_leakage = (not args.skip_leakage_check) and do_predict

    if do_leakage:
        print("\n==============================")
        print("🧪 LEAKAGE CHECK (pre-prediction)")
        print("==============================")

        import pandas as pd
        from awards_predictor.config import AWARDS, HIST_ENRICHED_PARQUET
        from awards_predictor.sanity.leakage_check import check_leakage
        from awards_predictor.train.train_awards import _build_training_table

        df_hist = pd.read_parquet(HIST_ENRICHED_PARQUET)

        strict = True
        if args.no_leakage_strict:
            strict = False
        if args.leakage_strict:
            strict = True

        corr_thr = float(args.leakage_corr_threshold)

        for award in AWARDS:
            table = _build_training_table(df_hist, award=award, feature_set="baseline")
            X, y, meta = table.X, table.y, table.meta

            # If split is available, check leakage only on TRAIN
            if isinstance(meta, pd.DataFrame) and "split" in meta.columns:
                m_train = meta["split"].astype(str).str.lower() == "train"
                Xc = X.loc[m_train] if hasattr(X, "loc") else X[m_train.values]
                yc = y.loc[m_train] if hasattr(y, "loc") else y[m_train.values]
            else:
                Xc, yc = X, y

            if hasattr(Xc, "columns"):
                check_leakage(Xc, yc, award=award, corr_threshold=corr_thr, strict=strict)
            else:
                print(f"[LEAKAGE] skip (X not DataFrame) award={award}")

        print("✅ Leakage check passed")

    # 4) PREDICT
    if do_predict:
        run([PYTHON, "-m", "awards_predictor.predict.predict_season",
             "--year", str(year), "--topk", str(topk)])

    # 5) EVAL
    did_eval = False
    eval_out = Path(args.eval_out)
    if args.evaluate:
        run([PYTHON, "-m", "awards_predictor.train.evaluate_models",
             "--out-dir", str(eval_out),
             "--val-years", str(args.val_years),
             "--test-years", str(args.test_years)])
        did_eval = True

    # 6) PLOTS
    if args.evaluate and args.plot:
        plot_out = args.plot_out or f"{args.eval_out}/plots"
        run(
            [
                PYTHON,
                "-m",
                "awards_predictor.plot.metrics",
                "--metrics",
                f"{args.eval_out}/metrics_by_award.csv",
                "--out-dir",
                plot_out,
                "--by-award",
            ]
        )



    print("\n✅ RUN ALL COMPLETED")
    print(f"📂 Predictions → data/target/{year}/asof_*/predictions/")
    if args.evaluate:
        print(f"📊 Evaluation → {eval_out}")
    if args.plot:
        print(f"🖼️  Plots → {Path(args.plot_out) if args.plot_out else (Path(args.eval_out)/'plots')}")


if __name__ == "__main__":
    main()
