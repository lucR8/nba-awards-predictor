from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PYTHON = sys.executable


def _build_env() -> dict:
    env = os.environ.copy()
    src = str(SRC_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    return env


def run(cmd: List[str]) -> None:
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=_build_env())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run NBA Awards pipeline (fetch → build → train → leakage-check → predict → evaluate → plot)."
    )
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--topk", type=int, default=5)

    # pretrained (optional)
    p.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained manifest JSON (e.g., models/pretrained/v1/models_manifest.json).",
    )
    p.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Optional explicit target snapshot dir (e.g., data/target/2026/asof_2026-01-26).",
    )

    # toggles
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-predict", action="store_true")

    # leakage
    p.add_argument("--skip-leakage-check", action="store_true")
    p.add_argument("--leakage-strict", action="store_true")
    p.add_argument("--no-leakage-strict", action="store_true")
    p.add_argument("--leakage-corr-threshold", type=float, default=0.95)

    # eval + plots
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--eval-out", type=str, default="reports/model_eval")
    p.add_argument("--val-years", type=int, default=2)
    p.add_argument("--test-years", type=int, default=3)

    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-out", type=str, default="")

    args = p.parse_args()

    year = int(args.year)
    topk = int(args.topk)

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
        run(
            [
                PYTHON,
                "-m",
                "awards_predictor.train.train_awards",
                "--val-years",
                str(args.val_years),
                "--test-years",
                str(args.test_years),
            ]
        )

    # 3.5) LEAKAGE CHECK
    do_predict = not args.skip_predict
    do_leakage = (not args.skip_leakage_check) and do_predict

    if do_leakage:
        print("\n==============================")
        print("🧪 LEAKAGE CHECK (pre-prediction)")
        print("==============================")

        import pandas as pd
        from awards_predictor.config import AWARDS, HIST_ENRICHED_PARQUET
        from awards_predictor.sanity.leakage_check import check_leakage
        from awards_predictor.train.train_awards import build_training_table

        df_hist = pd.read_parquet(HIST_ENRICHED_PARQUET)

        strict = True
        if args.no_leakage_strict:
            strict = False
        if args.leakage_strict:
            strict = True

        corr_thr = float(args.leakage_corr_threshold)

        for award in AWARDS:
            table = build_training_table(df_hist, award=award, feature_set="baseline")
            X, y, meta = table.X, table.y, table.meta

            if isinstance(meta, pd.DataFrame) and "split" in meta.columns:
                m_train = meta["split"].astype(str).str.lower() == "train"
                Xc = X.loc[m_train]
                yc = y.loc[m_train]
            else:
                Xc, yc = X, y

            check_leakage(Xc, yc, award=award, corr_threshold=corr_thr, strict=strict)

        print("✅ Leakage check passed")

    # 4) PREDICT
    if do_predict:
        cmd = [
            PYTHON,
            "-m",
            "awards_predictor.predict.predict_season",
            "--year",
            str(year),
            "--topk",
            str(topk),
        ]

        if args.pretrained:
            print(f"[RUN_ALL] Using pretrained manifest: {args.pretrained}")
            cmd += ["--pretrained", args.pretrained]
        else:
            print("[RUN_ALL] Using latest TRAINED run in models/ (not pretrained).")
            # Optional: if you maintain a pointer written by training
            latest_file = REPO_ROOT / "models" / "LATEST_RUN_ID.txt"
            if latest_file.exists():
                rid = latest_file.read_text(encoding="utf-8").strip()
                if rid:
                    cmd += ["--run-id", rid]

        if args.snapshot_dir:
            cmd += ["--snapshot-dir", args.snapshot_dir]

        run(cmd)

    # 5) EVAL
    eval_out = Path(args.eval_out)
    if args.evaluate:
        run(
            [
                PYTHON,
                "-m",
                "awards_predictor.train.evaluate_models",
                "--out-dir",
                str(eval_out),
                "--val-years",
                str(args.val_years),
                "--test-years",
                str(args.test_years),
            ]
        )

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
        out = Path(args.plot_out) if args.plot_out else (Path(args.eval_out) / "plots")
        print(f"🖼️  Plots → {out}")


if __name__ == "__main__":
    main()
