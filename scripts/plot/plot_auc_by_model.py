from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="Learning-curve CSV files (one per model).",
    )
    ap.add_argument("--award", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    rows = []

    for csv_path in args.csvs:
        df = pd.read_csv(csv_path)

        # keep only full-data point
        df = df[df["frac"] == df["frac"].max()]
        df = df[df["status"] == "ok"]

        if df.empty:
            continue

        row = df.iloc[0]
        rows.append(
            {
                "model": row["model"],
                "val_auc_mean": row["val_auc_mean"],
                "val_auc_std": row["val_auc_std"],
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False)

    # ---- plot
    plt.figure(figsize=(7, 4))
    plt.errorbar(
        plot_df["val_auc_mean"],
        plot_df["model"],
        xerr=plot_df["val_auc_std"],
        fmt="o",
        capsize=4,
    )
    plt.xlabel("Validation AUC")
    plt.title(f"Validation AUC by model — {args.award.upper()}")
    plt.grid(axis="x", alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
