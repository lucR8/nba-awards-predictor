from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--title", default="", type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # robust column choice (works whether you stored *_mean or not)
    val_col = "val_auc_mean" if "val_auc_mean" in df.columns else "val_auc"
    tr_col = "train_auc_mean" if "train_auc_mean" in df.columns else "train_auc"

    df = df.sort_values("C")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["C"], df[tr_col], marker="o", label="Train AUC")
    plt.plot(df["C"], df[val_col], marker="o", label="Val AUC")
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("ROC AUC")
    plt.ylim(0.0, 1.0)

    title = args.title.strip()
    if not title:
        parts = []
        if "award" in df.columns and df["award"].nunique() == 1:
            parts.append(str(df["award"].iloc[0]))
        if "feature_set" in df.columns and df["feature_set"].nunique() == 1:
            parts.append(str(df["feature_set"].iloc[0]))
        if "penalty" in df.columns and df["penalty"].nunique() == 1:
            parts.append(f"logreg {df['penalty'].iloc[0]}")
        title = " / ".join(parts) if parts else "LogReg hyperparameter sweep"
    plt.title(title)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
