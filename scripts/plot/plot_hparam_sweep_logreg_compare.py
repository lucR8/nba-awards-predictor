from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _label(df: pd.DataFrame, fallback: str) -> str:
    if "penalty" in df.columns and df["penalty"].nunique() == 1:
        pen = str(df["penalty"].iloc[0])
        if pen == "elasticnet" and "l1_ratio" in df.columns:
            try:
                r = float(df["l1_ratio"].dropna().iloc[0])
                return f"elasticnet (l1_ratio={r:g})"
            except Exception:
                return "elasticnet"
        return pen
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="LogReg sweep comparison (Val AUC vs C)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    for i, p in enumerate(args.csvs):
        df = pd.read_csv(p).sort_values("C")
        val_col = "val_auc_mean" if "val_auc_mean" in df.columns else "val_auc"
        plt.plot(df["C"], df[val_col], marker="o", label=_label(df, f"run{i+1}"))

    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Val ROC AUC")
    plt.ylim(0.0, 1.0)
    plt.title(args.title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
