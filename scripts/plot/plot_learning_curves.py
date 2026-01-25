from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from scripts._bootstrap import *  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values("n_seasons")

    plt.figure()
    plt.plot(df["n_seasons"], df["train_auc"], marker="o", label="train AUC")
    plt.plot(df["n_seasons"], df["val_auc"], marker="o", label="val AUC")
    plt.xlabel("Number of train seasons (earliest → latest)")
    plt.ylabel("AUC")
    plt.title(f"Learning curve — {df['award'].iloc[0]} — {df['model'].iloc[0]} ({df['feature_set'].iloc[0]})")
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
