import argparse, pandas as pd
from pathlib import Path
from awards_predictor.features.feature_builder import build_mvp_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--input", type=str, required=True, help="players csv")
    ap.add_argument("--teams", type=str, required=True, help="teams csv")
    ap.add_argument("--out", type=str, required=True, help="output parquet path")
    args = ap.parse_args()

    df = build_mvp_features(args.input, args.teams)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[OK] Wrote features to {args.out}. Rows={len(df)}")

if __name__ == "__main__":
    main()
