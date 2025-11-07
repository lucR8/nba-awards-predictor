import argparse, pandas as pd
from pathlib import Path
from joblib import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    X = pd.read_parquet(args.features)
    model = load(args.model)
    # Align columns expected by pipeline (we rely on fit columns order)
    proba = model.predict_proba(X)[:,1]
    out = X[["player_id","player_name","team","position"]].copy()
    out["mvp_probability"] = proba
    out = out.sort_values("mvp_probability", ascending=False).head(args.topk)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] Wrote predictions to {args.out}")
    print(out)

if __name__ == "__main__":
    main()
