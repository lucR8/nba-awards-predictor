import argparse, json
from pathlib import Path
import pandas as pd
from joblib import dump
from awards_predictor.models.mvp_model import train_ranker_as_classifier
from awards_predictor.models.utils import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="model output path (.pkl)")
    ap.add_argument("--metrics", type=str, required=True, help="metrics json output")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    pipe, df_labeled = train_ranker_as_classifier(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, args.out)
    # Evaluate on train (placeholder until proper CV is added)
    from awards_predictor.evaluation.metrics import basic_binary_metrics
    # Predict proba of class 1
    import numpy as np
    proba = pipe.predict_proba(df_labeled[df_labeled.columns.difference(['label_mvp'])])[:,1]
    metrics = basic_binary_metrics(df_labeled['label_mvp'].values, proba)
    save_json(metrics, args.metrics)
    print(f"[OK] Model saved to {args.out}")
    print(f"[METRICS] {json.dumps(metrics)}")

if __name__ == "__main__":
    main()
