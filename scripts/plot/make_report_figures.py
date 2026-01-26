from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
RANKING_METRICS = [
    ("test_hit@1", "Hit@1 (test)"),
    ("test_hit@5", "Hit@5 (test)"),
    ("test_mrr", "MRR (test)"),
]

def _safe_title(s: str) -> str:
    return s.replace("_", r"\_")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def select_best_model_per_award(
    metrics_df: pd.DataFrame,
    criterion: str = "test_auc",
) -> dict[str, str]:
    """
    Returns: {award: best_model_name}
    criterion: 'test_auc' or 'test_mrr' etc.
    """
    if criterion not in metrics_df.columns:
        raise ValueError(f"criterion={criterion} not in metrics_by_award columns: {metrics_df.columns.tolist()}")

    best = (
        metrics_df.sort_values(["award", criterion], ascending=[True, False])
                 .groupby("award", as_index=False)
                 .head(1)
    )
    return dict(zip(best["award"], best["model"]))

def load_top_features_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize typical column names:
    # - linear models: feature, coef
    # - permutation importance: feature, importance_mean, importance_std
    # - sometimes: feature, importance (single)
    cols = {c.lower(): c for c in df.columns}

    # Find feature column
    feat_col = None
    for key in ["feature", "name", "col"]:
        if key in cols:
            feat_col = cols[key]
            break
    if feat_col is None:
        raise ValueError(f"Could not find a feature column in {path}. Columns={df.columns.tolist()}")

    # Find value column (importance)
    val_col = None
    std_col = None

    if "importance_mean" in cols:
        val_col = cols["importance_mean"]
        if "importance_std" in cols:
            std_col = cols["importance_std"]
    elif "importance" in cols:
        val_col = cols["importance"]
    elif "coef" in cols:
        val_col = cols["coef"]
    elif "coefficient" in cols:
        val_col = cols["coefficient"]

    if val_col is None:
        raise ValueError(f"Could not find an importance/coef column in {path}. Columns={df.columns.tolist()}")

    out = df[[feat_col, val_col] + ([std_col] if std_col else [])].copy()
    out = out.rename(columns={feat_col: "feature", val_col: "value"})
    if std_col:
        out = out.rename(columns={std_col: "std"})
    else:
        out["std"] = np.nan

    # If coef: plot absolute magnitude for "importance"
    # Keep sign available (useful to interpret), but rank by abs
    out["abs_value"] = out["value"].abs()
    out = out.sort_values("abs_value", ascending=False)
    return out


# -----------------------------
# Figure makers
# -----------------------------
def make_temporal_split_figure(out_png: Path, train_years=(1996, 2018), val_years=(2019, 2021), test_years=(2022, 2025)):
    """
    Simple schematic. Adjust year ranges to your actual split if needed.
    """
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.set_axis_off()

    segments = [
        ("Train", train_years, 0.15),
        ("Val", val_years, 0.55),
        ("Test", test_years, 0.80),
    ]

    # Normalize to [0, 1] by year range
    min_y = min(train_years[0], val_years[0], test_years[0])
    max_y = max(train_years[1], val_years[1], test_years[1])
    span = max_y - min_y

    def x(y):  # year -> [0.05, 0.95]
        return 0.05 + 0.90 * ((y - min_y) / span)

    y0, h = 0.40, 0.28
    for label, (a, b), _ in segments:
        xa, xb = x(a), x(b)
        ax.add_patch(plt.Rectangle((xa, y0), xb - xa, h, fill=True, alpha=0.25, edgecolor="black"))
        ax.text((xa + xb) / 2, y0 + h / 2, f"{label}\n{a}–{b}", ha="center", va="center", fontsize=11)

    ax.text(0.5, 0.92, "Strict temporal split (contiguous season blocks)", ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def make_auc_val_to_test_slope(metrics_df: pd.DataFrame, out_png: Path):
    df = metrics_df.copy()
    df["delta"] = df["test_auc"] - df["val_auc"]

    fig, ax = plt.subplots(figsize=(10, 5))
    # One point per (award, model)
    for award, g in df.groupby("award"):
        ax.scatter(g["val_auc"], g["test_auc"], label=award)

    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Validation AUC")
    ax.set_ylabel("Test AUC")
    ax.set_title("AUC generalization (val → test) across awards/models")
    ax.legend(loc="lower right", ncols=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def make_ranking_metrics_dotplot(metrics_df: pd.DataFrame, out_png: Path):
    # Long form: one row per metric
    rows = []
    for metric, metric_name in RANKING_METRICS:
        tmp = metrics_df[["award", "model", metric]].copy()
        tmp["metric"] = metric_name
        tmp = tmp.rename(columns={metric: "value"})
        rows.append(tmp)
    df = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    y_map = {m: i for i, m in enumerate([m for _, m in RANKING_METRICS][::-1])}  # top-to-bottom
    # place metrics groups with offsets per award
    awards = sorted(df["award"].unique())
    metric_levels = [m for _, m in RANKING_METRICS][::-1]

    # offsets for awards in each metric line
    offsets = np.linspace(-0.25, 0.25, len(awards))

    for ai, award in enumerate(awards):
        g = df[df["award"] == award]
        for m in metric_levels:
            gg = g[g["metric"] == m]
            # take best model per award for clarity (max value)
            if gg.empty:
                continue
            best = gg.sort_values("value", ascending=False).head(1)
            ax.scatter(best["value"].iloc[0], y_map[m] + offsets[ai], label=award if m == metric_levels[0] else None)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("value (higher is better)")
    ax.set_title("Ranking metrics (test) — best model per award")
    ax.set_xlim(0.0, 1.02)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def make_mean_winner_rank_strip(metrics_df: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(11, 5))

    # plot all models points, colored by award
    for award, g in metrics_df.groupby("award"):
        ax.scatter(g["test_mean_winner_rank"], [award] * len(g), alpha=0.8, label=award)

    ax.set_xlabel("mean winner rank (test) — lower is better")
    ax.set_title("Mean winner rank across models (test)")
    ax.invert_xaxis()  # optional: best on the right? if you prefer, remove this line
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def make_top_features_plots(
    metrics_df: pd.DataFrame,
    top_features_dir: Path,
    out_dir: Path,
    criterion: str = "test_auc",
    k: int = 15,
):
    best_map = select_best_model_per_award(metrics_df, criterion=criterion)

    for award, model in best_map.items():
        # expected file pattern: top_features_{award}_{model}.csv
        # your filenames include e.g. top_features_smoy_logreg_elasticnet.csv
        pattern = f"top_features_{award}_{model}.csv"
        candidates = list(top_features_dir.glob(pattern))

        if not candidates:
            # fallback: sometimes you exported only some models; try any model for award
            candidates = list(top_features_dir.glob(f"top_features_{award}_*.csv"))

        if not candidates:
            print(f"[WARN] No top-features CSV found for award={award} (looked for {pattern})")
            continue

        csv_path = candidates[0]
        df = load_top_features_csv(csv_path).head(k)

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # barh: top at top
        df = df.iloc[::-1].copy()

        ax.barh(df["feature"], df["abs_value"])
        title = f"{award.upper()} — Top {len(df)} features ({model}, by {criterion})"
        ax.set_title(_safe_title(title))
        ax.set_xlabel("importance (|coef| or permutation importance)")
        ax.set_ylabel("")
        fig.tight_layout()

        out_png = out_dir / f"fig_top_features_{award}.png"
        fig.savefig(out_png, dpi=220)
        plt.close(fig)

        print(f"[OK] {award}: {out_png.name} from {csv_path.name}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True, help="Path to metrics_by_award.csv")
    ap.add_argument("--top-features-dir", type=Path, required=True, help="Directory containing top_features_{award}_{model}.csv")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for PNGs (e.g., figures/)")
    ap.add_argument("--best-criterion", type=str, default="test_auc", help="How to choose best model per award (test_auc, test_mrr, ...)")
    ap.add_argument("--k", type=int, default=15, help="Top-K features to display")
    ap.add_argument("--train-range", type=str, default="1996-2018", help="Temporal split: train years a-b")
    ap.add_argument("--val-range", type=str, default="2019-2021", help="Temporal split: val years a-b")
    ap.add_argument("--test-range", type=str, default="2022-2025", help="Temporal split: test years a-b")
    args = ap.parse_args()

    out_dir = args.out_dir
    _ensure_dir(out_dir)

    metrics_df = pd.read_csv(args.metrics)

    # Temporal split schematic
    def parse_range(s: str):
        a, b = s.split("-")
        return int(a), int(b)

    make_temporal_split_figure(
        out_png=out_dir / "fig_temporal_split.png",
        train_years=parse_range(args.train_range),
        val_years=parse_range(args.val_range),
        test_years=parse_range(args.test_range),
    )

    # Global plots from metrics_by_award.csv
    make_auc_val_to_test_slope(metrics_df, out_dir / "auc_val_to_test_slope.png")
    make_ranking_metrics_dotplot(metrics_df, out_dir / "ranking_metrics_test_dotplot.png")
    make_mean_winner_rank_strip(metrics_df, out_dir / "mean_winner_rank_test_strip.png")

    # Top-features per award (best model per award)
    make_top_features_plots(
        metrics_df=metrics_df,
        top_features_dir=args.top_features_dir,
        out_dir=out_dir,
        criterion=args.best_criterion,
        k=args.k,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
