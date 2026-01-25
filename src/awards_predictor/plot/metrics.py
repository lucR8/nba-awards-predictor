# src/awards_predictor/plot/metrics.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


AWARD_ORDER = ["mvp", "dpoy", "smoy", "roy", "mip"]


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metrics CSV: {missing}. Available={list(df.columns)}")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["award"] = df["award"].astype(str).str.lower()
    df["award"] = pd.Categorical(df["award"], categories=AWARD_ORDER, ordered=True)
    df["model"] = df["model"].astype(str)

    # numeric coercion
    numeric_cols = [
        "val_auc",
        "test_auc",
        "val_mrr",
        "test_mrr",
        "val_hit@1",
        "test_hit@1",
        "val_hit@5",
        "test_hit@5",
        "test_mean_winner_rank",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _legend_outside(ax) -> None:
    # put legend outside on the right
    leg = ax.legend(title="model", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(10)
        if leg.get_title():
            leg.get_title().set_fontsize(11)


def plot_auc_slope(df_award: pd.DataFrame, out_dir: Path, award: str) -> None:
    # slope chart: val -> test per model
    d = df_award[["model", "val_auc", "test_auc"]].dropna()
    if d.empty:
        return

    long = d.melt(id_vars=["model"], value_vars=["val_auc", "test_auc"], var_name="split", value_name="auc")
    long["split"] = long["split"].map({"val_auc": "val", "test_auc": "test"}).fillna(long["split"])

    plt.figure(figsize=(8.5, 5))
    ax = sns.lineplot(
        data=long,
        x="split",
        y="auc",
        hue="model",
        marker="o",
        linewidth=2.2,
        markersize=8,
        palette="tab10",
    )
    ax.set_title(f"{award.upper()} — AUC (val → test)")
    ax.set_xlabel("")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.01)
    ax.grid(True, axis="y", alpha=0.35)
    _legend_outside(ax)
    _savefig(out_dir / "auc_val_to_test_slope.png")


def plot_ranking_dotplot(df_award: pd.DataFrame, out_dir: Path, award: str) -> None:
    cols = ["test_hit@1", "test_hit@5", "test_mrr"]
    cols = [c for c in cols if c in df_award.columns]
    if not cols:
        return

    d = df_award[["model"] + cols].copy()
    d = d.dropna(subset=cols, how="all")
    if d.empty:
        return

    long = d.melt(id_vars=["model"], value_vars=cols, var_name="metric", value_name="value").dropna()
    metric_map = {"test_hit@1": "Hit@1 (test)", "test_hit@5": "Hit@5 (test)", "test_mrr": "MRR (test)"}
    long["metric"] = long["metric"].map(metric_map).fillna(long["metric"])

    plt.figure(figsize=(9.5, 5.5))
    ax = sns.stripplot(
        data=long,
        x="value",
        y="metric",
        hue="model",
        dodge=True,
        size=9,
        alpha=0.95,
        palette="tab10",
    )
    ax.set_title(f"{award.upper()} — Ranking metrics (test)")
    ax.set_xlabel("value (higher is better)")
    ax.set_ylabel("")
    ax.set_xlim(0.0, 1.02)
    ax.grid(True, axis="x", alpha=0.35)
    _legend_outside(ax)
    _savefig(out_dir / "ranking_metrics_test_dotplot.png")


def plot_mean_winner_rank(df_award: pd.DataFrame, out_dir: Path, award: str) -> None:
    if "test_mean_winner_rank" not in df_award.columns:
        return
    d = df_award[["model", "test_mean_winner_rank"]].dropna()
    if d.empty:
        return

    plt.figure(figsize=(9.5, 4.8))
    # FIX: no palette without hue (seaborn FutureWarning)
    ax = sns.stripplot(
        data=d,
        x="test_mean_winner_rank",
        y="model",
        size=10,
        alpha=0.95,
    )
    ax.set_title(f"{award.upper()} — Mean winner rank (test) (lower is better)")
    ax.set_xlabel("mean winner rank")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.35)
    _savefig(out_dir / "mean_winner_rank_test_strip.png")


def plot_topk_table(df_award: pd.DataFrame, out_dir: Path, award: str, k: int = 5) -> None:
    keep = ["model", "test_auc", "test_mrr", "test_hit@1", "test_hit@5", "test_mean_winner_rank"]
    keep = [c for c in keep if c in df_award.columns]
    if "test_auc" not in keep:
        return

    d = df_award[keep].copy()
    d = d.dropna(subset=["test_auc"])
    if d.empty:
        return

    d = d.sort_values("test_auc", ascending=False).head(k)

    # nicer formatting
    fmt = d.copy()
    for c in fmt.columns:
        if c == "model":
            continue
        fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")
    ax.set_title(f"{award.upper()} — Top {k} models (test)", pad=10)

    table = ax.table(
        cellText=fmt.values,
        colLabels=fmt.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    _savefig(out_dir / "top5_models_table.png")


def plot_summary_overview(df: pd.DataFrame, out_dir: Path) -> None:
    # simple overview: best model per award (test_auc)
    if "test_auc" not in df.columns:
        return

    best = (
        df.dropna(subset=["test_auc"])
        .sort_values(["award", "test_auc"], ascending=[True, False])
        .groupby("award", as_index=False, observed=False)  # FIX: silence pandas FutureWarning
        .head(1)
    )
    if best.empty:
        return

    plt.figure(figsize=(9, 4.5))
    ax = sns.barplot(data=best, x="award", y="test_auc", hue="model", palette="tab10", errorbar=None)
    ax.set_title("Best model per award (test AUC)")
    ax.set_xlabel("award")
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.5, 1.01)
    _legend_outside(ax)
    _savefig(out_dir / "_summary" / "best_test_auc_per_award.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="reports/model_eval/metrics_by_award.csv")
    parser.add_argument("--out-dir", type=str, default="reports/model_eval/plots")
    parser.add_argument("--by-award", action="store_true", help="Create one folder per award with clean plots.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(metrics_path)
    _ensure_columns(df, ["award", "model"])
    df = _prep(df)

    sns.set_theme(style="whitegrid", context="talk")

    # always write a clean overview summary too
    plot_summary_overview(df, out_dir)

    if args.by_award:
        for award in AWARD_ORDER:
            df_award = df[df["award"].astype(str) == award].copy()
            if df_award.empty:
                continue
            ddir = out_dir / award
            plot_auc_slope(df_award, ddir, award)
            plot_ranking_dotplot(df_award, ddir, award)
            plot_mean_winner_rank(df_award, ddir, award)
            plot_topk_table(df_award, ddir, award, k=5)

        print(f"[OK] wrote award plots to: {out_dir.resolve()}")
    else:
        # fallback: only overview
        print(f"[OK] wrote summary plots to: {(out_dir / '_summary').resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
