from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingReport:
    hit_at_1: float
    hit_at_5: float
    mrr: float
    mean_winner_rank: float
    n_seasons: int


def _winner_rank(scores: pd.Series, y_true: pd.Series) -> int | None:
    """
    Compute rank (1=best) of the true winner within one season.
    If y_true has no positive, returns None.
    """
    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    if y.sum() == 0:
        return None

    # assume 1 winner; if multiple positives, take best-ranked positive
    order = scores.rank(ascending=False, method="first")
    ranks = order[y == 1]
    return int(ranks.min())


def season_ranking_report(df: pd.DataFrame, *, season_col: str, score_col: str, label_col: str) -> RankingReport:
    ranks = []
    for _, g in df.groupby(season_col, sort=True):
        r = _winner_rank(g[score_col], g[label_col])
        if r is not None:
            ranks.append(r)

    if not ranks:
        return RankingReport(0.0, 0.0, 0.0, float("nan"), 0)

    ranks_np = np.array(ranks, dtype=float)
    hit1 = float(np.mean(ranks_np <= 1))
    hit5 = float(np.mean(ranks_np <= 5))
    mrr = float(np.mean(1.0 / ranks_np))
    mean_rank = float(np.mean(ranks_np))
    return RankingReport(hit1, hit5, mrr, mean_rank, len(ranks))
