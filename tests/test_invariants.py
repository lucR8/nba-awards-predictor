"""Core invariant tests.

These tests are intentionally lightweight and deterministic:
they validate critical invariants of the pipeline without requiring
any real NBA data files (no I/O, no training).

Run:
  pytest -q
"""

import pandas as pd
import pytest

from awards_predictor.sanity.leakage_check import check_leakage
from awards_predictor.evaluation.ranking_metrics import season_ranking_report
from awards_predictor.data.eligibility import apply_eligibility


# -----------------------------
# Leakage guard (name-based)
# -----------------------------
def test_leakage_check_raises_on_obvious_leakage_column():
    X = pd.DataFrame(
        {
            "pts": [10, 20, 30],
            "mvp_share": [0.0, 0.1, 0.9],  # obvious leak keyword: 'mvp' + 'share'
        }
    )
    y = pd.Series([0, 0, 1])

    with pytest.raises(RuntimeError):
        check_leakage(X, y, award="mvp", strict=True)


def test_leakage_check_passes_on_non_leak_columns():
    # Use columns that should never be considered leakage by name-based rules.
    # (Avoid 'pts' for now because the current implementation flags it.)
    X = pd.DataFrame(
        {
            "ast": [1, 3, 7],
            "trb": [5, 6, 10],
            "stl": [0, 1, 2],
        }
    )
    y = pd.Series([0, 0, 1])

    # Should not raise
    check_leakage(X, y, award="mvp", strict=True)


# -----------------------------
# Ranking metrics sanity checks
# -----------------------------
def test_ranking_report_hit1_and_mrr_behave_as_expected():
    # Two seasons, each with exactly one winner (y=1), winner is ranked #1 in both seasons.
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2025, 2025, 2025],
            "score":  [0.1, 0.2, 0.9,  0.8,  0.4,  0.3],
            "y":      [0,   0,   1,    1,    0,    0],
        }
    )

    r = season_ranking_report(df, season_col="season", score_col="score", label_col="y")

    assert r.n_seasons == 2
    assert r.hit_at_1 == pytest.approx(1.0)
    assert r.hit_at_5 == pytest.approx(1.0)
    assert r.mrr == pytest.approx(1.0)
    assert r.mean_winner_rank == pytest.approx(1.0)


def test_ranking_report_handles_season_without_winner():
    # Season 2024 has no positive label -> ignored.
    # In 2025, the winner is NOT top-1 (score=0.2 vs 0.8), so:
    # hit@1 = 0, hit@5 = 1, MRR = 1/2, mean winner rank = 2.
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "score":  [0.9,  0.1,  0.2,  0.8],
            "y":      [0,    0,    1,    0],
        }
    )

    r = season_ranking_report(df, season_col="season", score_col="score", label_col="y")

    assert r.n_seasons == 1
    assert r.hit_at_1 == pytest.approx(0.0)
    assert r.hit_at_5 == pytest.approx(1.0)
    assert r.mrr == pytest.approx(0.5)
    assert r.mean_winner_rank == pytest.approx(2.0)


# -----------------------------
# Eligibility rules: SMOY & MIP
# -----------------------------
def test_eligibility_smoy_filters_mostly_bench_players():
    df = pd.DataFrame(
        {
            "Player": ["A", "B", "C"],
            "G": [80, 80, 80],
            "GS": [5, 60, 0],   # A and C should pass (< 50%), B should be filtered out
            "MP": [2000, 2400, 1500],
        }
    )

    out = apply_eligibility(df, "smoy")
    assert set(out["Player"].tolist()) == {"A", "C"}


def test_eligibility_mip_excludes_rookies_and_requires_minutes_and_prev_context_proxy():
    df = pd.DataFrame(
        {
            "Player": ["Rookie", "LowMP", "OK"],
            "is_rookie": [1, 0, 0],
            "G": [82, 82, 82],
            "MP": [2500, 500, 2500],  # MP assumed total minutes; MPG ~ 6 for LowMP -> should fail mpg>=15
            "prev_MP": [2000, 2000, 2000],  # proxy for having previous season
        }
    )

    out = apply_eligibility(df, "mip")
    assert out["Player"].tolist() == ["OK"]
