import pandas as pd

from awards_predictor.features.columns import infer_percentile_cols, is_leakage_feature


def test_infer_percentile_cols_detects_common_patterns():
    cols = [
        "pct_ts",
        "fg_pctile",
        "ast_percentile",
        "random_feature",
        "MP",
        "winner",           # leakage-like
        "mvp_share",        # leakage-like
        "pct_is_mvp_winner" # explicit leakage pattern
    ]

    out = infer_percentile_cols(cols)

    # expected percentile-like columns
    assert "pct_ts" in out
    assert "fg_pctile" in out
    assert "ast_percentile" in out

    # non-percentile columns should not appear
    assert "random_feature" not in out
    assert "MP" not in out

    # leakage-like should be excluded
    assert "winner" not in out
    assert "mvp_share" not in out
    assert "pct_is_mvp_winner" not in out


def test_is_leakage_feature_flags_expected_patterns():
    assert is_leakage_feature("is_mvp_winner") is True
    assert is_leakage_feature("winner") is True
    assert is_leakage_feature("pct_is_mvp_winner") is True

    assert is_leakage_feature("pts") is False
    assert is_leakage_feature("pct_ts") is False
