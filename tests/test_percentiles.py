import pandas as pd
from awards_predictor.features.percentiles import add_percentiles, zscore_by_group

def test_add_percentiles():
    df = pd.DataFrame({"x":[1,2,3]})
    out = add_percentiles(df, cols=["x"])
    assert "x_pctile" in out.columns
    assert out["x_pctile"].between(0,1).all()

def test_zscore_by_group():
    df = pd.DataFrame({"g":["A","A","B","B"], "x":[1,2,10,12]})
    out = zscore_by_group(df, cols=["x"], groupby="g")
    assert "x_z" in out.columns
    assert abs(out[out["g"]=="A"]["x_z"].mean()) < 1e-6  # mean≈0 within group
