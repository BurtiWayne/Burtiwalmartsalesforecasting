"""
Tests for the data, feature and model pipeline.

The one test that really earns its place is `test_no_short_lag_features`. It is
easy for someone (including future me) to add a lag_1 feature because it improves
the score, without noticing it makes the forecast impossible to actually produce.
This test fails loudly if that happens.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import analysis as A
from src import model as M
from src.data_prep import add_calendar_features, handle_missing, load_clean, load_raw
from src.features import FEATURE_COLUMNS, HORIZON, build_features, build_future_frame

DATA = str(Path(__file__).resolve().parents[1] / "data" / "Walmart_DataSet.csv")


@pytest.fixture(scope="module")
def df():
    return load_clean(DATA)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def test_shape_matches_spec(df):
    assert len(df) == 6435
    assert df["Store"].nunique() == 45
    assert df["Date"].nunique() == 143


def test_balanced_panel(df):
    assert df["Store"].nunique() * df["Date"].nunique() == len(df)


def test_dates_parsed_day_first():
    """05-02-2010 must be 5 Feb, not 2 May. Getting this wrong wrecks seasonality."""
    raw = load_raw(DATA)
    first = raw.sort_values("Date")["Date"].min()
    assert first.month == 2 and first.day == 5


def test_no_nulls_after_cleaning(df):
    assert df.isnull().sum().sum() == 0


def test_handle_missing_imputes_within_store():
    """A gap in one store must not be filled using another store's level."""
    df = load_clean(DATA)
    small = df[df["Store"].isin([20, 33])].copy()
    idx = small[small["Store"] == 33].index[10]
    original = small.loc[idx, "Weekly_Sales"]
    small.loc[idx, "Weekly_Sales"] = np.nan

    filled = handle_missing(small)
    imputed = filled.loc[idx, "Weekly_Sales"]

    store33_max = df.loc[df["Store"] == 33, "Weekly_Sales"].max()
    assert imputed < store33_max, "Imputation leaked a larger store's level"
    assert abs(imputed - original) / original < 0.25


# --------------------------------------------------------------------------- #
# Features - the leakage guard
# --------------------------------------------------------------------------- #
def test_no_short_lag_features():
    """No feature may depend on data inside the forecast horizon."""
    for col in FEATURE_COLUMNS:
        if col.startswith("lag_"):
            assert int(col.split("_")[1]) >= HORIZON, f"{col} leaks future data"


def test_rolling_windows_are_shifted(df):
    """roll_12 at time t must not include anything from the last `HORIZON` weeks."""
    feat = build_features(df[M.RAW_COLUMNS])
    store = feat[feat["Store"] == 1].sort_values("Date").reset_index(drop=True)

    i = 100
    expected = store["Weekly_Sales"].iloc[i - HORIZON - 11 : i - HORIZON + 1].mean()
    assert np.isclose(store["roll_12"].iloc[i], expected, rtol=1e-6)


def test_future_frame_shape(df):
    future, dates = build_future_frame(df[M.RAW_COLUMNS])
    assert len(dates) == HORIZON
    assert len(future) == HORIZON * df["Store"].nunique()
    assert future["Weekly_Sales"].isna().all()
    assert dates.min() > df["Date"].max()


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def test_outliers_are_seasonal(df):
    """Extreme weeks should be Nov/Dec. If not, something is wrong upstream."""
    out = A.outlier_analysis(df)
    months = set(out["global_by_month"].index)
    assert months <= {"Nov", "Dec"}


def test_worst_store_is_low_volume_not_volatile(df):
    """The best/worst gap is structural, so the worst store should be stable."""
    gap = A.performance_gap(df)
    perf = gap["performance"]
    assert gap["ratio"] > 5
    assert gap["no_overlap"]
    assert perf.loc[gap["worst_store"], "cv_pct"] < perf["cv_pct"].median()


def test_peak_week_is_pre_christmas(df):
    seas = A.seasonality_analysis(df)
    assert int(seas["by_week"].idxmax()) in (50, 51, 52)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_model_beats_seasonal_naive(df):
    """If the model cannot beat 'repeat last year', it is not worth shipping."""
    res = M.train_validate(df)
    baselines = M.baseline_scores(df)
    assert res["metrics"]["MAPE"] < baselines["Seasonal Naive (t-52)"]
    assert res["metrics"]["MAPE"] < 6.0


def test_validation_split_is_time_ordered(df):
    res = M.train_validate(df)
    assert res["train"]["Date"].max() < res["test"]["Date"].min()


def test_forecast_output_shape(df):
    res = M.train_validate(df)
    fc = M.forecast_future(df, res["store_mape"])
    assert len(fc) == HORIZON * df["Store"].nunique()
    assert (fc["Forecast"] > 0).all()
    assert (fc["Lower_95"] <= fc["Forecast"]).all()
    assert (fc["Upper_95"] >= fc["Forecast"]).all()


def test_forecast_captures_christmas_peak(df):
    """The December weeks must forecast above the store's own average."""
    res = M.train_validate(df)
    fc = M.forecast_future(df, res["store_mape"])
    dec = fc[fc["Date"].dt.month == 12].groupby("Store")["Forecast"].max()
    avg = fc.groupby("Store")["Forecast"].mean()
    assert (dec > avg).mean() > 0.9
