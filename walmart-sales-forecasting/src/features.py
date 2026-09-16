"""
Feature engineering for 12-week-ahead forecasting.

The single most important constraint in this file: every feature must be knowable
at forecast time. The tempting move is to feed the model last week's sales
(lag_1). It scores beautifully - about 3.67% MAPE in testing - but it is a lie.
To predict week 12 you would need week 11's actuals, which do not exist yet.

So no lag shorter than the forecast horizon, and every rolling window is shifted
by the full horizon before it is computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 12

FEATURE_COLUMNS = [
    "Store",
    "WeekOfYear",
    "Month",
    "Holiday_Flag",
    "t",
    "week_sin",
    "week_cos",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "lag_52",
    "lag_53",
    "roll_12",
    "roll_52",
    "seasonal_index",
]

TARGET = "Weekly_Sales"

# Features that need a warm-up period and will be NaN at the start of each store.
WARMUP_COLUMNS = ["lag_52", "roll_12", "roll_52", "seasonal_index"]


def build_features(data: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Create only features that are genuinely available `horizon` weeks in advance.

    Expects columns: Store, Date, Weekly_Sales, Holiday_Flag, Temperature,
    Fuel_Price, CPI, Unemployment. Rows with a NaN target (future rows) are fine -
    they get features but no label.
    """
    d = data.sort_values(["Store", "Date"]).copy()

    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["Month"] = d["Date"].dt.month
    d["Year"] = d["Date"].dt.year

    # Linear time index in weeks, for any underlying growth or decline.
    d["t"] = (d["Date"] - d["Date"].min()).dt.days // 7

    # Cyclical encoding so the model knows week 52 and week 1 are neighbours
    # rather than 51 units apart.
    d["week_sin"] = np.sin(2 * np.pi * d["WeekOfYear"] / 52)
    d["week_cos"] = np.cos(2 * np.pi * d["WeekOfYear"] / 52)

    g = d.groupby("Store")[TARGET]

    # 52 and 53 are both greater than the 12-week horizon, so these are already
    # observed by the time we need to forecast.
    d["lag_52"] = g.shift(52)
    d["lag_53"] = g.shift(53)

    # Shift by the FULL horizon before rolling, so no future information leaks in.
    shifted = g.shift(horizon)
    d["roll_12"] = shifted.rolling(12, min_periods=6).mean().values
    d["roll_52"] = shifted.rolling(52, min_periods=26).mean().values

    # Same week last year divided by that period's level: a per-store seasonal
    # index that separates "it's Christmas" from "this store got bigger".
    d["seasonal_index"] = d["lag_52"] / d["roll_52"]

    return d


def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the first ~year of each store, where the lag-52 features are empty."""
    return df.dropna(subset=WARMUP_COLUMNS)


def build_future_frame(
    history: pd.DataFrame, horizon: int = HORIZON
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Construct the rows we want predictions for.

    We genuinely do not know December's temperature or CPI in advance, so:
      - Temperature comes from the historical seasonal average for that
        week-of-year at that store (a fair estimate of regional climate).
      - CPI, Fuel_Price and Unemployment are carried forward from their last
        observed value. They are monthly step functions that move slowly.

    Since these variables carry almost no predictive signal (~1% of total feature
    importance), the cost of getting them slightly wrong is small. Estimating them
    honestly beats pretending we have perfect foresight.
    """
    last_date = history["Date"].max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W-FRI"
    )

    hist = history.copy()
    hist["WeekOfYear"] = hist["Date"].dt.isocalendar().week.astype(int)

    # Holiday weeks repeat on the same ISO weeks each year, and holiday dates are
    # known years ahead, so this is legitimate information.
    holiday_weeks = set(
        history.loc[history["Holiday_Flag"] == 1, "Date"].dt.isocalendar().week.unique()
    )

    rows = []
    for store, g in history.groupby("Store"):
        g = g.sort_values("Date")
        store_hist = hist[hist["Store"] == store]
        for date in future_dates:
            week = int(date.isocalendar().week)
            nearby = store_hist.loc[
                store_hist["WeekOfYear"].between(week - 1, week + 1), "Temperature"
            ]
            rows.append(
                {
                    "Store": store,
                    "Date": date,
                    "Weekly_Sales": np.nan,
                    "Holiday_Flag": int(week in holiday_weeks),
                    "Temperature": nearby.mean()
                    if len(nearby)
                    else g["Temperature"].mean(),
                    "Fuel_Price": g["Fuel_Price"].iloc[-1],
                    "CPI": g["CPI"].iloc[-1],
                    "Unemployment": g["Unemployment"].iloc[-1],
                }
            )

    return pd.DataFrame(rows), future_dates
