"""
Forecasting models and validation.

Four approaches, compared on the same honest holdout:

  1. Naive           - next 12 weeks = last observed week. The floor.
  2. Seasonal naive  - next 12 weeks = same 12 weeks last year. A genuinely
                       strong baseline in seasonal retail.
  3. Holt-Winters    - classical triple exponential smoothing, fitted per store.
  4. Random Forest   - one global model across all stores, leak-free features.

Validation is a time-ordered holdout of the final 12 weeks. Never a random split:
that would scatter future weeks into training and produce a meaningless score.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import (
    FEATURE_COLUMNS,
    HORIZON,
    TARGET,
    build_features,
    build_future_frame,
    drop_warmup,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

RAW_COLUMNS = [
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mape(actual, pred) -> float:
    """Mean absolute percentage error.

    Scale-free, which matters here: stores range from 260k to 2.1M a week, so raw
    MAE would be dominated entirely by the large stores.
    """
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def rmse(actual, pred) -> float:
    return float(np.sqrt(mean_squared_error(actual, pred)))


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_scores(df: pd.DataFrame, horizon: int = HORIZON) -> dict[str, float]:
    """Naive and seasonal-naive MAPE, averaged across stores."""
    naive, snaive = [], []
    for _, g in df.groupby("Store"):
        y = g.sort_values("Date").set_index("Date")[TARGET]
        train, test = y[:-horizon], y[-horizon:]
        naive.append(mape(test.values, np.repeat(train.iloc[-1], horizon)))
        snaive.append(mape(test.values, train.iloc[-52 : -52 + horizon].values))
    return {
        "Naive (last value)": float(np.mean(naive)),
        "Seasonal Naive (t-52)": float(np.mean(snaive)),
    }


def holt_winters_score(df: pd.DataFrame, horizon: int = HORIZON) -> float:
    """Per-store triple exponential smoothing with a 52-week season."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    errors = []
    for _, g in df.groupby("Store"):
        y = g.sort_values("Date").set_index("Date")[TARGET].asfreq("W-FRI")
        train, test = y[:-horizon], y[-horizon:]
        try:
            fit = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=52,
                initialization_method="estimated",
            ).fit()
            errors.append(mape(test.values, fit.forecast(horizon).values))
        except Exception:
            errors.append(np.nan)
    return float(np.nanmean(errors))


# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #
def make_model(n_estimators: int = 400) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_validate(df: pd.DataFrame, horizon: int = HORIZON) -> dict:
    """
    Fit on everything before the final `horizon` weeks, score on those weeks.

    Returns the fitted model, the scored test frame, headline metrics, per-store
    error, and feature importances.
    """
    feat = drop_warmup(build_features(df[RAW_COLUMNS], horizon))

    split_date = feat["Date"].max() - pd.Timedelta(weeks=horizon - 1)
    train = feat[feat["Date"] < split_date]
    test = feat[feat["Date"] >= split_date].copy()

    model = make_model()
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    test["prediction"] = model.predict(test[FEATURE_COLUMNS])
    test["abs_pct_err"] = 100 * np.abs(test[TARGET] - test["prediction"]) / test[TARGET]

    return {
        "model": model,
        "train": train,
        "test": test,
        "split_date": split_date,
        "metrics": {
            "MAPE": mape(test[TARGET], test["prediction"]),
            "MAE": float(mean_absolute_error(test[TARGET], test["prediction"])),
            "RMSE": rmse(test[TARGET], test["prediction"]),
            "R2": float(r2_score(test[TARGET], test["prediction"])),
        },
        "store_mape": test.groupby("Store")["abs_pct_err"].mean(),
        "importance": pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False),
    }


def compare_models(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """All four approaches on the same holdout, ranked by MAPE."""
    scores = baseline_scores(df, horizon)
    scores["Holt-Winters"] = holt_winters_score(df, horizon)
    scores["Random Forest"] = train_validate(df, horizon)["metrics"]["MAPE"]

    out = pd.DataFrame({"MAPE %": pd.Series(scores)}).sort_values("MAPE %")
    out["vs best"] = (out["MAPE %"] / out["MAPE %"].min()).round(2)
    return out.round(2)


def forecast_future(
    df: pd.DataFrame,
    store_mape: pd.Series | None = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Refit on all available data and predict the next `horizon` weeks per store.

    Validation is over by this point and the model is chosen, so throwing away the
    last 12 weeks would just be wasteful.

    If `store_mape` is supplied (from train_validate), each point forecast gets a
    95% band derived from that store's own validated error - an honest interval
    rather than a made-up one.
    """
    history = df[RAW_COLUMNS].copy()

    full = drop_warmup(build_features(history, horizon))
    model = make_model(n_estimators=500)
    model.fit(full[FEATURE_COLUMNS], full[TARGET])

    future, future_dates = build_future_frame(history, horizon)
    combined = build_features(pd.concat([history, future], ignore_index=True), horizon)
    rows = combined[combined["Date"].isin(future_dates)].copy()

    # Safety net for any store whose history is too short for a clean lag.
    for col in ["lag_52", "lag_53", "roll_12", "roll_52", "seasonal_index"]:
        if rows[col].isna().any():
            rows[col] = rows.groupby("Store")[col].transform(lambda s: s.fillna(s.mean()))
            rows[col] = rows[col].fillna(rows[col].median())

    rows["Forecast"] = model.predict(rows[FEATURE_COLUMNS]).round(0)

    out = rows[["Store", "Date", "Forecast"]].copy()
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)

    if store_mape is not None:
        err = out["Store"].map(store_mape).fillna(float(store_mape.mean()))
        out["Lower_95"] = (out["Forecast"] * (1 - 1.96 * err / 100)).round(0)
        out["Upper_95"] = (out["Forecast"] * (1 + 1.96 * err / 100)).round(0)
        out["Expected_MAPE"] = err.round(2)

    return out.sort_values(["Store", "Date"]).reset_index(drop=True)


def store_summary(forecast: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Roll the forecast up per store and compare it to the previous 12 weeks.

    The change_% column is the actual inventory instruction: a store at +20% needs
    a real stock build, a store near flat does not.
    """
    horizon_weeks = forecast["Date"].nunique()
    last_date = history["Date"].max()
    recent = history[history["Date"] > last_date - pd.Timedelta(weeks=horizon_weeks)]

    summary = forecast.groupby("Store").agg(
        forecast_total=("Forecast", "sum"),
        avg_weekly=("Forecast", "mean"),
        peak_week_sales=("Forecast", "max"),
    ).round(0)

    summary["recent_actual_total"] = (
        recent.groupby("Store")["Weekly_Sales"].sum().round(0)
    )
    summary["change_pct"] = (
        100 * (summary["forecast_total"] / summary["recent_actual_total"] - 1)
    ).round(1)
    summary["peak_week"] = (
        forecast.loc[forecast.groupby("Store")["Forecast"].idxmax()]
        .set_index("Store")["Date"]
        .dt.date
    )

    return summary.sort_values("forecast_total", ascending=False)
