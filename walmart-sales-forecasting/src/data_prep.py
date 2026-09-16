"""
Data loading and cleaning for the Walmart weekly sales dataset.

The raw file uses a DD-MM-YYYY date format. If you let pandas infer it, 05-02-2010
is silently read as May 2nd instead of Feb 5th, which quietly destroys every
seasonal conclusion downstream. We always parse it explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Common places the CSV might live, checked in order.
DEFAULT_CANDIDATES = (
    "data/Walmart_DataSet.csv",
    "Walmart_DataSet.csv",
    "walmart.csv",
    "data/walmart.csv",
)

EXPECTED_COLUMNS = [
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
]


def resolve_path(path: str | None = None) -> str:
    """Find the dataset, either at an explicit path or one of the usual spots."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No dataset at {path!r}")
        return path

    root = Path(__file__).resolve().parents[1]
    for candidate in DEFAULT_CANDIDATES:
        for base in (Path.cwd(), root):
            full = base / candidate
            if full.exists():
                return str(full)

    raise FileNotFoundError(
        "Could not locate the Walmart CSV. Tried: "
        + ", ".join(DEFAULT_CANDIDATES)
        + ". Pass an explicit path instead."
    )


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Read the CSV and parse dates with the correct day-first format."""
    resolved = resolve_path(path)
    df = pd.read_csv(resolved)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Store-aware missing-value treatment.

    The shipped dataset has no nulls, so this is a no-op in practice. It exists so
    the pipeline stays correct if you swap in a messier extract.

    The rule that matters: impute WITHIN each store, never across them. Store 33
    averages ~260k a week and Store 20 averages ~2.1M, so a global mean would be
    wrong for both of them.
    """
    d = df.sort_values(["Store", "Date"]).copy()

    # Sales are strongly autocorrelated week to week, so neighbouring weeks are
    # the best available guess.
    d["Weekly_Sales"] = d.groupby("Store")["Weekly_Sales"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )

    # CPI and Unemployment are reported monthly, so they are already step
    # functions - forward-fill IS the correct behaviour, not a compromise.
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        d[col] = d.groupby("Store")[col].transform(lambda s: s.ffill().bfill())

    # A missing flag almost certainly means "not a holiday".
    d["Holiday_Flag"] = d["Holiday_Flag"].fillna(0).astype(int)

    return d


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the calendar apart. A raw date is useless to most models."""
    d = df.copy()
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    d["MonthName"] = d["Date"].dt.month_name().str[:3]
    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["Quarter"] = d["Date"].dt.quarter
    d["Days_Since_Start"] = (d["Date"] - d["Date"].min()).dt.days
    return d


def load_clean(path: str | None = None) -> pd.DataFrame:
    """Load, clean and enrich in one call. This is the entry point most code wants."""
    df = load_raw(path)
    df = handle_missing(df)
    df = add_calendar_features(df)
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summary of the checks worth running before trusting any of this."""
    n_stores = df["Store"].nunique()
    n_weeks = df["Date"].nunique()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "stores": n_stores,
        "weeks": n_weeks,
        "date_min": df["Date"].min().date(),
        "date_max": df["Date"].max().date(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_store_dates": int(df.duplicated(subset=["Store", "Date"]).sum()),
        "balanced_panel": n_stores * n_weeks == len(df),
    }
