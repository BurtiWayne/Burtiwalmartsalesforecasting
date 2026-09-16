"""
The six business questions, as reusable functions.

A recurring theme across all of these: the pooled, chain-wide statistic is
consistently misleading. Every external factor correlates with sales at |r| < 0.11
when you pool the stores together, which tempts you to conclude "nothing matters".
Break it out per store and real, actionable relationships appear in a handful of
outlets. Each function below therefore returns both views.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_analysis(df: pd.DataFrame) -> dict:
    """
    Outliers judged two ways: globally, and within each store.

    Only the second is meaningful. A 900k week is routine for Store 20 but would be
    an all-time record for Store 33, and only the per-store test can see that.

    The finding that matters: the globally extreme weeks are ALL in November and
    December. They are Thanksgiving and Christmas, not data errors, and they are
    precisely the weeks the inventory team most needs to plan for. We keep them.
    """
    lo, hi = iqr_bounds(df["Weekly_Sales"])
    global_out = df[(df["Weekly_Sales"] < lo) | (df["Weekly_Sales"] > hi)]

    per_store, flagged = [], []
    for store, g in df.groupby("Store"):
        lo_s, hi_s = iqr_bounds(g["Weekly_Sales"])
        mask = (g["Weekly_Sales"] < lo_s) | (g["Weekly_Sales"] > hi_s)
        per_store.append(
            {
                "Store": store,
                "n_outliers": int(mask.sum()),
                "pct": round(100 * mask.mean(), 1),
                "lower": lo_s,
                "upper": hi_s,
            }
        )
        flagged.append(g[mask])

    flagged = pd.concat(flagged) if flagged else pd.DataFrame()

    return {
        "global_bounds": (lo, hi),
        "global_outliers": global_out,
        "global_count": len(global_out),
        "global_pct": round(100 * len(global_out) / len(df), 2),
        "global_by_month": global_out["Date"].dt.month_name().str[:3].value_counts(),
        "per_store": pd.DataFrame(per_store).set_index("Store"),
        "per_store_count": int(sum(r["n_outliers"] for r in per_store)),
        "flagged_rows": flagged,
    }


# --------------------------------------------------------------------------- #
# (a) Unemployment
# --------------------------------------------------------------------------- #
def unemployment_impact(df: pd.DataFrame) -> dict:
    r, p = stats.pearsonr(df["Weekly_Sales"], df["Unemployment"])

    rows = []
    for store, g in df.groupby("Store"):
        corr, pval = stats.pearsonr(g["Weekly_Sales"], g["Unemployment"])
        rows.append(
            {
                "Store": store,
                "corr_unemp": corr,
                "p_value": pval,
                "avg_unemp": g["Unemployment"].mean(),
                "unemp_range": g["Unemployment"].max() - g["Unemployment"].min(),
                "avg_sales": g["Weekly_Sales"].mean(),
            }
        )

    per_store = pd.DataFrame(rows).set_index("Store").round(3)
    per_store["significant"] = per_store["p_value"] < 0.05

    return {
        "pooled_r": r,
        "pooled_p": p,
        "pooled_r2_pct": 100 * r**2,
        "per_store": per_store,
        # Stores whose sales genuinely move with the local labour market.
        "most_sensitive": per_store.nsmallest(10, "corr_unemp"),
        # Stores that simply sit in bad labour markets, a different question.
        "worst_markets": per_store.nlargest(10, "avg_unemp"),
        "n_negative": int((per_store["corr_unemp"] < 0).sum()),
        "n_positive": int((per_store["corr_unemp"] > 0).sum()),
    }


# --------------------------------------------------------------------------- #
# (b) Seasonality
# --------------------------------------------------------------------------- #
def seasonality_analysis(df: pd.DataFrame) -> dict:
    overall = df["Weekly_Sales"].mean()
    by_week = df.groupby(df["Date"].dt.isocalendar().week.astype(int))["Weekly_Sales"].mean()
    by_month = df.groupby(df["Date"].dt.month)["Weekly_Sales"].mean()

    holiday = df.loc[df["Holiday_Flag"] == 1, "Weekly_Sales"]
    normal = df.loc[df["Holiday_Flag"] == 0, "Weekly_Sales"]
    t_stat, p_val = stats.ttest_ind(holiday, normal, equal_var=False)

    week_lift = (100 * (by_week / overall - 1)).round(1)

    return {
        "by_week": by_week,
        "by_month": by_month,
        "week_lift_pct": week_lift,
        "peak_weeks": by_week.nlargest(6),
        "trough_weeks": by_week.nsmallest(4),
        "holiday_mean": holiday.mean(),
        "normal_mean": normal.mean(),
        "holiday_lift_pct": 100 * (holiday.mean() - normal.mean()) / normal.mean(),
        "holiday_t": t_stat,
        "holiday_p": p_val,
        # The catch: Holiday_Flag marks the week CONTAINING each holiday. Christmas
        # falls in week 52, but the shopping happens in week 51 - which is not
        # flagged at all, despite being the biggest week of the year.
        "flag_misses_peak": int(by_week.idxmax()) not in set(
            df.loc[df["Holiday_Flag"] == 1, "Date"].dt.isocalendar().week.unique()
        ),
    }


# --------------------------------------------------------------------------- #
# (c) Temperature
# --------------------------------------------------------------------------- #
def temperature_impact(df: pd.DataFrame) -> dict:
    r, p = stats.pearsonr(df["Weekly_Sales"], df["Temperature"])

    bins = [-10, 30, 50, 70, 90, 110]
    labels = [
        "Freezing (<30F)",
        "Cold (30-50F)",
        "Mild (50-70F)",
        "Warm (70-90F)",
        "Hot (>90F)",
    ]
    banded = df.assign(TempBand=pd.cut(df["Temperature"], bins=bins, labels=labels))
    by_band = banded.groupby("TempBand", observed=True)["Weekly_Sales"].agg(
        ["mean", "median", "count"]
    )
    by_band["vs_overall_pct"] = (
        100 * (by_band["mean"] / df["Weekly_Sales"].mean() - 1)
    ).round(1)

    per_store = df.groupby("Store").apply(
        lambda g: g["Weekly_Sales"].corr(g["Temperature"]), include_groups=False
    )

    return {
        "pooled_r": r,
        "pooled_p": p,
        "pooled_r2_pct": 100 * r**2,
        "by_band": by_band.round(0),
        "per_store": per_store.round(3),
        "n_negative": int((per_store < 0).sum()),
        "most_sensitive": per_store.nsmallest(5),
        # The relationship is an inverted U, not a line. Sales peak in the 30-50F
        # band (which is mostly just November-December in disguise) and collapse
        # above 90F, which is a genuine heat-suppression effect.
        "note": "Non-linear: cold-band peak is a season proxy; hot-band drop is real.",
    }


# --------------------------------------------------------------------------- #
# (d) CPI
# --------------------------------------------------------------------------- #
def cpi_impact(df: pd.DataFrame) -> dict:
    r, p = stats.pearsonr(df["Weekly_Sales"], df["CPI"])

    rows = []
    for store, g in df.groupby("Store"):
        rows.append(
            {
                "Store": store,
                "corr_cpi": g["CPI"].corr(g["Weekly_Sales"]),
                "cpi_mean": g["CPI"].mean(),
                "cpi_drift": g["CPI"].max() - g["CPI"].min(),
                "avg_sales": g["Weekly_Sales"].mean(),
            }
        )
    per_store = pd.DataFrame(rows).set_index("Store").round(3)

    banded = df.assign(
        CPI_Band=pd.cut(
            df["CPI"],
            bins=[0, 140, 180, 220, 250],
            labels=["Low (<140)", "Mid (140-180)", "High (180-220)", "Very High (>220)"],
        )
    )

    return {
        "pooled_r": r,
        "pooled_p": p,
        "pooled_r2_pct": 100 * r**2,
        "per_store": per_store,
        "by_band": banded.groupby("CPI_Band", observed=True)["Weekly_Sales"].mean().round(0),
        "most_negative": per_store.nsmallest(5, "corr_cpi"),
        "most_positive": per_store.nlargest(5, "corr_cpi"),
        # The structural point: CPI varies ~99 points BETWEEN stores but only ~11
        # points WITHIN a store over three years. It is primarily a regional label
        # and only secondarily a time series.
        "between_store_range": per_store["cpi_mean"].max() - per_store["cpi_mean"].min(),
        "median_within_store_drift": per_store["cpi_drift"].median(),
    }


# --------------------------------------------------------------------------- #
# (e) and (f) Store performance
# --------------------------------------------------------------------------- #
def store_performance(df: pd.DataFrame) -> pd.DataFrame:
    perf = df.groupby("Store")["Weekly_Sales"].agg(
        total_sales="sum",
        avg_weekly="mean",
        median_weekly="median",
        std_weekly="std",
        best_week="max",
        worst_week="min",
    ).round(0)
    perf["cv_pct"] = (100 * perf["std_weekly"] / perf["avg_weekly"]).round(1)
    perf["share_pct"] = (100 * perf["total_sales"] / perf["total_sales"].sum()).round(2)
    perf = perf.sort_values("total_sales", ascending=False)
    perf["rank"] = range(1, len(perf) + 1)
    return perf


def performance_gap(df: pd.DataFrame) -> dict:
    """
    How significant is the best-vs-worst gap? By every measure, extremely.

    But an 8x gap almost certainly does not mean the worst store is badly run.
    Gaps this clean and this stable are structural - store format, square footage,
    catchment population. The worst store here is also one of the least volatile in
    the chain, which says "small store, consistent" rather than "large store,
    failing". Benchmark it against similarly-sized peers instead.
    """
    perf = store_performance(df)
    best, worst = perf.index[0], perf.index[-1]

    b = df.loc[df["Store"] == best, "Weekly_Sales"]
    w = df.loc[df["Store"] == worst, "Weekly_Sales"]

    t_stat, p_val = stats.ttest_ind(b, w, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(b, w)
    pooled_sd = np.sqrt((b.std() ** 2 + w.std() ** 2) / 2)
    f_stat, p_anova = stats.f_oneway(
        *[g["Weekly_Sales"].values for _, g in df.groupby("Store")]
    )

    return {
        "performance": perf,
        "best_store": int(best),
        "worst_store": int(worst),
        "best_total": float(b.sum()),
        "worst_total": float(w.sum()),
        "absolute_gap": float(b.sum() - w.sum()),
        "ratio": float(b.sum() / w.sum()),
        "t_stat": float(t_stat),
        "t_p_value": float(p_val),
        "mannwhitney_p": float(p_u),
        "cohens_d": float((b.mean() - w.mean()) / pooled_sd),
        # If this is True, the best store's worst week still beats the worst
        # store's best week - the distributions do not overlap at all.
        "no_overlap": bool(b.min() > w.max()),
        "anova_f": float(f_stat),
        "anova_p": float(p_anova),
        "top10_share_pct": float(perf["share_pct"].head(10).sum()),
        "bottom10_share_pct": float(perf["share_pct"].tail(10).sum()),
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Weekly_Sales",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
    ]
    return df[cols].corr()
