"""
End-to-end pipeline: clean -> analyse -> validate -> forecast -> export.

    python run_pipeline.py                 # full run
    python run_pipeline.py --no-figures    # skip plot generation (faster)

Writes CSVs to outputs/ and charts to outputs/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import analysis as A  # noqa: E402
from src import model as M  # noqa: E402
from src.data_prep import data_quality_report, load_clean  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

BLUE, RED, GREEN, GREY = "#3b6ea5", "#c0504d", "#2e7d32", "#a6a6a6"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold",
                     "savefig.bbox": "tight", "figure.autolayout": True})


def banner(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def save(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def make_figures(df, val, forecast) -> None:
    banner("GENERATING FIGURES")

    # 1. Chain-wide time series
    chain = df.groupby("Date")["Weekly_Sales"].sum()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(chain.index, chain.values, color=BLUE, lw=1.5, label="weekly total")
    ax.plot(chain.index, chain.rolling(8, center=True).mean(), color=RED, lw=2.4,
            label="8-week average")
    ax.set_title("Chain-wide weekly sales — two Christmas spikes dominate")
    ax.set_ylabel("Total weekly sales")
    ax.legend()
    save(fig, "01_chain_sales.png")

    # 2. Seasonality
    seas = A.seasonality_analysis(df)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(seas["by_week"].index, seas["by_week"].values, color=BLUE, lw=2)
    ax.axhline(df["Weekly_Sales"].mean(), color=RED, ls="--", label="overall average")
    top = seas["peak_weeks"]
    ax.scatter(top.index, top.values, color=RED, s=60, zorder=5)
    for w, v in top.items():
        ax.annotate(f"wk {w}", (w, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Week 51 runs +68% above average — the whole year in six weeks")
    ax.set_xlabel("ISO week")
    ax.set_ylabel("Avg weekly sales")
    ax.legend()
    save(fig, "02_seasonality.png")

    # 3. Store performance
    perf = A.store_performance(df)
    colors = [GREEN] * 10 + [GREY] * (len(perf) - 20) + [RED] * 10
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(perf.index.astype(str), perf["total_sales"], color=colors)
    ax.set_title("Total sales by store — an 8.1x gap between best and worst")
    ax.set_xlabel("Store")
    ax.set_ylabel("Total sales (143 weeks)")
    ax.tick_params(axis="x", labelsize=7)
    save(fig, "03_store_performance.png")

    # 4. Correlations
    corr = A.correlation_matrix(df)["Weekly_Sales"].drop("Weekly_Sales").sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(corr.index, corr.values,
            color=[RED if v < 0 else GREEN for v in corr.values])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(-0.15, 0.15)
    ax.set_title("Every external factor correlates at |r| < 0.11")
    save(fig, "04_correlations.png")

    # 5. Feature importance
    imp = val["importance"].head(10)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.invert_yaxis()
    ax.set_title("Feature importance — same week last year dominates")
    ax.set_xlabel("importance")
    save(fig, "05_feature_importance.png")

    # 6. Forecast
    chain_fc = forecast.groupby("Date")[["Forecast", "Lower_95", "Upper_95"]].sum()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(chain.index, chain.values, color=BLUE, lw=1.4, label="actual")
    ax.plot(chain_fc.index, chain_fc["Forecast"], color=RED, lw=2.4, marker="o",
            ms=4, label="forecast")
    ax.fill_between(chain_fc.index, chain_fc["Lower_95"], chain_fc["Upper_95"],
                    color=RED, alpha=0.18, label="95% interval")
    ax.axvline(df["Date"].max(), color=GREY, ls=":", lw=2)
    ax.set_title("12-week forecast — the model learned Christmas on its own")
    ax.set_ylabel("Total weekly sales")
    ax.legend()
    save(fig, "06_forecast.png")

    # 7. Model comparison
    comp = M.compare_models(df)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh(comp.index, comp["MAPE %"],
                   color=[GREEN if v == comp["MAPE %"].min() else GREY
                          for v in comp["MAPE %"]])
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontweight="bold")
    ax.set_xlim(0, comp["MAPE %"].max() * 1.2)
    ax.set_title("12-week-ahead forecast error (lower is better)")
    ax.set_xlabel("MAPE %")
    save(fig, "07_model_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="path to the CSV")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    banner("1. LOAD & CLEAN")
    df = load_clean(args.data)
    for k, v in data_quality_report(df).items():
        print(f"  {k:24} {v}")

    banner("2. OUTLIER ANALYSIS")
    out = A.outlier_analysis(df)
    print(f"  Globally extreme weeks : {out['global_count']} ({out['global_pct']}%)")
    print(f"  Months they fall in    : {dict(out['global_by_month'])}")
    print(f"  Per-store outliers     : {out['per_store_count']}")
    print("  -> All in Nov/Dec. These are Thanksgiving and Christmas, not errors.")
    print("     Deleting them would make the model blind to peak season. Keeping them.")

    banner("3. BUSINESS QUESTIONS")
    unemp = A.unemployment_impact(df)
    print(f"  (a) Unemployment  pooled r = {unemp['pooled_r']:.3f} "
          f"({unemp['pooled_r2_pct']:.1f}% of variance)")
    print(f"      Most sensitive stores: "
          f"{list(unemp['most_sensitive'].index[:3])}")

    seas = A.seasonality_analysis(df)
    print(f"  (b) Seasonality   peak = week {int(seas['by_week'].idxmax())} "
          f"(+{seas['week_lift_pct'].max():.0f}%), "
          f"holiday-flag lift only +{seas['holiday_lift_pct']:.1f}%")

    temp = A.temperature_impact(df)
    print(f"  (c) Temperature   pooled r = {temp['pooled_r']:.3f} "
          f"({temp['pooled_r2_pct']:.1f}% of variance), non-linear")

    cpi = A.cpi_impact(df)
    print(f"  (d) CPI           pooled r = {cpi['pooled_r']:.3f}; varies "
          f"{cpi['between_store_range']:.0f} pts between stores vs "
          f"{cpi['median_within_store_drift']:.0f} within")

    gap = A.performance_gap(df)
    perf = gap["performance"]
    print(f"  (e) Top stores    {list(perf.index[:5])} "
          f"({gap['top10_share_pct']:.1f}% of revenue from top 10)")
    print(f"  (f) Worst store   {gap['worst_store']} — {gap['ratio']:.2f}x below "
          f"store {gap['best_store']}, Cohen's d = {gap['cohens_d']:.2f}, "
          f"no overlap: {gap['no_overlap']}")

    banner("4. MODEL VALIDATION")
    val = M.train_validate(df)
    for k, v in val["metrics"].items():
        print(f"  {k:6} {v:,.4f}")
    print("\n  Model comparison:")
    print(M.compare_models(df).to_string())

    banner("5. FORECAST")
    forecast = M.forecast_future(df, val["store_mape"])
    summary = M.store_summary(forecast, df)
    total = summary["forecast_total"].sum()
    recent = summary["recent_actual_total"].sum()
    print(f"  Window        : {forecast['Date'].min().date()} to "
          f"{forecast['Date'].max().date()}")
    print(f"  Chain total   : {total:,.0f}")
    print(f"  vs prev 12 wks: {100*(total/recent-1):+.1f}%")

    banner("6. EXPORT")
    forecast.to_csv(OUT / "walmart_forecast_next_12_weeks.csv", index=False)
    summary.to_csv(OUT / "walmart_store_forecast_summary.csv")
    perf.to_csv(OUT / "walmart_store_performance_ranking.csv")
    for name in ("walmart_forecast_next_12_weeks.csv",
                 "walmart_store_forecast_summary.csv",
                 "walmart_store_performance_ranking.csv"):
        print(f"  wrote outputs/{name}")

    if not args.no_figures:
        make_figures(df, val, forecast)

    banner("DONE")


if __name__ == "__main__":
    main()
