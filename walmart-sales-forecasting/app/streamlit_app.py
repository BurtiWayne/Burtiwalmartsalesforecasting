"""
Walmart Retail Sales - interactive dashboard.

Run locally:
    streamlit run app/streamlit_app.py

Everything heavy (model fitting, forecasting) is cached, so the first load takes
roughly 30-60 seconds and every interaction after that is instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `src` importable whether you run from the repo root or elsewhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import analysis as A  # noqa: E402
from src import model as M  # noqa: E402
from src.data_prep import data_quality_report, load_clean  # noqa: E402

st.set_page_config(
    page_title="Walmart Sales Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE, RED, GREEN, GREY = "#3b6ea5", "#c0504d", "#2e7d32", "#a6a6a6"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      .insight {
        background: rgba(59,110,165,.08);
        border-left: 4px solid #3b6ea5;
        padding: .85rem 1.1rem; border-radius: 6px; margin: .6rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached data and model layer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data...")
def get_data() -> pd.DataFrame:
    return load_clean()


@st.cache_data(show_spinner="Validating model on held-out weeks...")
def get_validation(_df: pd.DataFrame) -> dict:
    res = M.train_validate(_df)
    # Drop the unpicklable model object before caching.
    return {
        "metrics": res["metrics"],
        "test": res["test"][
            ["Store", "Date", "Weekly_Sales", "prediction", "abs_pct_err"]
        ],
        "store_mape": res["store_mape"],
        "importance": res["importance"],
        "split_date": res["split_date"],
    }


@st.cache_data(show_spinner="Comparing models...")
def get_comparison(_df: pd.DataFrame) -> pd.DataFrame:
    scores = M.baseline_scores(_df)
    scores["Holt-Winters"] = M.holt_winters_score(_df)
    scores["Random Forest"] = get_validation(_df)["metrics"]["MAPE"]
    out = pd.DataFrame({"MAPE %": pd.Series(scores)}).sort_values("MAPE %")
    out["vs best"] = (out["MAPE %"] / out["MAPE %"].min()).round(2)
    return out.round(2)


@st.cache_data(show_spinner="Generating the 12-week forecast...")
def get_forecast(_df: pd.DataFrame, _store_mape: pd.Series) -> pd.DataFrame:
    return M.forecast_future(_df, _store_mape)


df = get_data()
quality = data_quality_report(df)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛒 Walmart Analytics")
    st.caption("Demand analysis & 12-week forecasting across 45 outlets")

    page = st.radio(
        "Section",
        [
            "Overview",
            "Seasonality",
            "Economic Factors",
            "Store Performance",
            "Forecast",
            "Model Details",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        f"**{quality['rows']:,}** rows · **{quality['stores']}** stores · "
        f"**{quality['weeks']}** weeks\n\n"
        f"{quality['date_min']} → {quality['date_max']}"
    )
    if quality["balanced_panel"]:
        st.success("Balanced panel · 0 nulls", icon="✅")

    st.divider()
    st.caption(
        "Built with pandas, scikit-learn & Plotly.\n\n"
        "[Source on GitHub](https://github.com/)"
    )


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
if page == "Overview":
    st.title("Retail Demand Analysis")
    st.markdown(
        "A retail chain with 45 outlets is struggling to match inventory to demand. "
        "This dashboard works through what actually drives their sales — and what "
        "demonstrably does not."
    )

    perf = A.store_performance(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", f"${df['Weekly_Sales'].sum()/1e9:.2f}B")
    c2.metric("Avg weekly / store", f"${df['Weekly_Sales'].mean()/1e3:,.0f}K")
    c3.metric("Best vs worst store", f"{perf['total_sales'].max()/perf['total_sales'].min():.1f}x")
    c4.metric("Forecast accuracy", "3.93% MAPE", "12 weeks ahead")

    st.divider()

    chain = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chain["Date"],
            y=chain["Weekly_Sales"],
            mode="lines",
            line=dict(color=BLUE, width=1.6),
            name="Weekly total",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chain["Date"],
            y=chain["Weekly_Sales"].rolling(8, center=True).mean(),
            mode="lines",
            line=dict(color=RED, width=2.5),
            name="8-week average",
        )
    )
    fig.update_layout(
        title="Chain-wide weekly sales, Feb 2010 – Oct 2012",
        height=420,
        hovermode="x unified",
        yaxis_title="Total weekly sales",
    )
    st.plotly_chart(fig, width="stretch")

    insight(
        "<b>Two spikes dominate everything.</b> Both are late December. The entire "
        "year's inventory risk is concentrated into roughly six weeks."
    )

    st.subheader("What actually predicts sales?")
    corr = A.correlation_matrix(df)["Weekly_Sales"].drop("Weekly_Sales").sort_values()
    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        color=corr.values,
        color_continuous_scale="RdYlGn",
        labels={"x": "Correlation with weekly sales", "y": ""},
    )
    fig.update_layout(height=300, coloraxis_showscale=False, xaxis_range=[-0.15, 0.15])
    st.plotly_chart(fig, width="stretch")

    insight(
        "<b>The headline finding of the whole project.</b> Every external factor — "
        "unemployment, CPI, temperature, fuel price, holiday flag — correlates with "
        "sales at |r| &lt; 0.11. Together they explain barely 2% of the variation. "
        "What actually explains sales is <b>which store it is</b> and <b>which week "
        "of the year it is</b>. This business runs on the calendar, not the economy."
    )

    st.subheader("Outliers: the case for keeping them")
    out = A.outlier_analysis(df)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Globally extreme weeks", out["global_count"])
        st.metric("Per-store outliers", out["per_store_count"])
        st.caption("Judged against each store's own normal range")
    with c2:
        by_month = out["global_by_month"]
        fig = px.bar(
            x=by_month.index, y=by_month.values,
            color_discrete_sequence=[RED],
            labels={"x": "", "y": "outlier weeks"},
        )
        fig.update_layout(height=250, title="When do the extreme weeks occur?")
        st.plotly_chart(fig, width="stretch")

    insight(
        "All 34 globally extreme weeks fall in <b>November and December</b>. These "
        "are Thanksgiving and Christmas — not data errors, and precisely the weeks "
        "the inventory team most needs to get right. The standard move of dropping "
        "IQR outliers would delete Christmas from the model. <b>We keep them.</b>"
    )


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #
elif page == "Seasonality":
    st.title("Seasonal Patterns")
    st.caption("Question (b): is there a seasonal trend — when, and why?")

    seas = A.seasonality_analysis(df)

    c1, c2, c3, c4 = st.columns(4)
    peak_week = int(seas["by_week"].idxmax())
    c1.metric("Peak week", f"Week {peak_week}", f"+{seas['week_lift_pct'].max():.0f}% vs avg")
    c2.metric("Thanksgiving (wk 47)", f"${seas['by_week'].get(47, 0)/1e6:.2f}M",
              f"+{seas['week_lift_pct'].get(47, 0):.0f}%")
    c3.metric("Holiday-flag lift", f"+{seas['holiday_lift_pct']:.1f}%",
              f"p = {seas['holiday_p']:.3f}")
    c4.metric("January trough", f"${seas['by_month'][1]/1e6:.2f}M", "weakest month")

    week = seas["by_week"].reset_index()
    week.columns = ["Week", "AvgSales"]
    fig = px.line(week, x="Week", y="AvgSales", markers=True)
    fig.update_traces(line=dict(color=BLUE, width=2.2), marker=dict(size=5))
    fig.add_hline(y=df["Weekly_Sales"].mean(), line_dash="dash", line_color=RED,
                  annotation_text="overall average")
    top = seas["peak_weeks"]
    fig.add_trace(
        go.Scatter(x=top.index, y=top.values, mode="markers+text",
                   marker=dict(size=13, color=RED),
                   text=[f"wk {w}" for w in top.index],
                   textposition="top center", name="peak weeks")
    )
    fig.update_layout(title="Average sales by week of year", height=430,
                      yaxis_title="Avg weekly sales")
    st.plotly_chart(fig, width="stretch")

    if seas["flag_misses_peak"]:
        insight(
            "<b>A real modelling trap.</b> The <code>Holiday_Flag</code> marks the week "
            "<i>containing</i> each holiday. Christmas Day falls in week 52, but the "
            "shopping happens in <b>week 51</b> — which is not flagged at all, despite "
            "being the biggest week of the year by a wide margin. That is why the "
            "flag only shows a +7.8% lift while week 51 runs +68%. "
            "<b>Week-of-year beats Holiday_Flag as a feature.</b>"
        )

    c1, c2 = st.columns(2)
    with c1:
        month = seas["by_month"].reset_index()
        month.columns = ["Month", "AvgSales"]
        month["Name"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = px.bar(month, x="Name", y="AvgSales", color_discrete_sequence=[BLUE])
        fig.add_hline(y=df["Weekly_Sales"].mean(), line_dash="dash", line_color=RED)
        fig.update_layout(title="Average sales by month", height=360,
                          xaxis_title="", yaxis_title="Avg weekly sales")
        st.plotly_chart(fig, width="stretch")

    with c2:
        pivot = df.pivot_table(index="WeekOfYear", columns="Year",
                               values="Weekly_Sales", aggfunc="mean")
        fig = go.Figure()
        for yr in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[yr],
                                     mode="lines", name=str(yr)))
        fig.update_layout(title="Same profile, repeated each year", height=360,
                          xaxis_title="ISO week", yaxis_title="Avg weekly sales")
        st.plotly_chart(fig, width="stretch")

    insight(
        "The 2010 and 2011 curves track each other closely — same shape, same peaks "
        "in the same weeks. <b>That repeatability is what makes the season "
        "forecastable.</b> If the years disagreed, we would have noise rather than "
        "seasonality."
    )

    st.subheader("Why it happens")
    st.markdown(
        """
| Period | Weeks | Behaviour | Driver |
|---|---|---|---|
| **Peak** | Week 51 | **+68%** | Pre-Christmas gifting, food, decorations |
| **Secondary peak** | Week 47 | **+41%** | Black Friday discounting |
| **Shoulder** | Weeks 48–50 | +5% to +29% | Christmas build-up |
| **Trough** | Weeks 1–4 | **−12%** | Post-holiday hangover, credit-card bills, weather |
| **Minor bump** | Week 22 | +4% | Memorial Day, start of summer season |

The mid-year is remarkably flat — June through August sit within a couple of
percent of each other. Outside the Q4 window this is a stable, predictable business.
        """
    )


# --------------------------------------------------------------------------- #
# Economic factors
# --------------------------------------------------------------------------- #
elif page == "Economic Factors":
    st.title("Economic & Environmental Factors")
    st.caption("Questions (a), (c), (d): unemployment, temperature and CPI")

    tab1, tab2, tab3 = st.tabs(["Unemployment", "Temperature", "CPI"])

    with tab1:
        unemp = A.unemployment_impact(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pooled correlation", f"{unemp['pooled_r']:.3f}",
                  f"explains {unemp['pooled_r2_pct']:.1f}% of variance")
        c2.metric("Stores negatively affected", f"{unemp['n_negative']} / 45")
        c3.metric("Stores positively affected", f"{unemp['n_positive']} / 45")

        ps = unemp["per_store"].sort_values("corr_unemp")
        fig = px.bar(x=ps.index.astype(str), y=ps["corr_unemp"],
                     color=ps["corr_unemp"], color_continuous_scale="RdYlGn",
                     labels={"x": "Store", "y": "Correlation"})
        fig.update_layout(title="Sales vs unemployment, by store", height=400,
                          coloraxis_showscale=False)
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")

        insight(
            "<b>The pooled statistic hides everything that matters.</b> Chain-wide, "
            f"r = {unemp['pooled_r']:.3f} would lead you to say unemployment is "
            "irrelevant. Break it out and <b>Store 38 (r = −0.79)</b> and "
            "<b>Store 44 (r = −0.78)</b> are severely exposed, while 16 stores show "
            "<i>positive</i> correlations — consistent with trade-down behaviour, "
            "where shoppers switch to a value retailer during a downturn."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Most sensitive to unemployment changes**")
            st.dataframe(unemp["most_sensitive"][
                ["corr_unemp", "avg_unemp", "avg_sales"]].head(6),
                use_container_width=True)
        with c2:
            st.markdown("**Operating in the worst labour markets**")
            st.dataframe(unemp["worst_markets"][
                ["avg_unemp", "corr_unemp", "avg_sales"]].head(6),
                use_container_width=True)

        insight(
            "<b>Store 38 appears on both lists</b> — high unemployment, high "
            "sensitivity to it, and low volume. That is the most economically "
            "fragile outlet in the chain and the first place a recession plan "
            "should look."
        )

    with tab2:
        temp = A.temperature_impact(df)
        c1, c2 = st.columns(2)
        c1.metric("Pooled correlation", f"{temp['pooled_r']:.3f}",
                  f"explains {temp['pooled_r2_pct']:.1f}% of variance")
        c2.metric("Stores negatively affected", f"{temp['n_negative']} / 45")

        band = temp["by_band"].reset_index()
        band.columns = ["Band", "Mean", "Median", "Count", "VsOverall"]
        fig = px.bar(band, x="Band", y="Mean", text="VsOverall",
                     color="VsOverall", color_continuous_scale="RdYlGn")
        fig.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
        fig.add_hline(y=df["Weekly_Sales"].mean(), line_dash="dash", line_color=RED)
        fig.update_layout(title="Average sales by temperature band", height=400,
                          xaxis_title="", yaxis_title="Avg weekly sales",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

        insight(
            "<b>The relationship is an inverted U, not a line</b> — which is exactly "
            "why the linear correlation looks so weak. Sales peak in the 30–50°F "
            "band and fall <b>24% below average above 90°F</b>. But be careful: the "
            "cold-band peak is largely <b>Christmas in disguise</b>, since that band "
            "contains November and December in most regions. The hot-weather drop is "
            "more likely to be real footfall suppression."
        )

        st.markdown(
            "**Practical read:** don't plan inventory <i>volume</i> off temperature. "
            "Use it for <b>category mix</b> — cold snaps shift demand toward hot food "
            "and heating, heat waves toward beverages and cooling.",
            unsafe_allow_html=True,
        )

    with tab3:
        cpi = A.cpi_impact(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pooled correlation", f"{cpi['pooled_r']:.3f}",
                  f"explains {cpi['pooled_r2_pct']:.1f}% of variance")
        c2.metric("CPI spread between stores", f"{cpi['between_store_range']:.0f} pts")
        c3.metric("Typical drift within a store", f"{cpi['median_within_store_drift']:.0f} pts")

        insight(
            "<b>The structural insight for this question.</b> CPI varies by ~91 points "
            "<i>between</i> stores but only ~10 points <i>within</i> a store across "
            "three years. CPI here is primarily a <b>regional label</b> and only "
            "secondarily a time series — the stores fall into distinct cost-of-living "
            "clusters that almost certainly map to different metro areas."
        )

        ps = cpi["per_store"]
        fig = px.scatter(ps, x="cpi_mean", y="avg_sales", color="corr_cpi",
                         size="cpi_drift", color_continuous_scale="RdYlGn",
                         hover_name=ps.index,
                         labels={"cpi_mean": "Store average CPI",
                                 "avg_sales": "Average weekly sales",
                                 "corr_cpi": "within-store r"})
        fig.update_layout(title="CPI level vs sales, coloured by within-store correlation",
                          height=430)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Genuinely hurt by rising CPI**")
            st.dataframe(cpi["most_negative"][["corr_cpi", "cpi_mean", "avg_sales"]],
                         use_container_width=True)
        with c2:
            st.markdown("**Positively associated with CPI**")
            st.dataframe(cpi["most_positive"][["corr_cpi", "cpi_mean", "avg_sales"]],
                         use_container_width=True)

        insight(
            "<b>Store 36 (r = −0.92)</b> is the one store where inflation genuinely "
            "looks like it suppresses demand. The positive cases (38, 44) sit in the "
            "<i>lowest</i>-CPI region — there, rising CPI and rising sales are both "
            "symptoms of a recovering local economy, not cause and effect. "
            "<b>Roughly 25 stores show no usable relationship at all.</b>"
        )


# --------------------------------------------------------------------------- #
# Store performance
# --------------------------------------------------------------------------- #
elif page == "Store Performance":
    st.title("Store Performance")
    st.caption("Questions (e) and (f): top performers, worst performer, and the gap")

    gap = A.performance_gap(df)
    perf = gap["performance"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Best · Store {gap['best_store']}", f"${gap['best_total']/1e6:.0f}M")
    c2.metric(f"Worst · Store {gap['worst_store']}", f"${gap['worst_total']/1e6:.0f}M")
    c3.metric("Ratio", f"{gap['ratio']:.2f}x", f"gap ${gap['absolute_gap']/1e6:.0f}M")
    c4.metric("Cohen's d", f"{gap['cohens_d']:.2f}", "0.8 is a 'large' effect")

    colors = [GREEN] * 10 + [GREY] * (len(perf) - 20) + [RED] * 10
    fig = go.Figure(go.Bar(x=perf.index.astype(str), y=perf["total_sales"],
                           marker_color=colors))
    fig.update_layout(title="Total sales over 143 weeks (green = top 10, red = bottom 10)",
                      height=420, xaxis_title="Store", yaxis_title="Total sales")
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 performers**")
        st.dataframe(perf[["total_sales", "avg_weekly", "share_pct", "cv_pct"]].head(5),
                     use_container_width=True)
    with c2:
        st.markdown("**Bottom 5 performers**")
        st.dataframe(perf[["total_sales", "avg_weekly", "share_pct", "cv_pct"]].tail(5),
                     use_container_width=True)

    insight(
        f"<b>Stores 20 and 4 are effectively tied</b> — separated by less than a "
        f"single week's sales across three years. The top 10 stores generate "
        f"<b>{gap['top10_share_pct']:.1f}%</b> of chain revenue; the bottom 10 "
        f"generate just <b>{gap['bottom10_share_pct']:.1f}%</b>."
    )

    st.subheader("How significant is the best-vs-worst gap?")
    c1, c2 = st.columns([2, 1])
    with c1:
        comp = df[df["Store"].isin([gap["best_store"], gap["worst_store"]])]
        fig = px.histogram(comp, x="Weekly_Sales", color="Store", nbins=60,
                           opacity=0.75, marginal="box",
                           color_discrete_map={gap["best_store"]: GREEN,
                                               gap["worst_store"]: RED})
        fig.update_layout(title="The distributions do not overlap at all", height=400)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown(
            f"""
| Test | Result |
|---|---|
| Welch t-test | t = {gap['t_stat']:.1f} |
| p-value | {gap['t_p_value']:.1e} |
| Mann-Whitney p | {gap['mannwhitney_p']:.1e} |
| Cohen's d | **{gap['cohens_d']:.2f}** |
| Ranges overlap? | **{'No' if gap['no_overlap'] else 'Yes'}** |
| ANOVA F (45 stores) | {gap['anova_f']:,.0f} |
            """
        )

    insight(
        "<b>But here is what I would lead with in a meeting.</b> An 8x gap almost "
        "certainly does <i>not</i> mean Store 33 is badly run. Gaps this clean and "
        "this stable are structural — store format, square footage, catchment "
        "population. Store 33 is also one of the <b>least volatile</b> stores in the "
        "chain (CV 9.3%). That pattern says <i>small store, consistent</i>, not "
        "<i>large store, failing</i>. Benchmark it against Stores 44, 5, 36 and 38, "
        "where it sits mid-pack."
    )

    st.subheader("Explore an individual store")
    store = st.selectbox("Store", sorted(df["Store"].unique()))
    sub = df[df["Store"] == store]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rank", f"#{int(perf.loc[store, 'rank'])} of 45")
    c2.metric("Avg weekly", f"${sub['Weekly_Sales'].mean()/1e3:,.0f}K")
    c3.metric("Volatility (CV)", f"{perf.loc[store, 'cv_pct']:.1f}%")
    c4.metric("Share of chain", f"{perf.loc[store, 'share_pct']:.2f}%")

    fig = px.line(sub, x="Date", y="Weekly_Sales")
    fig.update_traces(line=dict(color=BLUE, width=1.5))
    fig.update_layout(title=f"Store {store} weekly sales", height=330)
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
elif page == "Forecast":
    st.title("12-Week Sales Forecast")
    st.caption("Question 2: forecast each store for the next 12 weeks")

    val = get_validation(df)
    forecast = get_forecast(df, val["store_mape"])
    summary = M.store_summary(forecast, df)

    total_fc = summary["forecast_total"].sum()
    total_recent = summary["recent_actual_total"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("12-week chain forecast", f"${total_fc/1e6:,.0f}M")
    c2.metric("vs previous 12 weeks", f"{100*(total_fc/total_recent-1):+.1f}%")
    c3.metric("Validated accuracy", f"{val['metrics']['MAPE']:.2f}% MAPE")
    c4.metric("Forecast window",
              f"{forecast['Date'].min().date()} → {forecast['Date'].max().date()}")

    chain_hist = df.groupby("Date")["Weekly_Sales"].sum()
    chain_fc = forecast.groupby("Date")[["Forecast", "Lower_95", "Upper_95"]].sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chain_hist.index, y=chain_hist.values, mode="lines",
                             line=dict(color=BLUE, width=1.6), name="Actual"))
    fig.add_trace(go.Scatter(x=chain_fc.index, y=chain_fc["Upper_95"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=chain_fc.index, y=chain_fc["Lower_95"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(192,80,77,.2)", name="95% interval"))
    fig.add_trace(go.Scatter(x=chain_fc.index, y=chain_fc["Forecast"],
                             mode="lines+markers",
                             line=dict(color=RED, width=2.6), name="Forecast"))
    fig.add_vline(x=df["Date"].max(), line_dash="dot", line_color=GREY)
    fig.update_layout(title="Chain-wide: history and 12-week forecast", height=440,
                      hovermode="x unified", yaxis_title="Total weekly sales")
    st.plotly_chart(fig, width="stretch")

    insight(
        "<b>The model learned Christmas on its own.</b> It ramps through November, "
        "spikes in late December and drops away in January — the correct shape, with "
        "nobody hand-coding a holiday rule. That is the seasonal signal in "
        "<code>lag_52</code> doing its job."
    )

    st.subheader("Per-store forecast")
    store = st.selectbox("Store", sorted(forecast["Store"].unique()), key="fc_store")
    hist_s = df[df["Store"] == store]
    fc_s = forecast[forecast["Store"] == store]

    c1, c2, c3 = st.columns(3)
    c1.metric("12-week forecast", f"${summary.loc[store, 'forecast_total']/1e6:.2f}M",
              f"{summary.loc[store, 'change_pct']:+.1f}% vs last 12 weeks")
    c2.metric("Peak week", str(summary.loc[store, "peak_week"]),
              f"${summary.loc[store, 'peak_week_sales']/1e3:,.0f}K")
    c3.metric("Expected error", f"±{val['store_mape'].get(store, np.nan):.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_s["Date"], y=hist_s["Weekly_Sales"], mode="lines",
                             line=dict(color=BLUE, width=1.4), name="Actual"))
    fig.add_trace(go.Scatter(x=fc_s["Date"], y=fc_s["Upper_95"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc_s["Date"], y=fc_s["Lower_95"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(192,80,77,.2)", name="95% interval"))
    fig.add_trace(go.Scatter(x=fc_s["Date"], y=fc_s["Forecast"], mode="lines+markers",
                             line=dict(color=RED, width=2.4), name="Forecast"))
    fig.add_vline(x=df["Date"].max(), line_dash="dot", line_color=GREY)
    fig.update_layout(title=f"Store {store}", height=400, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Inventory planning table")
    st.caption(
        "Read `change_pct` as the instruction: a store at +20% needs a real stock "
        "build, a store near flat does not."
    )
    st.dataframe(summary, use_container_width=True, height=340)

    st.download_button(
        "⬇ Download full forecast (CSV)",
        forecast.to_csv(index=False).encode(),
        "walmart_forecast_next_12_weeks.csv",
        "text/csv",
    )


# --------------------------------------------------------------------------- #
# Model details
# --------------------------------------------------------------------------- #
else:
    st.title("Model Details")
    st.caption("How the forecast is built, validated, and where it is weak")

    val = get_validation(df)
    comparison = get_comparison(df)

    st.subheader("Model comparison")
    st.caption(
        "All four scored on the same time-ordered holdout: the final 12 weeks, "
        "which include Thanksgiving and Christmas."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(comparison, use_container_width=True)
    with c2:
        fig = px.bar(comparison.reset_index(), x="MAPE %", y="index", orientation="h",
                     color_discrete_sequence=[BLUE], text="MAPE %")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=280, yaxis_title="", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    insight(
        "<b>Seasonal naive at ~5.5% is the number to pay attention to.</b> Simply "
        "repeating last year gets you within 5.5% — which independently confirms "
        "that this business is driven by a stable annual cycle, not by economic "
        "conditions. The Random Forest wins because it can learn store-specific "
        "seasonal shapes <i>and</i> borrow strength across all 45 stores at once."
    )

    st.subheader("The rule that matters: no lag-1 cheating")
    st.markdown(
        """
The tempting approach is to feed the model last week's sales. It scores better —
**3.67% MAPE** in testing — but it is a lie. To predict week 12 you would need
week 11's actuals, which do not exist yet. A lag-1 model is a *one-step-ahead*
model wearing a *twelve-step-ahead* costume.

Every feature used here is one we genuinely have on hand at forecast time:

| Feature | Available 12 weeks out? | Why |
|---|---|---|
| `Store`, `WeekOfYear`, `Month` | Yes | It's a calendar |
| `Holiday_Flag` | Yes | Holiday dates are known years ahead |
| `lag_52`, `lag_53` | Yes | 52 > 12, so already observed |
| `roll_12`, `roll_52` | Yes | Shifted by the full horizon before rolling |
| `seasonal_index` | Yes | Derived from the above |
| `Temperature`, `CPI`, `Fuel_Price`, `Unemployment` | **Estimated** | Seasonal average / carried forward |
        """
    )

    st.subheader("Where the signal comes from")
    imp = val["importance"]
    fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                 color_discrete_sequence=[BLUE],
                 labels={"x": "importance", "y": ""})
    fig.update_layout(height=470, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    econ = float(imp[["Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]].sum())
    insight(
        f"<b>This chart is the whole story of the dataset in one picture.</b> "
        f"<code>lag_52</code> — same week last year — dominates. Every economic and "
        f"weather variable <i>combined</i> contributes just <b>{100*econ:.1f}%</b>. "
        "This is the empirical confirmation of the EDA: the business runs on the "
        "calendar."
    )

    st.subheader("Error diagnostics")
    test = val["test"]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(test, x="Weekly_Sales", y="prediction", opacity=0.6,
                         color_discrete_sequence=[BLUE])
        lims = [test["Weekly_Sales"].min(), test["Weekly_Sales"].max()]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                 line=dict(color=RED, dash="dash"), name="perfect"))
        fig.update_layout(title="Predicted vs actual (holdout)", height=380)
        st.plotly_chart(fig, width="stretch")
    with c2:
        se = val["store_mape"].sort_values()
        fig = px.bar(x=se.index.astype(str), y=se.values,
                     color=se.values, color_continuous_scale="RdYlGn_r",
                     labels={"x": "Store", "y": "MAPE %"})
        fig.update_layout(title="Forecast error by store", height=380,
                          coloraxis_showscale=False)
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE", f"{val['metrics']['MAPE']:.2f}%")
    c2.metric("MAE", f"${val['metrics']['MAE']/1e3:,.0f}K")
    c3.metric("RMSE", f"${val['metrics']['RMSE']/1e3:,.0f}K")
    c4.metric("R²", f"{val['metrics']['R2']:.4f}")

    st.subheader("Honest limitations")
    st.markdown(
        """
- **143 weeks is thin** for annual seasonality — only two complete Christmases to learn from.
- **We don't know store size, format, or location.** This is the biggest missing variable; it would likely explain most of the between-store gap and let us benchmark fairly.
- **Future temperature and economic values are estimated, not known.** Their near-zero importance makes this low-risk, but it is a real assumption.
- **No promotions, pricing or competitor data.** Markdowns and promotional calendars are major retail demand drivers and they are entirely absent here.
- **The model cannot anticipate structural breaks** — a new competitor, a renovation, or a closure would invalidate its `lag_52` logic for that store.
        """
    )
