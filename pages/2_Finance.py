"""
Finance Domain Dashboard
Visualises synthetic stock-market time series data.
Charts: Price series · ECDF returns · Ridge distributions · Fan forecast · Correlation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

from modules.data_generator import generate_finance_data
from modules.distribution_viz import ecdf_plot, ridge_plot, kde_histogram_comparison
from modules.uncertainty_viz import time_series_with_ci, fan_chart
from modules.composition_viz import correlation_heatmap
from modules.ai_insights import InsightEngine, generate_domain_summary
from modules.grammar_engine import build_spec
from modules.perceptual import OKABE_ITO, apply_template
from modules.critique import score_chart, critique_text

st.set_page_config(page_title="Finance | GAP-VizAI", layout="wide", page_icon="💹")

st.title("💹 Finance Domain — Stock Market Analytics")
st.caption("Synthetic correlated stock dataset · 5 tickers · 365 trading days · Cholesky-correlated returns")
st.divider()


@st.cache_data
def load_data():
    return generate_finance_data()

returns_df, prices_df, tickers = load_data()

with st.sidebar:
    st.header("Filters")
    sel_tickers = st.multiselect("Tickers", tickers, default=tickers)
    window = st.slider("Rolling Window (days)", 5, 30, 10)
    horizon = st.slider("Forecast Horizon (days)", 10, 60, 30)
    ci_level = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
    st.divider()
    st.caption("GAP-VizAI · Finance Module")

ret = returns_df[sel_tickers]
prc = prices_df[sel_tickers]


# ── Row 1: Price Time Series with CI Bands ────────────────────────────────────
st.subheader("1 · Stock Price Trajectories with Confidence Bands")

fig_price = go.Figure()
for i, tick in enumerate(sel_tickers):
    prices = prc[tick]
    color = OKABE_ITO[i % len(OKABE_ITO)]
    # Rolling mean & CI
    rm = prices.rolling(window).mean()
    rs = prices.rolling(window).std()
    from scipy.stats import t
    t_crit = t.ppf(0.5 + ci_level / 2, df=window - 1)
    margin = t_crit * rs / np.sqrt(window)
    valid = ~rm.isna()
    d = prc.index[valid]
    # CI band
    fig_price.add_trace(go.Scatter(
        x=np.concatenate([d, d[::-1]]),
        y=np.concatenate([(rm + margin)[valid], (rm - margin)[valid][::-1]]),
        fill="toself", fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [40])}",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig_price.add_trace(go.Scatter(
        x=prc.index, y=prices, mode="lines",
        name=tick, line=dict(color=color, width=2),
    ))

fig_price.update_layout(
    title=f"Stock Prices with {int(ci_level*100)}% CI (rolling {window}-day window)",
    xaxis_title="Date", yaxis_title="Price (USD)",
    plot_bgcolor="#FAFAFA", paper_bgcolor="white",
    legend=dict(orientation="h", y=1.02, x=0),
)
st.plotly_chart(fig_price, use_container_width=True)


# ── Row 2: ECDF of Daily Returns ──────────────────────────────────────────────
st.subheader("2 · ECDF — Daily Returns Comparison")
c1, c2 = st.columns(2)

with c1:
    ret_long = ret.reset_index().melt(id_vars="date", var_name="ticker", value_name="return")
    fig_ecdf = ecdf_plot(ret_long, x_col="return", color_col="ticker",
                         title="ECDF: Daily Returns per Ticker", palette=OKABE_ITO)
    # Add theoretical normal reference
    x_ref = np.linspace(ret.values.min(), ret.values.max(), 300)
    fig_ecdf.add_trace(go.Scatter(
        x=x_ref, y=norm.cdf(x_ref, loc=ret.values.mean(), scale=ret.values.std()),
        mode="lines", name="Normal ref.",
        line=dict(color="gray", dash="dash", width=1.5),
    ))
    st.plotly_chart(fig_ecdf, use_container_width=True)

with c2:
    fig_kde = kde_histogram_comparison(
        ret_long, x_col="return", color_col="ticker",
        title="KDE vs Histogram: Return Distributions", palette=OKABE_ITO,
    )
    st.plotly_chart(fig_kde, use_container_width=True)

with st.expander("📐 Grammar Spec — ECDF"):
    spec = build_spec(ret_long, x="return", geom_type="ecdf", color="ticker",
                      domain="finance", stat_type="ecdf",
                      title="ECDF of Daily Returns",
                      notes="Normal distribution reference overlay added for tail-risk assessment.")
    st.json(spec.to_dict())
    st.metric("Grammar Compliance", f"{spec.grammar_compliance_score()*100:.0f}%")


# ── Row 3: Ridge Plot of Return Distributions ─────────────────────────────────
st.subheader("3 · Ridge Plot — Return Distributions by Ticker")

fig_ridge = ridge_plot(ret_long, x_col="return", group_col="ticker",
                       title="Ridge Plot: Daily Return Distributions by Ticker",
                       palette=OKABE_ITO)
st.plotly_chart(fig_ridge, use_container_width=True)


# ── Row 4: Forecast Fan Chart ─────────────────────────────────────────────────
st.subheader("4 · Price Forecast with Uncertainty Fan Chart")

tick_for_fan = st.selectbox("Select Ticker for Forecast", sel_tickers)
hist_prices = prc[tick_for_fan].values
hist_x = prc.index

tick_ret = ret[tick_for_fan]
mu    = tick_ret.mean()
sigma = tick_ret.std()

# Simple random-walk forecast
last_price = hist_prices[-1]
forecast_mean = last_price * np.exp(mu * np.arange(1, horizon + 1))
forecast_std  = last_price * sigma * np.sqrt(np.arange(1, horizon + 1))
forecast_x    = pd.date_range(hist_x[-1], periods=horizon + 1, freq="D")[1:]

fig_fan = fan_chart(
    historical_x=hist_x[-90:],
    historical_y=hist_prices[-90:],
    forecast_x=forecast_x,
    forecast_mean=forecast_mean,
    forecast_std=forecast_std,
    title=f"{tick_for_fan} — {horizon}-Day Price Forecast with Uncertainty Fan",
    color=OKABE_ITO[sel_tickers.index(tick_for_fan)],
    n_bands=3,
)
st.plotly_chart(fig_fan, use_container_width=True)

st.caption(
    "Fan chart uses a log-normal random walk model. "
    "Bands represent ±1σ, ±2σ, ±3σ forecast intervals. "
    "**Not financial advice.**"
)


# ── Row 5: Correlation Heatmap ────────────────────────────────────────────────
st.subheader("5 · Return Correlation Matrix")

fig_heatmap = correlation_heatmap(ret, cols=sel_tickers,
                                  title="Pearson Correlation — Daily Returns")
st.plotly_chart(fig_heatmap, use_container_width=True)


# ── Row 6: Risk Metrics ───────────────────────────────────────────────────────
st.subheader("6 · Risk Metrics Summary")

metrics_df = pd.DataFrame({
    "Ticker":        sel_tickers,
    "Ann. Return %": [ret[t].mean() * 252 * 100 for t in sel_tickers],
    "Ann. Volatility %": [ret[t].std() * np.sqrt(252) * 100 for t in sel_tickers],
    "Sharpe Ratio":  [ret[t].mean() / ret[t].std() * np.sqrt(252) for t in sel_tickers],
    "VaR (95%) %":   [abs(np.percentile(ret[t], 5)) * 100 for t in sel_tickers],
    "Max Drawdown %":[abs(((prc[t] / prc[t].cummax()) - 1).min()) * 100 for t in sel_tickers],
}).set_index("Ticker").round(3)

st.dataframe(metrics_df.style.background_gradient(cmap="Blues"), use_container_width=True)


# ── Row 7: AI Insights ────────────────────────────────────────────────────────
st.subheader("7 · AI Insight Discovery")

engine = InsightEngine(ret.reset_index(drop=True), numeric_cols=sel_tickers, domain="finance")
cards = engine.insight_cards()

st.markdown(generate_domain_summary("finance", {
    "n_days": len(ret),
    "mean_ret": float(ret.mean().mean()),
    "ann_ret": float(ret.mean().mean() * 252),
    "var_95": float(abs(np.percentile(ret.values, 5))),
}))

c1, c2, c3 = st.columns(3)
for i, card in enumerate(cards[:6]):
    with [c1, c2, c3][i % 3]:
        border = "#D55E00" if card["severity"] == "high" else "#0072B2"
        st.markdown(
            f'<div style="border-left:4px solid {border};background:#F7F9FC;'
            f'border-radius:6px;padding:10px 12px;margin:6px 0;">'
            f'<b>{card["icon"]} {card["title"]}</b><br>'
            f'<span style="font-size:0.85rem;">{card["body"]}</span></div>',
            unsafe_allow_html=True,
        )
