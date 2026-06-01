"""
AI Insight Discovery — Cross-Domain Analysis
Runs the full InsightEngine across all three domains and presents
anomalies, trends, correlations, and NL narrative summaries.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from modules.data_generator import generate_healthcare_data, generate_finance_data, generate_ecommerce_data
from modules.ai_insights import (
    InsightEngine, detect_anomalies_iqr, detect_trend,
    compute_correlations, generate_domain_summary,
)
from modules.perceptual import OKABE_ITO, apply_template
from modules.composition_viz import correlation_heatmap

st.set_page_config(page_title="AI Insights | GAP-VizAI", layout="wide", page_icon="🤖")

st.title("🤖 AI Insight Discovery — Cross-Domain Analysis")
st.caption("Automated anomaly detection · trend analysis · correlation discovery · NL narrative generation")
st.divider()


@st.cache_data
def load_all():
    hc = generate_healthcare_data()
    ret, prc, tickers = generate_finance_data()
    ec = generate_ecommerce_data()
    return hc, ret, prc, tickers, ec

hc, ret, prc, tickers, ec = load_all()


# ── Insight Summary Metrics ───────────────────────────────────────────────────
hc_engine = InsightEngine(hc,
    ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature", "los_days"],
    domain="healthcare")
fin_engine = InsightEngine(ret.reset_index(drop=True), list(ret.columns), domain="finance")
ec_engine  = InsightEngine(ec, ["sales", "profit", "avg_rating", "return_rate"], domain="ecommerce")

hc_cards  = hc_engine.insight_cards()
fin_cards = fin_engine.insight_cards()
ec_cards  = ec_engine.insight_cards()
total_insights = len(hc_cards) + len(fin_cards) + len(ec_cards)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Insights Found", total_insights)
m2.metric("Anomalies Detected",   sum(1 for c in hc_cards + fin_cards + ec_cards if c["type"] == "anomaly"))
m3.metric("Trends Identified",    sum(1 for c in hc_cards + fin_cards + ec_cards if c["type"] == "trend"))
m4.metric("Correlations Found",   sum(1 for c in hc_cards + fin_cards + ec_cards if c["type"] == "correlation"))
st.divider()


# ── Domain Narrative Summaries ────────────────────────────────────────────────
st.subheader("1 · AI-Generated Domain Narratives")

tab1, tab2, tab3 = st.tabs(["🏥 Healthcare", "💹 Finance", "🛒 E-Commerce"])

with tab1:
    st.markdown(generate_domain_summary("healthcare", {
        "n_patients": len(hc),
        "mean_hr": hc["heart_rate"].mean(),
        "std_hr": hc["heart_rate"].std(),
        "hr_increase": (hc[hc["risk_level"]=="Critical"]["heart_rate"].mean() -
                        hc[hc["risk_level"]=="Low"]["heart_rate"].mean()) /
                        hc[hc["risk_level"]=="Low"]["heart_rate"].mean() * 100,
        "mean_los": hc["los_days"].mean(),
    }))

with tab2:
    st.markdown(generate_domain_summary("finance", {
        "n_days": len(ret),
        "mean_ret": float(ret.mean().mean()),
        "ann_ret": float(ret.mean().mean() * 252),
        "var_95": float(abs(np.percentile(ret.values, 5))),
    }))

with tab3:
    elec = ec[ec["category"] == "Electronics"]["sales"].sum()
    st.markdown(generate_domain_summary("ecommerce", {
        "n_categories": 7, "n_regions": 4,
        "total_sales": ec["sales"].sum(),
        "electronics_share": elec / ec["sales"].sum() * 100,
        "seasonal_peak": "Q4", "quality_pos": 62, "shipping_neg": 42,
    }))


# ── Anomaly Detection Results ─────────────────────────────────────────────────
st.subheader("2 · Anomaly Detection Results")

col_sel = st.selectbox("Analyse column (Healthcare)", ["heart_rate", "systolic_bp", "spo2", "los_days"])
anomalies = detect_anomalies_iqr(hc, col_sel)

c1, c2 = st.columns([2, 1])
with c1:
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=hc.index, y=hc[col_sel],
        mode="markers",
        marker=dict(color=OKABE_ITO[0], size=4, opacity=0.5),
        name="Normal",
    ))
    if anomalies:
        anom_idx = [a.index for a in anomalies[:50]]
        anom_val = [a.value for a in anomalies[:50]]
        fig_anom.add_trace(go.Scatter(
            x=anom_idx, y=anom_val,
            mode="markers",
            marker=dict(color="#D55E00", size=10, symbol="x", line=dict(width=2)),
            name="Anomaly",
        ))
    q1, q3 = hc[col_sel].quantile(0.25), hc[col_sel].quantile(0.75)
    iqr = q3 - q1
    fig_anom.add_hline(y=q3 + 1.5 * iqr, line=dict(color="gray", dash="dot"), annotation_text="IQR upper")
    fig_anom.add_hline(y=q1 - 1.5 * iqr, line=dict(color="gray", dash="dot"), annotation_text="IQR lower")
    fig_anom.update_layout(
        title=f"Anomaly Detection: {col_sel} (IQR method, threshold=1.5)",
        xaxis_title="Patient Index", yaxis_title=col_sel,
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
    )
    st.plotly_chart(fig_anom, use_container_width=True)

with c2:
    st.markdown(f"**{len(anomalies)} anomalies detected** in `{col_sel}`")
    st.markdown(f"IQR upper fence: `{q3 + 1.5*iqr:.2f}`")
    st.markdown(f"IQR lower fence: `{q1 - 1.5*iqr:.2f}`")
    st.markdown("---")
    for a in anomalies[:8]:
        severity = "🔴" if a.z_score > 3 else "🟡"
        st.markdown(
            f"{severity} **idx {a.index}**: {a.value:.2f} "
            f"({a.z_score:.1f}σ {a.direction})"
        )


# ── Trend Analysis ────────────────────────────────────────────────────────────
st.subheader("3 · Trend Analysis")

c1, c2 = st.columns(2)

with c1:
    hc_trends = []
    for col in ["heart_rate", "systolic_bp", "spo2", "temperature", "los_days"]:
        t = detect_trend(hc.reset_index(), "index", col)
        hc_trends.append({"Column": col, "Direction": t.direction,
                           "Slope": t.slope, "R²": t.r_squared, "Significance": t.significance})
    trend_df = pd.DataFrame(hc_trends)
    st.markdown("**Healthcare — Trend Summary**")
    st.dataframe(trend_df.style.applymap(
        lambda v: "color:#009E73" if v == "significant" else "color:#D55E00",
        subset=["Significance"]
    ), use_container_width=True)

with c2:
    fin_trends = []
    for col in list(ret.columns):
        t = detect_trend(ret.reset_index(drop=True), "index", col)
        fin_trends.append({"Ticker": col, "Direction": t.direction,
                            "Slope": t.slope, "R²": t.r_squared, "Significance": t.significance})
    ftdf = pd.DataFrame(fin_trends)
    st.markdown("**Finance — Trend Summary**")
    st.dataframe(ftdf, use_container_width=True)


# ── Correlation Analysis ─────────────────────────────────────────────────────
st.subheader("4 · Correlation Analysis")
c1, c2 = st.columns(2)

with c1:
    hc_corr_cols = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature", "los_days"]
    fig_hc_hm = correlation_heatmap(hc, hc_corr_cols, title="Healthcare — Vital Signs Correlation")
    st.plotly_chart(fig_hc_hm, use_container_width=True)

with c2:
    fig_fin_hm = correlation_heatmap(ret, list(ret.columns), title="Finance — Returns Correlation")
    st.plotly_chart(fig_fin_hm, use_container_width=True)

# Top correlation pairs
st.markdown("**Top Significant Correlations — Healthcare**")
hc_pairs = compute_correlations(hc, hc_corr_cols)
for c in hc_pairs[:5]:
    st.markdown(f"- {c.to_text()}")

st.markdown("**Top Significant Correlations — Finance Returns**")
fin_pairs = compute_correlations(ret, list(ret.columns))
for c in fin_pairs[:5]:
    st.markdown(f"- {c.to_text()}")


# ── All Insight Cards ─────────────────────────────────────────────────────────
st.subheader("5 · All Discovered Insights")

domain_cards = [("🏥 Healthcare", hc_cards), ("💹 Finance", fin_cards), ("🛒 E-Commerce", ec_cards)]

for domain_name, cards in domain_cards:
    if not cards:
        continue
    st.markdown(f"#### {domain_name}")
    cols = st.columns(3)
    for i, card in enumerate(cards[:6]):
        with cols[i % 3]:
            type_colors = {"anomaly": "#D55E00", "trend": "#0072B2", "correlation": "#009E73"}
            border = type_colors.get(card["type"], "#555")
            st.markdown(
                f'<div style="border-left:4px solid {border};background:#F7F9FC;'
                f'border-radius:6px;padding:10px 12px;margin:6px 0;min-height:80px;">'
                f'<b>{card["icon"]} {card["title"]}</b><br>'
                f'<span style="font-size:0.82rem;">{card["body"]}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")


# ── Insight Type Distribution ─────────────────────────────────────────────────
st.subheader("6 · Insight Distribution by Type and Domain")

all_cards = (
    [("Healthcare", c) for c in hc_cards] +
    [("Finance",    c) for c in fin_cards] +
    [("E-Commerce", c) for c in ec_cards]
)
if all_cards:
    insight_summary = pd.DataFrame([
        {"Domain": d, "Type": c["type"].capitalize()} for d, c in all_cards
    ])
    counts = insight_summary.groupby(["Domain", "Type"]).size().reset_index(name="Count")
    fig_ins = px.bar(counts, x="Domain", y="Count", color="Type",
                     barmode="group", title="Insights by Domain and Type",
                     color_discrete_sequence=OKABE_ITO)
    fig_ins.update_layout(plot_bgcolor="#FAFAFA", paper_bgcolor="white")
    st.plotly_chart(fig_ins, use_container_width=True)
