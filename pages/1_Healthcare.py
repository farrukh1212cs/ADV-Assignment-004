"""
Healthcare Domain Dashboard
Visualises ICU patient data (synthetic MIMIC-like dataset).
Charts: Violin · ECDF · KDE vs Histogram · Ridge · Uncertainty bands · AI Insights
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from modules.data_generator import generate_healthcare_data, generate_temporal_health_data
from modules.distribution_viz import (
    ecdf_plot, violin_plot, kde_histogram_comparison, ridge_plot, box_strip_plot,
)
from modules.uncertainty_viz import time_series_with_ci
from modules.ai_insights import InsightEngine, generate_domain_summary
from modules.grammar_engine import build_spec
from modules.perceptual import OKABE_ITO, RISK_COLORS
from modules.critique import score_chart, critique_text

st.set_page_config(page_title="Healthcare | GAP-VizAI", layout="wide", page_icon="🏥")

st.title("🏥 Healthcare Domain — ICU Patient Analytics")
st.caption("Synthetic MIMIC-IV style dataset · 500 patients · 5 vitals · 4 risk levels")
st.divider()

# ── Data & Sidebar ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return generate_healthcare_data(), generate_temporal_health_data()

df, df_temp = load_data()

with st.sidebar:
    st.header("Filters")
    risk_sel = st.multiselect(
        "Risk Level", options=["Low", "Medium", "High", "Critical"],
        default=["Low", "Medium", "High", "Critical"],
    )
    age_range = st.slider("Age Range", 18, 90, (18, 90))
    vital_sel = st.selectbox("Primary Vital", ["heart_rate", "systolic_bp", "spo2", "temperature", "los_days"])
    st.divider()
    st.caption("GAP-VizAI · Healthcare Module")

filtered = df[df["risk_level"].isin(risk_sel) & df["age"].between(*age_range)]


# ── Row 1: Violin + ECDF ──────────────────────────────────────────────────────
st.subheader("1 · Distribution by Risk Level")
c1, c2 = st.columns(2)

with c1:
    fig = violin_plot(
        filtered, x_col="risk_level", y_col=vital_sel,
        title=f"Violin: {vital_sel.replace('_',' ').title()} by Risk Level",
        palette=list(RISK_COLORS.values()), color_map=RISK_COLORS,
    )
    st.plotly_chart(fig, use_container_width=True)

    spec = build_spec(filtered, x="risk_level", geom_type="violin", y=vital_sel,
                      color="risk_level", domain="healthcare",
                      title="Violin: Vital Signs by Risk Level",
                      notes="Box inside violin shows IQR; point layer shows outliers only.")
    with st.expander("📐 Grammar Spec (VizSpec JSON)", expanded=False):
        st.json(spec.to_dict())
        st.metric("Grammar Compliance", f"{spec.grammar_compliance_score()*100:.0f}%")

with c2:
    fig2 = ecdf_plot(
        filtered, x_col=vital_sel, color_col="risk_level",
        title=f"ECDF: {vital_sel.replace('_',' ').title()}",
        palette=list(RISK_COLORS.values()),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Row 2: KDE vs Histogram ───────────────────────────────────────────────────
st.subheader("2 · KDE vs Histogram Comparison")

fig3 = kde_histogram_comparison(
    filtered, x_col=vital_sel, color_col="risk_level",
    title=f"KDE vs Histogram: {vital_sel.replace('_',' ').title()} by Risk Level",
    palette=list(RISK_COLORS.values()),
)
st.plotly_chart(fig3, use_container_width=True)

with st.expander("ℹ️ Why compare KDE and Histogram?"):
    st.markdown("""
    | Aspect | Histogram | KDE |
    |--------|-----------|-----|
    | Binning artefacts | Yes (bin width sensitive) | No (bandwidth parameter) |
    | Continuous estimate | No | Yes |
    | Overplotting | No | No |
    | Best for | Quick counts | Shape comparison |

    GAP-VizAI applies **Silverman's rule-of-thumb** bandwidth for all KDE estimates,
    ensuring perceptually optimal smoothing without over-fitting.
    """)


# ── Row 3: Ridge Plot ─────────────────────────────────────────────────────────
st.subheader("3 · Ridge Plot — Multi-Distribution View")

vitals_for_ridge = ["heart_rate", "systolic_bp", "spo2", "temperature"]
vit = st.selectbox("Vital for Ridge", vitals_for_ridge, index=0, key="ridge_sel")

fig4 = ridge_plot(
    filtered, x_col=vit, group_col="risk_level",
    title=f"Ridge Plot: {vit.replace('_',' ').title()} Distribution by Risk Level",
    color_map=RISK_COLORS,
)
st.plotly_chart(fig4, use_container_width=True)


# ── Row 4: Temporal Uncertainty Bands ────────────────────────────────────────
st.subheader("4 · Temporal Heart Rate with 95% Confidence Bands")

sample_ids = df_temp["patient_id"].unique()[:6]
df_sample = df_temp[df_temp["patient_id"].isin(sample_ids)].copy()

fig5 = time_series_with_ci(
    df_sample, date_col="hour", value_col="heart_rate", group_col="risk_level",
    window=6, ci=0.95,
    title="Heart Rate Trajectory with 95% CI (per Risk Group)",
    palette=list(RISK_COLORS.values()),
)
st.plotly_chart(fig5, use_container_width=True)


# ── Row 5: Summary Stats Table ────────────────────────────────────────────────
st.subheader("5 · Vital Signs Summary by Risk Level")

summary = filtered.groupby("risk_level")[
    ["heart_rate", "systolic_bp", "spo2", "temperature", "los_days"]
].agg(["mean", "std"]).round(2)
summary.columns = [" ".join(c).title() for c in summary.columns]
st.dataframe(summary, use_container_width=True)


# ── Row 6: AI Insights ────────────────────────────────────────────────────────
st.subheader("6 · AI Insight Discovery")

numeric_cols = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature", "los_days"]
engine = InsightEngine(filtered, numeric_cols=numeric_cols, domain="healthcare")
cards = engine.insight_cards()

st.markdown(generate_domain_summary("healthcare", {
    "n_patients": len(filtered),
    "mean_hr": filtered["heart_rate"].mean(),
    "std_hr":  filtered["heart_rate"].std(),
    "hr_increase": (
        (filtered[filtered["risk_level"]=="Critical"]["heart_rate"].mean() -
         filtered[filtered["risk_level"]=="Low"]["heart_rate"].mean()) /
        filtered[filtered["risk_level"]=="Low"]["heart_rate"].mean() * 100
    ) if "Low" in risk_sel and "Critical" in risk_sel else 0,
    "mean_los": filtered["los_days"].mean(),
}))

st.markdown("")
cols = st.columns(min(3, max(1, len(cards))))
for i, card in enumerate(cards[:6]):
    with cols[i % len(cols)]:
        border = "#D55E00" if card["severity"] == "high" else "#0072B2"
        st.markdown(
            f'<div style="border-left:4px solid {border};background:#F7F9FC;'
            f'border-radius:6px;padding:10px 12px;margin:6px 0;">'
            f'<b>{card["icon"]} {card["title"]}</b><br>'
            f'<span style="font-size:0.85rem;">{card["body"]}</span></div>',
            unsafe_allow_html=True,
        )


# ── Row 7: Visualization Critique ─────────────────────────────────────────────
st.subheader("7 · Visualization Quality Critique")

report = score_chart("violin", system="gap_vizai")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Perceptual", f"{report.perceptual_score*100:.0f}/100")
c2.metric("Accessibility", f"{report.accessibility_score*100:.0f}/100")
c3.metric("Encoding", f"{report.encoding_correctness*100:.0f}/100")
c4.metric("Grammar", f"{report.grammar_compliance*100:.0f}/100")
c5.metric("Overall", f"{report.overall_score*100:.0f}/100", delta="vs manual baseline")

for line in critique_text(report):
    st.markdown(line)
