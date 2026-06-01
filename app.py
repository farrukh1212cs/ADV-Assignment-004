"""
GAP-VizAI – Grammar-Aware Perception-driven Visualization AI System
Main landing page: system overview, architecture, and domain summary.
COMP-834 Advanced Data Visualization | PAK-AUSTRIA FACHHOCHSCHULE
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from modules.data_generator import (
    generate_healthcare_data, generate_finance_data, generate_ecommerce_data,
)
from modules.perceptual import OKABE_ITO, accessibility_score

st.set_page_config(
    page_title="GAP-VizAI | COMP-834",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title  { font-size:2.4rem; font-weight:700; color:#0072B2; text-align:center; }
.sub-title   { font-size:1.05rem; color:#555; text-align:center; margin-bottom:1.5rem; }
.stage-card  { border-radius:10px; padding:14px 10px; text-align:center; color:white; min-height:110px; }
.pillar-card { background:#F7F9FC; border-radius:8px; padding:18px; border-left:4px solid #0072B2; }
.metric-val  { font-size:2rem; font-weight:700; color:#0072B2; }
.metric-lbl  { font-size:0.85rem; color:#666; }
.insight-box { background:#E8F4FD; border-radius:6px; padding:10px 14px;
               border-left:3px solid #0072B2; margin:6px 0; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">GAP-VizAI Framework</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Grammar-Aware Perception-driven Visualization AI System<br>'
    '<b>COMP-834 Advanced Data Visualization</b> | PAK-AUSTRIA FACHHOCHSCHULE | Spring 2026</div>',
    unsafe_allow_html=True,
)
st.divider()


# ── Four Pillars ───────────────────────────────────────────────────────────────
st.subheader("System Pillars")
c1, c2, c3, c4 = st.columns(4)

pillars = [
    ("🎨", "Perceptual Design",
     "Okabe-Ito colorblind-safe palette · CIELAB color correction · "
     "Typography hierarchy · Grid-aligned layouts · WCAG AA compliance"),
    ("📐", "Grammar of Graphics",
     "Declarative VizSpec layers: Data → Aesthetics → Geom → Stat → Scale → Facet · "
     "Structured compilation pipeline · Reproducible specifications"),
    ("📊", "Statistical Visualization",
     "ECDF · KDE · Violin · Ridge plots · "
     "Confidence bands · Uncertainty representation across 3 research domains"),
    ("🤖", "AI Insight Discovery",
     "IQR & z-score anomaly detection · Trend analysis · "
     "Pearson correlation · NL insight generation · Visualization critique scoring"),
]

for col, (icon, title, desc) in zip([c1, c2, c3, c4], pillars):
    with col:
        st.markdown(f"""
        <div class="pillar-card">
            <h4>{icon} {title}</h4>
            <p style="font-size:0.85rem;color:#444;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ── Architecture Pipeline ──────────────────────────────────────────────────────
st.subheader("System Architecture — 7-Stage Pipeline")

stages = [
    ("1", "Data<br>Ingestion",      "3 Domains<br>Healthcare · Finance<br>E-Commerce",  "#0072B2"),
    ("2", "AI Prompt<br>Layer",     "NL Query →<br>Structured Schema<br>Chain-of-Thought", "#009E73"),
    ("3", "Grammar<br>Engine",      "Data → Geom<br>Stat → Scale<br>VizSpec Compiler",  "#E69F00"),
    ("4", "Perceptual<br>Optimizer","CIELAB Colors<br>Typography<br>Grid Layout",        "#56B4E9"),
    ("5", "Distribution<br>Engine", "ECDF · KDE<br>Violin · Ridge<br>Uncertainty",       "#D55E00"),
    ("6", "AI Insight<br>Layer",    "Anomaly Detect.<br>Trend · Corr.<br>NL Summary",    "#CC79A7"),
    ("7", "Critique<br>Module",     "Quality Score<br>Accessibility<br>AI Compliance",   "#595959"),
]

cols = st.columns(7)
for col, (num, name, desc, clr) in zip(cols, stages):
    with col:
        st.markdown(f"""
        <div class="stage-card" style="background:{clr};">
            <div style="font-size:1.6rem;font-weight:700;">{num}</div>
            <div style="font-size:0.82rem;font-weight:600;margin:4px 0;">{name}</div>
            <div style="font-size:0.70rem;opacity:0.88;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ── Live Domain Metrics ────────────────────────────────────────────────────────
st.subheader("Research Domains — Live Data Summary")

@st.cache_data
def load_summaries():
    hc = generate_healthcare_data()
    ret, pr, tickers = generate_finance_data()
    ec = generate_ecommerce_data()
    return hc, ret, pr, tickers, ec

hc, ret, pr, tickers, ec = load_summaries()

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown("#### 🏥 Healthcare Domain")
    m1, m2 = st.columns(2)
    m1.metric("Patients",    f"{len(hc):,}")
    m2.metric("Risk Groups", "4")
    m1.metric("Mean HR",     f"{hc['heart_rate'].mean():.1f} bpm")
    m2.metric("Mean SpO₂",   f"{hc['spo2'].mean():.1f}%")
    risk_counts = hc["risk_level"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.55,
        marker_colors=["#009E73","#E69F00","#D55E00","#CC0000"],
    ))
    fig_pie.update_layout(
        height=180, margin=dict(l=0,r=0,t=20,b=0),
        showlegend=True, legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with d2:
    st.markdown("#### 💹 Finance Domain")
    m1, m2 = st.columns(2)
    m1.metric("Trading Days", f"{len(ret):,}")
    m2.metric("Tickers",      f"{len(tickers)}")
    ann_ret = ret.mean() * 252
    m1.metric("Best Annual",  f"{ann_ret.max()*100:.1f}%")
    m2.metric("Worst Annual", f"{ann_ret.min()*100:.1f}%")
    fig_line = go.Figure()
    for i, t in enumerate(tickers):
        fig_line.add_trace(go.Scatter(
            x=pr.index, y=pr[t],
            name=t, mode="lines",
            line=dict(color=OKABE_ITO[i], width=1.5),
        ))
    fig_line.update_layout(
        height=180, margin=dict(l=0,r=0,t=10,b=0),
        showlegend=True, legend=dict(font=dict(size=9)),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#EBEBEB"),
    )
    st.plotly_chart(fig_line, use_container_width=True)

with d3:
    st.markdown("#### 🛒 E-Commerce Domain")
    m1, m2 = st.columns(2)
    m1.metric("Categories",   "7")
    m2.metric("Regions",      "4")
    total_sales = ec["sales"].sum()
    m1.metric("Total Sales",  f"${total_sales/1e6:.1f}M")
    m2.metric("Avg Rating",   f"{ec['avg_rating'].mean():.2f} ★")
    cat_sales = ec.groupby("category")["sales"].sum().sort_values(ascending=True)
    fig_bar = go.Figure(go.Bar(
        x=cat_sales.values / 1e6,
        y=cat_sales.index,
        orientation="h",
        marker_color=OKABE_ITO[:len(cat_sales)],
    ))
    fig_bar.update_layout(
        height=180, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        xaxis_title="Sales ($M)", yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()


# ── Colour Accessibility Panel ─────────────────────────────────────────────────
st.subheader("Perceptual Design — Okabe-Ito Palette Accessibility")

acc = accessibility_score(OKABE_ITO)
a1, a2, a3, a4 = st.columns(4)
a1.metric("WCAG AA Compliance",  f"{acc['aa_compliance']*100:.0f}%")
a2.metric("WCAG AAA Compliance", f"{acc['aaa_compliance']*100:.0f}%")
a3.metric("Mean Contrast Ratio", f"{acc['mean_contrast']:.2f}:1")
a4.metric("Palette Size",        f"{len(OKABE_ITO)} colors")

color_cols = st.columns(len(OKABE_ITO))
names = ["Blue","Orange","Bl.Green","Sky Blue","Vermillion","Rd.Purple","Yellow","Black"]
for col, clr, name, ratio in zip(color_cols, OKABE_ITO, names, acc["ratios"]):
    with col:
        tag = "AA ✅" if ratio >= 4.5 else "Fail ⚠️"
        st.markdown(
            f'<div style="background:{clr};height:40px;border-radius:4px;margin-bottom:4px;"></div>'
            f'<div style="font-size:0.70rem;text-align:center;">{name}<br>{ratio:.1f}:1 {tag}</div>',
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem;'>"
    "GAP-VizAI | COMP-834 Assignment 4 | PAK-AUSTRIA FACHHOCHSCHULE | Spring 2026"
    "</div>",
    unsafe_allow_html=True,
)
