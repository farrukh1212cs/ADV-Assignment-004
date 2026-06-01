"""
E-Commerce Domain Dashboard
Visualises synthetic retail sales, composition, and sentiment data.
Charts: Treemap · Sankey · Stacked bar · Violin ratings · Sentiment breakdown
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from modules.data_generator import generate_ecommerce_data, generate_sentiment_data
from modules.composition_viz import (
    treemap_chart, build_ecommerce_sankey, stacked_bar_chart, correlation_heatmap,
)
from modules.distribution_viz import violin_plot, ecdf_plot, ridge_plot
from modules.uncertainty_viz import error_band_comparison
from modules.ai_insights import InsightEngine, generate_domain_summary
from modules.perceptual import OKABE_ITO, SENTIMENT_COLORS, apply_template
from modules.critique import score_chart, critique_text

st.set_page_config(page_title="E-Commerce | GAP-VizAI", layout="wide", page_icon="🛒")

st.title("🛒 E-Commerce Domain — Retail Analytics")
st.caption("Synthetic retail dataset · 7 categories · 4 regions · 12 months · 336 records")
st.divider()


@st.cache_data
def load_data():
    return generate_ecommerce_data(), generate_sentiment_data()

df, sentiment_df = load_data()

with st.sidebar:
    st.header("Filters")
    cat_sel = st.multiselect("Categories", df["category"].unique().tolist(),
                              default=df["category"].unique().tolist())
    reg_sel = st.multiselect("Regions", df["region"].unique().tolist(),
                              default=df["region"].unique().tolist())
    month_range = st.slider("Month Range", 1, 12, (1, 12))
    st.divider()
    st.caption("GAP-VizAI · E-Commerce Module")

filtered = df[
    df["category"].isin(cat_sel) &
    df["region"].isin(reg_sel) &
    df["month"].between(*month_range)
]


# ── Row 1: KPI Cards ──────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales",  f"${filtered['sales'].sum()/1e6:.2f}M")
k2.metric("Total Profit", f"${filtered['profit'].sum()/1e6:.2f}M")
k3.metric("Avg Rating",   f"{filtered['avg_rating'].mean():.2f} ★")
k4.metric("Avg Return %", f"{filtered['return_rate'].mean()*100:.1f}%")


# ── Row 2: Treemap ────────────────────────────────────────────────────────────
st.subheader("1 · Sales Treemap — Category × Region Composition")

tree_df = filtered.groupby(["category", "region"])["sales"].sum().reset_index()

fig_tree = treemap_chart(
    tree_df,
    path=["category", "region"],
    values="sales",
    title="Sales Composition: Category → Region (Treemap)",
    palette=OKABE_ITO,
)
st.plotly_chart(fig_tree, use_container_width=True)


# ── Row 3: Sankey Diagram ────────────────────────────────────────────────────
st.subheader("2 · Sales Flow — Region → Category (Sankey)")

fig_sankey = build_ecommerce_sankey(filtered)
st.plotly_chart(fig_sankey, use_container_width=True)

with st.expander("ℹ️ Reading the Sankey Diagram"):
    st.markdown("""
    - **Left nodes**: Geographic regions (North, South, East, West)
    - **Right nodes**: Product categories
    - **Link width**: Proportional to sales volume ($K)
    - **Link colour**: Derived from the source region

    Sankey diagrams follow the grammar-of-graphics **flow geom** with
    `source → target → value` aesthetics. They are most effective for
    showing proportional contributions in compositional data.
    """)


# ── Row 4: Stacked Bar — Monthly Sales ───────────────────────────────────────
st.subheader("3 · Monthly Sales Composition")
c1, c2 = st.columns(2)

monthly = filtered.groupby(["month", "category"])["sales"].sum().reset_index()

with c1:
    fig_stack = stacked_bar_chart(
        monthly, x_col="month", y_col="sales", color_col="category",
        title="Monthly Sales by Category (Absolute)", palette=OKABE_ITO,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

with c2:
    fig_norm = stacked_bar_chart(
        monthly, x_col="month", y_col="sales", color_col="category",
        title="Monthly Sales by Category (Normalised %)", palette=OKABE_ITO,
        normalize=True,
    )
    st.plotly_chart(fig_norm, use_container_width=True)


# ── Row 5: Rating Distributions ──────────────────────────────────────────────
st.subheader("4 · Customer Rating Distributions")
c1, c2 = st.columns(2)

with c1:
    fig_viol = violin_plot(
        filtered, x_col="category", y_col="avg_rating",
        title="Rating Distribution by Category (Violin)", palette=OKABE_ITO,
    )
    st.plotly_chart(fig_viol, use_container_width=True)

with c2:
    fig_ridge_r = ridge_plot(
        filtered, x_col="avg_rating", group_col="category",
        title="Rating Distributions — Ridge Plot by Category", palette=OKABE_ITO,
    )
    st.plotly_chart(fig_ridge_r, use_container_width=True)


# ── Row 6: Sentiment Breakdown ───────────────────────────────────────────────
st.subheader("5 · Customer Sentiment Analysis")
c1, c2 = st.columns(2)

with c1:
    pivot = sentiment_df.pivot(index="topic", columns="sentiment", values="count").fillna(0)
    pivot = pivot[["Positive", "Neutral", "Negative"]]
    fig_sent = go.Figure()
    for col, clr in SENTIMENT_COLORS.items():
        if col in pivot.columns:
            fig_sent.add_trace(go.Bar(
                name=col, x=pivot.index, y=pivot[col],
                marker_color=clr,
            ))
    fig_sent.update_layout(
        barmode="stack", title="Customer Sentiment by Topic",
        xaxis_title="Topic", yaxis_title="Count",
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
    )
    st.plotly_chart(fig_sent, use_container_width=True)

with c2:
    total_per_topic = sentiment_df.groupby("topic")["count"].sum()
    pct_df = sentiment_df.copy()
    pct_df["pct"] = pct_df.apply(
        lambda r: r["count"] / total_per_topic[r["topic"]] * 100, axis=1
    )
    fig_pct = go.Figure()
    for sent, clr in SENTIMENT_COLORS.items():
        sub = pct_df[pct_df["sentiment"] == sent]
        fig_pct.add_trace(go.Bar(
            name=sent, x=sub["topic"], y=sub["pct"],
            marker_color=clr,
        ))
    fig_pct.update_layout(
        barmode="stack", title="Sentiment Distribution (%) by Topic",
        xaxis_title="Topic", yaxis_title="Percentage (%)",
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
    )
    st.plotly_chart(fig_pct, use_container_width=True)


# ── Row 7: Profit vs Return Rate ─────────────────────────────────────────────
st.subheader("6 · Profit Margin vs Return Rate — Scatter")

cat_agg = filtered.groupby("category").agg(
    avg_profit=("profit", "mean"),
    avg_return_rate=("return_rate", "mean"),
    total_sales=("sales", "sum"),
    avg_rating=("avg_rating", "mean"),
).reset_index()

fig_scatter = go.Figure()
for i, row in cat_agg.iterrows():
    fig_scatter.add_trace(go.Scatter(
        x=[row["avg_return_rate"] * 100],
        y=[row["avg_profit"]],
        mode="markers+text",
        name=row["category"],
        marker=dict(
            size=row["total_sales"] / cat_agg["total_sales"].max() * 50 + 10,
            color=OKABE_ITO[i % len(OKABE_ITO)],
            opacity=0.8,
            line=dict(color="white", width=1.5),
        ),
        text=[row["category"]],
        textposition="top center",
        textfont=dict(size=10),
    ))

fig_scatter.update_layout(
    title="Avg Profit vs Return Rate (bubble size = total sales)",
    xaxis_title="Return Rate (%)",
    yaxis_title="Average Profit ($)",
    plot_bgcolor="#FAFAFA", paper_bgcolor="white",
    showlegend=False,
)
st.plotly_chart(fig_scatter, use_container_width=True)


# ── Row 8: AI Insights ────────────────────────────────────────────────────────
st.subheader("7 · AI Insight Discovery")

numeric_cols = ["sales", "profit", "avg_rating", "return_rate", "units"]
engine = InsightEngine(filtered, numeric_cols=numeric_cols, domain="ecommerce")
cards = engine.insight_cards()

electronics = filtered[filtered["category"] == "Electronics"]["sales"].sum()
total = filtered["sales"].sum()
electronics_share = electronics / total * 100 if total > 0 else 0

st.markdown(generate_domain_summary("ecommerce", {
    "n_categories": len(cat_sel),
    "n_regions": len(reg_sel),
    "total_sales": filtered["sales"].sum(),
    "electronics_share": electronics_share,
    "seasonal_peak": "Q4",
    "quality_pos": 62,
    "shipping_neg": 42,
}))

c1, c2, c3 = st.columns(3)
for i, card in enumerate(cards[:6]):
    with [c1, c2, c3][i % 3]:
        border = "#D55E00" if card["severity"] == "high" else "#009E73"
        st.markdown(
            f'<div style="border-left:4px solid {border};background:#F7F9FC;'
            f'border-radius:6px;padding:10px 12px;margin:6px 0;">'
            f'<b>{card["icon"]} {card["title"]}</b><br>'
            f'<span style="font-size:0.85rem;">{card["body"]}</span></div>',
            unsafe_allow_html=True,
        )


# ── Row 9: Critique ───────────────────────────────────────────────────────────
st.subheader("8 · Visualization Quality Critique — Treemap & Sankey")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Treemap Quality Report**")
    r = score_chart("treemap", system="gap_vizai")
    cols5 = st.columns(5)
    metrics = r.to_dict()
    keys = list(metrics.keys())[:5]
    for j, (k, v) in enumerate([(k, metrics[k]) for k in keys]):
        cols5[j].metric(k.split()[-1], f"{v*100:.0f}")
    for line in critique_text(r): st.markdown(line)

with c2:
    st.markdown("**Sankey Quality Report**")
    r2 = score_chart("sankey", system="gap_vizai")
    cols5b = st.columns(5)
    for j, (k, v) in enumerate([(k, r2.to_dict()[k]) for k in list(r2.to_dict().keys())[:5]]):
        cols5b[j].metric(k.split()[-1], f"{v*100:.0f}")
    for line in critique_text(r2): st.markdown(line)
