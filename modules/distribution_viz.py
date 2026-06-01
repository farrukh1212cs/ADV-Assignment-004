"""
Statistical distribution visualizations:
  - ECDF (Empirical Cumulative Distribution Function)
  - KDE vs Histogram comparison
  - Violin plots
  - Ridge plots (stacked density)
  - Box + strip plots
All return Plotly figures with perceptual defaults applied.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from modules.perceptual import OKABE_ITO, apply_template, color_with_alpha


def ecdf_plot(
    df: pd.DataFrame,
    x_col: str,
    color_col: str = None,
    title: str = "ECDF",
    palette: list = None,
) -> go.Figure:
    palette = palette or OKABE_ITO
    fig = go.Figure()

    groups = df[color_col].unique() if color_col else [None]
    for i, grp in enumerate(groups):
        data = df[df[color_col] == grp][x_col].dropna() if grp is not None else df[x_col].dropna()
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=sorted_data, y=cdf,
            mode="lines",
            name=str(grp) if grp is not None else x_col,
            line=dict(color=color, width=2.5),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Cumulative Probability",
        yaxis=dict(range=[0, 1.05]),
    )
    return apply_template(fig)


def violin_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    title: str = "Violin Plot",
    palette: list = None,
    color_map: dict = None,
) -> go.Figure:
    palette = palette or OKABE_ITO
    groups = df[x_col].unique()
    fig = go.Figure()

    for i, grp in enumerate(groups):
        sub = df[df[x_col] == grp][y_col].dropna()
        color = (color_map or {}).get(grp, palette[i % len(palette)])
        fig.add_trace(go.Violin(
            y=sub,
            name=str(grp),
            box_visible=True,
            meanline_visible=True,
            fillcolor=color_with_alpha(color, 0.5),
            line_color=color,
            points="outliers",
            jitter=0.3,
        ))

    fig.update_layout(title=title, yaxis_title=y_col)
    return apply_template(fig)


def kde_histogram_comparison(
    df: pd.DataFrame,
    x_col: str,
    color_col: str = None,
    title: str = "KDE vs Histogram",
    palette: list = None,
) -> go.Figure:
    palette = palette or OKABE_ITO
    groups = df[color_col].unique() if color_col else [None]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Histogram (binned counts)", "KDE (kernel density estimate)"],
        shared_yaxes=False,
    )

    for i, grp in enumerate(groups):
        data = df[df[color_col] == grp][x_col].dropna() if grp is not None else df[x_col].dropna()
        color = palette[i % len(palette)]
        name = str(grp) if grp is not None else x_col

        # Histogram
        fig.add_trace(
            go.Histogram(x=data, name=name, marker_color=color_with_alpha(color, 0.6),
                         marker_line=dict(color=color, width=1),
                         nbinsx=30, showlegend=True, legendgroup=name),
            row=1, col=1,
        )

        # KDE
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min() - data.std(), data.max() + data.std(), 300)
        fig.add_trace(
            go.Scatter(x=x_range, y=kde(x_range), mode="lines",
                       name=name, line=dict(color=color, width=2.5),
                       fill="tozeroy", fillcolor=color_with_alpha(color, 0.15),
                       showlegend=False, legendgroup=name),
            row=1, col=2,
        )

    fig.update_layout(title=title, barmode="overlay")
    return apply_template(fig)


def ridge_plot(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    title: str = "Ridge Plot",
    palette: list = None,
    color_map: dict = None,
) -> go.Figure:
    """Stacked KDE ridge plot — each group is offset vertically."""
    palette = palette or OKABE_ITO
    groups = df[group_col].unique()
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    x_pad = (x_max - x_min) * 0.1
    x_range = np.linspace(x_min - x_pad, x_max + x_pad, 400)

    # Compute all KDEs first to find spacing
    kde_results = {}
    max_y = 0
    for grp in groups:
        data = df[df[group_col] == grp][x_col].dropna()
        if len(data) < 2:
            continue
        kde = gaussian_kde(data)
        y = kde(x_range)
        kde_results[grp] = y
        max_y = max(max_y, y.max())

    spacing = max_y * 2.0
    fig = go.Figure()

    tick_vals, tick_texts = [], []
    for i, grp in enumerate(groups):
        if grp not in kde_results:
            continue
        y = kde_results[grp]
        offset = i * spacing
        color = (color_map or {}).get(grp, palette[i % len(palette)])

        # Filled area
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_range, x_range[::-1]]),
            y=np.concatenate([y + offset, np.full_like(x_range, offset)]),
            fill="toself",
            fillcolor=color_with_alpha(color, 0.5),
            line=dict(color=color, width=2),
            name=str(grp),
            hoverinfo="name",
        ))
        # Baseline
        fig.add_trace(go.Scatter(
            x=[x_range[0], x_range[-1]], y=[offset, offset],
            mode="lines", line=dict(color="#CCCCCC", width=0.8),
            showlegend=False, hoverinfo="skip",
        ))
        tick_vals.append(offset + max_y * 0.3)
        tick_texts.append(str(grp))

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis=dict(tickvals=tick_vals, ticktext=tick_texts, showgrid=False),
        showlegend=False,
    )
    return apply_template(fig)


def box_strip_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Box Plot",
    palette: list = None,
    color_map: dict = None,
) -> go.Figure:
    palette = palette or OKABE_ITO
    groups = df[x_col].unique()
    fig = go.Figure()

    for i, grp in enumerate(groups):
        sub = df[df[x_col] == grp][y_col].dropna()
        color = (color_map or {}).get(grp, palette[i % len(palette)])
        fig.add_trace(go.Box(
            y=sub, name=str(grp),
            marker_color=color,
            boxmean="sd",
            jitter=0.4,
            pointpos=-1.8,
            boxpoints="outliers",
        ))

    fig.update_layout(title=title, yaxis_title=y_col)
    return apply_template(fig)
