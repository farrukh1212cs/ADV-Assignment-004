"""
Composition visualization module:
  - Treemaps (hierarchical part-to-whole)
  - Sankey diagrams (flow composition)
  - Stacked bar charts
  - Waffle chart approximation
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.perceptual import OKABE_ITO, apply_template, color_with_alpha


def treemap_chart(
    df: pd.DataFrame,
    path: list[str],
    values: str,
    color: str = None,
    title: str = "Treemap",
    palette: list = None,
) -> go.Figure:
    palette = palette or OKABE_ITO
    kwargs = dict(path=path, values=values, title=title)
    if color:
        kwargs["color"] = color
        kwargs["color_continuous_scale"] = ["#56B4E9", "#0072B2"]
    else:
        kwargs["color"] = path[0]
        kwargs["color_discrete_sequence"] = palette

    fig = px.treemap(df, **kwargs)
    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percentParent:.1%}<extra></extra>",
        marker_line=dict(color="white", width=2),
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def sankey_chart(
    sources: list[str],
    targets: list[str],
    values: list[float],
    title: str = "Sankey Diagram",
    palette: list = None,
) -> go.Figure:
    palette = palette or OKABE_ITO
    all_nodes = list(dict.fromkeys(sources + targets))
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    src_idx = [node_idx[s] for s in sources]
    tgt_idx = [node_idx[t] for t in targets]
    link_colors = [color_with_alpha(palette[i % len(palette)], 0.4) for i in src_idx]
    node_colors = [palette[i % len(palette)] for i in range(len(all_nodes))]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=22,
            line=dict(color="#333333", width=0.8),
            label=all_nodes,
            color=node_colors,
        ),
        link=dict(
            source=src_idx,
            target=tgt_idx,
            value=values,
            color=link_colors,
            hovertemplate="%{source.label} → %{target.label}<br>Value: %{value:,.0f}<extra></extra>",
        ),
    ))
    fig.update_layout(title=title, font_size=12, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def stacked_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = "Stacked Bar Chart",
    palette: list = None,
    normalize: bool = False,
) -> go.Figure:
    palette = palette or OKABE_ITO
    groups = df[color_col].unique()
    x_vals = df[x_col].unique()
    fig = go.Figure()

    for i, grp in enumerate(groups):
        sub = df[df[color_col] == grp]
        color = palette[i % len(palette)]

        if normalize:
            totals = df.groupby(x_col)[y_col].sum()
            y_vals = []
            for x in x_vals:
                row = sub[sub[x_col] == x]
                total = totals.get(x, 1)
                y_vals.append(float(row[y_col].sum()) / total if total > 0 else 0)
            y_label = "Share"
        else:
            y_dict = sub.groupby(x_col)[y_col].sum().to_dict()
            y_vals = [y_dict.get(x, 0) for x in x_vals]
            y_label = y_col

        fig.add_trace(go.Bar(
            x=x_vals, y=y_vals, name=str(grp),
            marker_color=palette[i % len(palette)],
        ))

    fig.update_layout(
        title=title,
        barmode="stack",
        yaxis_title="Share (proportion)" if normalize else y_col,
        xaxis_title=x_col,
    )
    return apply_template(fig)


def build_ecommerce_sankey(df: pd.DataFrame) -> go.Figure:
    """Build a Region → Category Sankey from the e-commerce dataframe."""
    region_cat = df.groupby(["region", "category"])["sales"].sum().reset_index()
    sources = list(region_cat["region"])
    targets = list(region_cat["category"])
    values  = list(region_cat["sales"] / 1000)   # in thousands
    return sankey_chart(sources, targets, values,
                        title="Sales Flow: Region → Category (in $K)")


def correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    title: str = "Correlation Matrix",
) -> go.Figure:
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, "#D55E00"], [0.5, "#FFFFFF"], [1, "#0072B2"]],
        zmid=0,
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{x} ↔ %{y}: r=%{z:.3f}<extra></extra>",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(title=title, height=400)
    return apply_template(fig)
