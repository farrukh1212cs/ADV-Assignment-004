"""
Uncertainty visualization module:
  - Time series with bootstrap confidence intervals
  - Prediction fan charts (forecast uncertainty)
  - Error bands on rolling statistics
  - Confidence band comparisons
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import t as t_dist
from modules.perceptual import OKABE_ITO, apply_template, color_with_alpha


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Return (lower, upper) bootstrap CI for a given statistic."""
    rng = np.random.default_rng(seed)
    boot = [statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    alpha = 1 - ci
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def rolling_mean_ci(series: pd.Series, window: int = 7, ci: float = 0.95):
    """Compute rolling mean with analytical t-distribution CI."""
    rm = series.rolling(window).mean()
    rs = series.rolling(window).std()
    t_crit = t_dist.ppf(0.5 + ci / 2, df=window - 1)
    margin = t_crit * rs / np.sqrt(window)
    return rm, rm - margin, rm + margin


def time_series_with_ci(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: str = None,
    window: int = 7,
    ci: float = 0.95,
    title: str = "Time Series with Confidence Bands",
    palette: list = None,
) -> go.Figure:
    palette = palette or OKABE_ITO
    fig = go.Figure()
    groups = df[group_col].unique() if group_col else [None]

    for i, grp in enumerate(groups):
        sub = df[df[group_col] == grp].sort_values(date_col) if grp is not None else df.sort_values(date_col)
        dates = sub[date_col]
        vals  = sub[value_col]
        color = palette[i % len(palette)]
        name  = str(grp) if grp is not None else value_col

        rm, lo, hi = rolling_mean_ci(vals.reset_index(drop=True), window=window, ci=ci)
        valid = ~rm.isna()
        d_v = dates.reset_index(drop=True)[valid]
        rm_v, lo_v, hi_v = rm[valid], lo[valid], hi[valid]

        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([d_v, d_v[::-1]]),
            y=pd.concat([hi_v, lo_v[::-1]]),
            fill="toself",
            fillcolor=color_with_alpha(color, 0.18),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Rolling mean
        fig.add_trace(go.Scatter(
            x=d_v, y=rm_v,
            mode="lines",
            name=f"{name} (mean ±{int(ci*100)}% CI)",
            line=dict(color=color, width=2.5),
        ))
        # Raw data
        fig.add_trace(go.Scatter(
            x=dates, y=vals,
            mode="lines",
            name=f"{name} (raw)",
            line=dict(color=color, width=0.8, dash="dot"),
            opacity=0.4,
            showlegend=False,
        ))

    fig.update_layout(title=title, xaxis_title=date_col, yaxis_title=value_col)
    return apply_template(fig)


def fan_chart(
    historical_x, historical_y,
    forecast_x,  forecast_mean,
    forecast_std,
    title: str = "Forecast with Uncertainty Bands",
    color: str = "#0072B2",
    n_bands: int = 3,
) -> go.Figure:
    """
    Fan chart: historical line + forecast mean + uncertainty bands at
    1σ, 2σ, 3σ (or configurable n_bands quantile pairs).
    """
    alphas = np.linspace(0.30, 0.08, n_bands)
    sigma_levels = np.arange(1, n_bands + 1, dtype=float)

    fig = go.Figure()

    # Add bands from widest to narrowest
    for k, (sigma, alpha) in enumerate(zip(sigma_levels[::-1], alphas)):
        lo = forecast_mean - sigma * forecast_std
        hi = forecast_mean + sigma * forecast_std
        fig.add_trace(go.Scatter(
            x=np.concatenate([forecast_x, forecast_x[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=color_with_alpha(color, alpha),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=k == 0,
            name=f"Uncertainty band (±{int(sigma)}σ)",
            hoverinfo="skip",
        ))

    # Forecast mean
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_mean,
        mode="lines",
        name="Forecast mean",
        line=dict(color=color, width=2.5, dash="dash"),
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_x, y=historical_y,
        mode="lines",
        name="Historical",
        line=dict(color=color, width=2.5),
    ))

    # Separator
    if len(historical_x):
        fig.add_vline(x=historical_x[-1], line=dict(color="gray", dash="dot", width=1))

    fig.update_layout(title=title)
    return apply_template(fig)


def error_band_comparison(
    groups: dict,
    title: str = "Mean ± 95% CI Comparison",
    palette: list = None,
) -> go.Figure:
    """
    Bar chart with bootstrapped error bars for comparing group statistics.
    `groups` is {label: array_like}.
    """
    palette = palette or OKABE_ITO
    labels, means, lowers, uppers, colors = [], [], [], [], []

    for i, (label, data) in enumerate(groups.items()):
        data = np.asarray(data)
        lo, hi = bootstrap_ci(data)
        m = float(np.mean(data))
        labels.append(label)
        means.append(m)
        lowers.append(m - lo)
        uppers.append(hi - m)
        colors.append(palette[i % len(palette)])

    fig = go.Figure(go.Bar(
        x=labels,
        y=means,
        error_y=dict(type="data", symmetric=False, array=uppers, arrayminus=lowers, visible=True),
        marker_color=colors,
    ))
    fig.update_layout(title=title, yaxis_title="Mean value")
    return apply_template(fig)
