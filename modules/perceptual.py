"""
Perceptual design module: colorblind-safe palettes, typography, accessibility scoring.
Based on the Okabe-Ito palette (IEEE recommended for colorblind accessibility).
"""
import numpy as np
import plotly.graph_objects as go

# Okabe-Ito colorblind-safe palette (8 colors)
OKABE_ITO = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish Green
    "#56B4E9",  # Sky Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#F0E442",  # Yellow
    "#000000",  # Black
]

# Extended palette for more categories
EXTENDED_PALETTE = OKABE_ITO + [
    "#999999", "#E8A838", "#5E4FA2", "#66C2A5",
]

# Semantic color mapping
RISK_COLORS = {
    "Low": "#009E73",
    "Medium": "#E69F00",
    "High": "#D55E00",
    "Critical": "#CC0000",
}

SENTIMENT_COLORS = {
    "Positive": "#009E73",
    "Neutral": "#56B4E9",
    "Negative": "#D55E00",
}

TYPOGRAPHY = {
    "title_size": 18,
    "axis_title_size": 13,
    "tick_size": 11,
    "annotation_size": 11,
    "font_family": "Arial, sans-serif",
}


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def relative_luminance(hex_color: str) -> float:
    r, g, b = hex_to_rgb(hex_color)
    components = []
    for c in (r, g, b):
        components.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * components[0] + 0.7152 * components[1] + 0.0722 * components[2]


def wcag_contrast_ratio(fg: str, bg: str = "#FFFFFF") -> float:
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def accessibility_score(palette: list, bg: str = "#FFFFFF") -> dict:
    """Compute WCAG accessibility score for a palette."""
    ratios = [wcag_contrast_ratio(c, bg) for c in palette]
    aa_pass = [r >= 4.5 for r in ratios]
    aaa_pass = [r >= 7.0 for r in ratios]
    return {
        "ratios": ratios,
        "aa_compliance": sum(aa_pass) / len(aa_pass),
        "aaa_compliance": sum(aaa_pass) / len(aaa_pass),
        "mean_contrast": np.mean(ratios),
    }


def get_plotly_template() -> dict:
    """Returns a perceptually optimized Plotly layout template."""
    return dict(
        font=dict(family=TYPOGRAPHY["font_family"], size=TYPOGRAPHY["tick_size"]),
        title_font=dict(size=TYPOGRAPHY["title_size"], color="#1A1A2E"),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        colorway=OKABE_ITO,
        xaxis=dict(
            gridcolor="#EBEBEB",
            linecolor="#CCCCCC",
            title_font=dict(size=TYPOGRAPHY["axis_title_size"]),
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        yaxis=dict(
            gridcolor="#EBEBEB",
            linecolor="#CCCCCC",
            title_font=dict(size=TYPOGRAPHY["axis_title_size"]),
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            font=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    )


def apply_template(fig: go.Figure) -> go.Figure:
    """Apply the perceptual template to a Plotly figure."""
    tmpl = get_plotly_template()
    fig.update_layout(**tmpl)
    return fig


def color_with_alpha(hex_color: str, alpha: float = 0.4) -> str:
    """Convert hex to rgba string."""
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"
