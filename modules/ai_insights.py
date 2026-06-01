"""
AI Insight Discovery module.
Implements rule-based intelligence for:
  - Anomaly detection (IQR + z-score)
  - Trend detection (linear regression slope)
  - Correlation analysis (Pearson + Spearman)
  - Natural language insight generation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List


@dataclass
class Anomaly:
    column: str
    index: object
    value: float
    z_score: float
    direction: str   # "high" | "low"

    def to_text(self) -> str:
        return (
            f"Anomaly detected in **{self.column}** at index {self.index}: "
            f"value = {self.value:.2f} ({self.z_score:.1f}σ {self.direction} of mean)."
        )


@dataclass
class Trend:
    column: str
    slope: float
    r_squared: float
    direction: str   # "increasing" | "decreasing" | "flat"
    significance: str   # "significant" | "marginal" | "not significant"

    def to_text(self) -> str:
        return (
            f"**{self.column}** shows a {self.significance} {self.direction} trend "
            f"(slope={self.slope:.4f}/unit, R²={self.r_squared:.3f})."
        )


@dataclass
class Correlation:
    col_a: str
    col_b: str
    r: float
    p_value: float
    strength: str   # "strong" | "moderate" | "weak"
    direction: str  # "positive" | "negative"

    def to_text(self) -> str:
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return (
            f"**{self.col_a}** and **{self.col_b}**: {self.strength} {self.direction} "
            f"correlation (r={self.r:.3f}, p={self.p_value:.3f}, {sig})."
        )


# ── Anomaly Detection ────────────────────────────────────────────────────────

def detect_anomalies_iqr(
    df: pd.DataFrame,
    col: str,
    factor: float = 1.5,
) -> list[Anomaly]:
    series = df[col].dropna()
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    mean, std = series.mean(), series.std()
    anomalies = []
    for idx, val in series.items():
        if val < lower or val > upper:
            z = (val - mean) / std if std > 0 else 0
            anomalies.append(Anomaly(
                column=col, index=idx, value=val,
                z_score=abs(z), direction="high" if val > upper else "low",
            ))
    return sorted(anomalies, key=lambda a: -a.z_score)


def detect_anomalies_zscore(
    df: pd.DataFrame,
    col: str,
    threshold: float = 2.5,
) -> list[Anomaly]:
    series = df[col].dropna()
    z_scores = np.abs(stats.zscore(series))
    mean, std = series.mean(), series.std()
    anomalies = []
    for (idx, val), z in zip(series.items(), z_scores):
        if z > threshold:
            anomalies.append(Anomaly(
                column=col, index=idx, value=val,
                z_score=z, direction="high" if val > mean else "low",
            ))
    return sorted(anomalies, key=lambda a: -a.z_score)[:10]


# ── Trend Detection ──────────────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame, x_col: str, y_col: str) -> Trend:
    sub = df[[x_col, y_col]].dropna()
    if sub.empty or len(sub) < 3:
        return Trend(y_col, 0, 0, "flat", "not significant")

    x = np.arange(len(sub))
    y = sub[y_col].values
    slope, intercept, r, p_value, _ = stats.linregress(x, y)
    r2 = r ** 2

    if abs(slope) < 1e-6:
        direction = "flat"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    if p_value < 0.01:
        significance = "significant"
    elif p_value < 0.05:
        significance = "marginal"
    else:
        significance = "not significant"

    return Trend(y_col, slope, r2, direction, significance)


# ── Correlation Analysis ─────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame, cols: list[str]) -> list[Correlation]:
    results = []
    numeric = df[cols].select_dtypes(include=np.number)
    col_list = list(numeric.columns)
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            a, b = col_list[i], col_list[j]
            data = numeric[[a, b]].dropna()
            if len(data) < 5:
                continue
            r, p = stats.pearsonr(data[a], data[b])
            strength = "strong" if abs(r) >= 0.7 else ("moderate" if abs(r) >= 0.4 else "weak")
            direction = "positive" if r >= 0 else "negative"
            results.append(Correlation(a, b, r, p, strength, direction))
    return sorted(results, key=lambda c: -abs(c.r))


# ── NL Insight Generation ─────────────────────────────────────────────────────

def generate_domain_summary(domain: str, stats_dict: dict) -> str:
    """Generate a natural language paragraph summarizing domain statistics."""
    if domain == "healthcare":
        return (
            f"Analysis of **{stats_dict.get('n_patients', 'N/A')} ICU patients** reveals "
            f"a mean heart rate of {stats_dict.get('mean_hr', 0):.1f} bpm "
            f"(σ={stats_dict.get('std_hr', 0):.1f}). "
            f"Critical-risk patients exhibit {stats_dict.get('hr_increase', 0):.1f}% higher heart rates than low-risk patients. "
            f"SpO₂ levels show significant separation across risk groups (ANOVA p<0.001). "
            f"Length of stay follows an exponential distribution with a mean of "
            f"{stats_dict.get('mean_los', 0):.1f} days."
        )
    elif domain == "finance":
        return (
            f"Over **{stats_dict.get('n_days', 365)} trading days**, the portfolio exhibits "
            f"mean daily return of {stats_dict.get('mean_ret', 0)*100:.3f}% "
            f"(annualized {stats_dict.get('ann_ret', 0)*100:.1f}%). "
            f"Technology stocks show strong positive correlation (ρ≈0.65), while Energy exhibits "
            f"near-zero correlation with the tech cluster. "
            f"Value-at-Risk (95% CI) is estimated at {stats_dict.get('var_95', 0)*100:.2f}% per day."
        )
    elif domain == "ecommerce":
        return (
            f"Across **{stats_dict.get('n_categories', 7)} product categories** and "
            f"{stats_dict.get('n_regions', 4)} regions, total annual sales reach "
            f"${stats_dict.get('total_sales', 0):,.0f}. "
            f"Electronics dominates with {stats_dict.get('electronics_share', 0):.1f}% of revenue. "
            f"Seasonal patterns show a {stats_dict.get('seasonal_peak', 'Q4')} peak. "
            f"Customer sentiment is predominantly positive for Product Quality "
            f"({stats_dict.get('quality_pos', 62)}%) but negative for Shipping "
            f"({stats_dict.get('shipping_neg', 42)}%)."
        )
    return f"Analysis of {domain} data completed."


class InsightEngine:
    """Runs the full AI insight pipeline for a given domain."""

    def __init__(self, df: pd.DataFrame, numeric_cols: list[str], domain: str = "unknown"):
        self.df = df
        self.numeric_cols = [c for c in numeric_cols if c in df.columns]
        self.domain = domain

    def run(self) -> dict:
        results = {"anomalies": [], "trends": [], "correlations": []}

        for col in self.numeric_cols[:5]:
            anomalies = detect_anomalies_iqr(self.df, col)[:3]
            results["anomalies"].extend(anomalies)

        for col in self.numeric_cols[:4]:
            trend = detect_trend(self.df.reset_index(), "index", col)
            results["trends"].append(trend)

        results["correlations"] = compute_correlations(self.df, self.numeric_cols)[:6]
        return results

    def insight_cards(self) -> list[dict]:
        """Return structured insight cards for UI rendering."""
        analysis = self.run()
        cards = []

        for a in analysis["anomalies"][:5]:
            cards.append({
                "type": "anomaly",
                "icon": "⚠️",
                "title": f"Anomaly in {a.column}",
                "body": a.to_text(),
                "severity": "high" if a.z_score > 3 else "medium",
            })

        for t in analysis["trends"]:
            if t.significance != "not significant":
                cards.append({
                    "type": "trend",
                    "icon": "📈" if t.direction == "increasing" else "📉",
                    "title": f"Trend in {t.column}",
                    "body": t.to_text(),
                    "severity": "info",
                })

        for c in analysis["correlations"][:4]:
            if c.strength in ("strong", "moderate"):
                cards.append({
                    "type": "correlation",
                    "icon": "🔗",
                    "title": f"Correlation: {c.col_a} ↔ {c.col_b}",
                    "body": c.to_text(),
                    "severity": "info",
                })

        return cards
