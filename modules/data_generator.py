"""
Synthetic data generators for three research domains:
  1. Healthcare (ICU patients, vitals, risk levels)
  2. Finance (stock returns and prices)
  3. E-Commerce (sales, sentiment, composition)
All generators are seeded for full reproducibility.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Healthcare ──────────────────────────────────────────────────────────────

def generate_healthcare_data(n_patients: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    risk_levels = rng.choice(["Low", "Medium", "High", "Critical"],
                              n_patients, p=[0.40, 0.30, 0.20, 0.10])
    risk_idx = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    records = []
    for i, risk in enumerate(risk_levels):
        r = risk_idx[risk]
        records.append({
            "patient_id":     f"P{i:04d}",
            "risk_level":     risk,
            "age":            int(rng.integers(18, 90)),
            "heart_rate":     float(np.clip(rng.normal(75 + r*15, 10 + r*5), 40, 180)),
            "systolic_bp":    float(np.clip(rng.normal(120 + r*15, 15), 70, 220)),
            "diastolic_bp":   float(np.clip(rng.normal(80 + r*5, 10), 40, 130)),
            "spo2":           float(np.clip(rng.normal(98 - r*3, 1.5), 70, 100)),
            "temperature":    float(np.clip(rng.normal(37.0 + r*0.5, 0.4), 35, 42)),
            "los_days":       float(np.clip(rng.exponential(3 + r*5), 1, 60)),
            "mortality":      int(rng.random() < (0.02 + r * 0.12)),
        })
    return pd.DataFrame(records)


def generate_temporal_health_data(n_patients: int = 60, n_hours: int = 48, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_patients):
        risk = rng.choice(["Low", "Medium", "High"], p=[0.5, 0.3, 0.2])
        base_hr = {"Low": 70, "Medium": 85, "High": 100}[risk]
        trend = rng.normal(0, 0.15)
        for h in range(n_hours):
            records.append({
                "patient_id": f"P{i:04d}",
                "hour":       h,
                "heart_rate": float(np.clip(base_hr + trend * h + rng.normal(0, 4), 40, 180)),
                "risk_level": risk,
            })
    return pd.DataFrame(records)


# ── Finance ──────────────────────────────────────────────────────────────────

def generate_finance_data(n_days: int = 365, seed: int = 42):
    rng = np.random.default_rng(seed)
    tickers = ["TECH_A", "TECH_B", "TECH_C", "FIN_D", "ENERGY_E"]
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # Correlated returns via Cholesky decomposition
    corr = np.array([
        [1.00, 0.70, 0.60, 0.30, 0.10],
        [0.70, 1.00, 0.65, 0.35, 0.15],
        [0.60, 0.65, 1.00, 0.30, 0.10],
        [0.30, 0.35, 0.30, 1.00, 0.20],
        [0.10, 0.15, 0.10, 0.20, 1.00],
    ])
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((5, n_days))
    correlated_Z = L @ Z  # shape (5, n_days)

    vols   = rng.uniform(0.012, 0.028, 5)
    drifts = rng.uniform(-0.0003, 0.0009, 5)

    returns, prices = {}, {}
    for i, tick in enumerate(tickers):
        r = drifts[i] + vols[i] * correlated_Z[i]
        returns[tick] = r
        prices[tick] = 100 * np.exp(np.cumsum(r))

    returns_df = pd.DataFrame(returns, index=dates)
    returns_df.index.name = "date"
    prices_df = pd.DataFrame(prices, index=dates)
    prices_df.index.name = "date"
    return returns_df, prices_df, tickers


# ── E-Commerce ────────────────────────────────────────────────────────────────

def generate_ecommerce_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = ["Electronics", "Clothing", "Books", "Food", "Home", "Sports", "Beauty"]
    regions    = ["North", "South", "East", "West"]
    records = []

    for cat in categories:
        base = rng.uniform(80_000, 600_000)
        for reg in regions:
            reg_factor = rng.uniform(0.7, 1.3)
            for month in range(1, 13):
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)
                sales = base * reg_factor * seasonal * rng.uniform(0.9, 1.1)
                margin = rng.uniform(0.05, 0.40)
                records.append({
                    "category":     cat,
                    "region":       reg,
                    "month":        month,
                    "sales":        round(sales, 2),
                    "profit":       round(sales * margin, 2),
                    "units":        int(sales / rng.uniform(20, 250)),
                    "return_rate":  round(rng.uniform(0.02, 0.15), 3),
                    "avg_rating":   float(np.clip(rng.normal(4.0, 0.5), 1, 5)),
                })
    return pd.DataFrame(records)


def generate_sentiment_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topics = ["Product Quality", "Shipping", "Customer Service", "Price", "Packaging"]
    probs  = {
        "Product Quality":  [0.62, 0.23, 0.15],
        "Shipping":         [0.38, 0.20, 0.42],
        "Customer Service": [0.50, 0.22, 0.28],
        "Price":            [0.30, 0.28, 0.42],
        "Packaging":        [0.65, 0.25, 0.10],
    }
    records = []
    for topic in topics:
        counts = rng.multinomial(300, probs[topic])
        for sent, cnt in zip(["Positive", "Neutral", "Negative"], counts):
            records.append({"topic": topic, "sentiment": sent, "count": int(cnt)})
    return pd.DataFrame(records)
