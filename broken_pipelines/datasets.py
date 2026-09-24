"""Synthetic but realistic datasets for the test-fixture pipelines.

Each generator produces a dataset with a genuine (weak-to-moderate) signal, so
that the flawed pipelines built on top of them fail for the reason we intend
rather than because the data is nonsense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 7


def credit_default(n: int = 1600, seed: int = SEED) -> pd.DataFrame:
    """Loan default classification, with a column that leaks the outcome."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = np.round(rng.lognormal(10.9, 0.45, n), -2)
    loan_amount = np.round(income * rng.uniform(0.2, 2.5, n), -2)
    employment_years = np.clip(rng.normal(6, 4, n), 0, 40).round(1)
    utilisation = np.clip(rng.beta(2, 3, n) + rng.normal(0, 0.05, n), 0, 1).round(3)
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.3, 0.25, 0.15])

    logit = (
        -1.1
        + 2.4 * utilisation
        + 0.9 * (loan_amount / np.maximum(income, 1))
        - 0.05 * employment_years
        - 0.015 * (age - 40)
    )
    default = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "age": age,
        "annual_income": income,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "credit_utilisation": utilisation,
        "region": region,
        # --- the planted leak: written by collections *after* the default happens
        "collections_contact_flag": np.where(rng.random(n) < 0.03, 1 - default, default),
        "extract_version": "v4",                       # constant column
        "defaulted": default,
    })
    df.loc[rng.random(n) < 0.12, "annual_income"] = np.nan      # real-world missingness
    return df


def telecom_churn(n: int = 1400, seed: int = SEED) -> pd.DataFrame:
    """Churn classification — clean input; the pipeline is what breaks it."""
    rng = np.random.default_rng(seed + 1)
    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.uniform(20, 120, n), 2)
    tickets = rng.poisson(1.2, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    fibre = rng.binomial(1, 0.42, n)

    logit = (
        -0.4
        - 0.035 * tenure
        + 0.012 * monthly
        + 0.28 * tickets
        + np.where(contract == "Month-to-month", 0.9, -0.5)
        + 0.3 * fibre
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_tickets": tickets,
        "contract_type": contract,
        "has_fibre": fibre,
        "churned": churn,
    })


def card_fraud(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Severely imbalanced fraud detection (~2% positives)."""
    rng = np.random.default_rng(seed + 2)
    amount = np.round(rng.lognormal(3.4, 1.1, n), 2)
    hour = rng.integers(0, 24, n)
    category = rng.choice(
        ["grocery", "fuel", "electronics", "travel", "gaming", "restaurant"],
        n, p=[0.3, 0.2, 0.12, 0.1, 0.08, 0.2],
    )
    device_age_days = rng.integers(0, 1500, n)
    foreign = rng.binomial(1, 0.07, n)

    logit = (
        -5.2
        + 0.6 * np.log1p(amount) / 2
        + 0.9 * foreign
        + np.where(np.isin(category, ["gaming", "electronics"]), 0.8, 0)
        + np.where((hour < 5) | (hour > 22), 0.7, 0)
        - 0.0008 * device_age_days
    )
    fraud = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "amount": amount,
        "hour_of_day": hour,
        "merchant_category": category,
        "device_age_days": device_age_days,
        "foreign_transaction": foreign,
        "is_fraud": fraud,
    })


def store_sales(n_stores: int = 40, days: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Daily store sales — time-ordered, repeated entities, messy categories."""
    rng = np.random.default_rng(seed + 3)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    store_level = rng.normal(9000, 2600, n_stores)
    raw_regions = ["North", "north ", "NORTH", "South", "south", "East", "West"]
    for s in range(n_stores):
        region = raw_regions[s % len(raw_regions)]
        for i, d in enumerate(dates):
            promo = int(rng.random() < 0.18)
            dow_effect = [1.0, 0.95, 0.97, 1.02, 1.18, 1.35, 0.8][d.dayofweek]
            trend = 1 + 0.0009 * i
            sales = store_level[s] * dow_effect * trend * (1 + 0.22 * promo) * rng.normal(1, 0.09)
            rows.append({
                "date": d,
                "store_id": f"S{s:03d}",
                "region": region,
                "day_of_week": d.dayofweek,
                "promo_running": promo,
                "weather": rng.choice(["clear", "rain", "unknown", "snow"], p=[0.6, 0.25, 0.1, 0.05]),
                "sales": round(max(sales, 0), 2),
            })
    df = pd.DataFrame(rows).sort_values(["date", "store_id"]).reset_index(drop=True)

    # --- the planted leak: a centred rolling mean, so each row's feature is
    # computed partly from days that have not happened yet.
    df["rolling_7d_avg_sales"] = (
        df.groupby("store_id")["sales"]
        .transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
        .round(2)
    )
    return df
