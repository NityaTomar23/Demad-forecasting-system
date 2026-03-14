"""
Generate a realistic synthetic sales dataset for the Demand Forecasting System.

Creates ~2 years of daily sales data across 3 stores and 5 products
with built-in trend, seasonality, promotions, and holiday effects.
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

STORES = ["Store_A", "Store_B", "Store_C"]
PRODUCTS = ["Product_1", "Product_2", "Product_3", "Product_4", "Product_5"]

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

SEED = 42

# Base daily sales range per product (mean, amplitude)
PRODUCT_BASE = {
    "Product_1": (50, 15),
    "Product_2": (80, 20),
    "Product_3": (30, 10),
    "Product_4": (65, 18),
    "Product_5": (45, 12),
}

# Store-level multipliers
STORE_MULTIPLIER = {
    "Store_A": 1.0,
    "Store_B": 1.3,
    "Store_C": 0.85,
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def _weekly_seasonality(day_of_week: np.ndarray) -> np.ndarray:
    """Weekends (5,6) get a sales boost."""
    return np.where(np.isin(day_of_week, [5, 6]), 1.25, 1.0)


def _monthly_seasonality(month: np.ndarray) -> np.ndarray:
    """Nov–Dec holiday season boost; Jan–Feb dip."""
    seasonal = np.ones_like(month, dtype=float)
    seasonal[np.isin(month, [11, 12])] = 1.35
    seasonal[np.isin(month, [1, 2])] = 0.80
    seasonal[np.isin(month, [6, 7])] = 1.10   # mild summer lift
    return seasonal


def _trend(n_days: int, slope: float = 0.02) -> np.ndarray:
    """Gentle upward trend over time."""
    return 1.0 + slope * np.arange(n_days) / n_days


# ── Main generator ──────────────────────────────────────────────────────────


def generate_sales_data() -> pd.DataFrame:
    """Return a DataFrame with columns: date, store, product, sales, promotion, holiday."""

    rng = np.random.default_rng(SEED)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    n_days = len(dates)

    rows = []

    for store in STORES:
        for product in PRODUCTS:
            base_mean, amplitude = PRODUCT_BASE[product]
            multiplier = STORE_MULTIPLIER[store]

            # Components
            trend = _trend(n_days)
            weekly = _weekly_seasonality(dates.dayofweek.values)
            monthly = _monthly_seasonality(dates.month.values)
            noise = rng.normal(0, amplitude, size=n_days)

            # Promotions: ~10 % of days
            promotion = rng.binomial(1, 0.10, size=n_days)
            promo_boost = np.where(promotion == 1, 1.20, 1.0)

            # Holidays: weekends + ~3 % random extra holidays
            is_weekend = np.isin(dates.dayofweek.values, [5, 6]).astype(int)
            extra_holiday = rng.binomial(1, 0.03, size=n_days)
            holiday = np.clip(is_weekend + extra_holiday, 0, 1)
            holiday_boost = np.where((holiday == 1) & (is_weekend == 0), 1.10, 1.0)

            # Final sales
            sales = (
                base_mean
                * multiplier
                * trend
                * weekly
                * monthly
                * promo_boost
                * holiday_boost
                + noise
            )
            sales = np.maximum(sales, 0).round().astype(int)

            for i, d in enumerate(dates):
                rows.append(
                    {
                        "date": d,
                        "store": store,
                        "product": product,
                        "sales": sales[i],
                        "promotion": int(promotion[i]),
                        "holiday": int(holiday[i]),
                    }
                )

    df = pd.DataFrame(rows)
    df.sort_values(["store", "product", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Entry-point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic sales dataset ...")
    df = generate_sales_data()

    out_path = os.path.join(os.path.dirname(__file__), "sales.csv")
    df.to_csv(out_path, index=False)

    print(f"[OK] Saved {len(df):,} rows to {out_path}")
    print(f"  Date range : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Stores     : {df['store'].nunique()}")
    print(f"  Products   : {df['product'].nunique()}")
    print(df.head(10).to_string(index=False))
