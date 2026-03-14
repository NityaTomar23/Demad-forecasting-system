"""
Feature engineering module for the Demand Forecasting System.

Converts raw sales data into a supervised ML dataset by creating:
- Lag features  (past sales values)
- Rolling statistics  (mean, std over recent windows)
- Calendar / seasonal features  (day of week, month, etc.)
"""

import pandas as pd
import numpy as np


# ── Lag features ────────────────────────────────────────────────────────────


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    target: str = "sales",
) -> pd.DataFrame:
    """Add lagged sales columns per (store, product) group.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date within each group.
    lags : list[int]
        Number of days to look back (default: [7, 14, 28]).
    target : str
        Column to lag.

    Returns
    -------
    pd.DataFrame
        DataFrame with new ``sales_lag_<N>`` columns.
    """
    if lags is None:
        lags = [7, 14, 28]

    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = (
            df.groupby(["store", "product"])[target].shift(lag)
        )
    return df


# ── Rolling statistics ──────────────────────────────────────────────────────


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    target: str = "sales",
) -> pd.DataFrame:
    """Add rolling mean and std over recent windows.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by date within groups.
    windows : list[int]
        Window sizes in days (default: [7, 14, 30]).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``sales_roll_mean_<W>`` and ``sales_roll_std_<W>`` columns.
    """
    if windows is None:
        windows = [7, 14, 30]

    df = df.copy()
    for w in windows:
        grp = df.groupby(["store", "product"])[target]
        df[f"{target}_roll_mean_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"{target}_roll_std_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
    return df


# ── Calendar / seasonal features ────────────────────────────────────────────


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the ``date`` column.

    New columns: ``day_of_week``, ``month``, ``week_of_year``, ``is_weekend``.
    """
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


# ── Categorical encoding ────────────────────────────────────────────────────


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode ``store`` and ``product`` so tree models can use them."""
    df = df.copy()
    for col in ["store", "product"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


# ── Orchestrator ─────────────────────────────────────────────────────────────


def prepare_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    1. Add lag features
    2. Add rolling features
    3. Add date features
    4. Encode categoricals
    5. Drop rows with NaN (caused by lagging)

    Returns
    -------
    pd.DataFrame
        Ready-to-train dataset.
    """
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=windows)
    df = add_date_features(df)
    df = encode_categoricals(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Feature engineering done - dropped {before - len(df):,} NaN rows, "
          f"{len(df):,} rows remain with {df.shape[1]} columns.")
    return df


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from data_processing import load_data, clean_data

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sales.csv")
    df = load_data(csv_path)
    df = clean_data(df)
    df = prepare_features(df)
    print(df.head())
    print("\nFeature columns:")
    print(df.columns.tolist())
