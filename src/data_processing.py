"""
Data processing module for the Demand Forecasting System.

Handles loading, cleaning, and train/test splitting of sales data.
Uses a time-based split to respect the temporal ordering of observations.
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load sales CSV and parse dates.

    Parameters
    ----------
    path : str
        Path to the sales CSV file.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with a parsed ``date`` column.
    """
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    df.sort_values(["store", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and handle missing / invalid values.

    Steps
    -----
    1. Drop duplicate rows.
    2. Forward-fill missing ``sales`` within each (store, product) group.
    3. Fill remaining NaNs in ``promotion`` and ``holiday`` with 0.
    4. Clip ``sales`` to >= 0.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input DataFrame.
    """
    required_cols = {"date", "store", "sales", "promotion", "holiday"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.drop_duplicates().copy()

    # Forward-fill sales within each group
    df["sales"] = (
        df.groupby(["store"])["sales"]
        .transform(lambda s: s.ffill().bfill())
    )

    df["promotion"] = df["promotion"].fillna(0).astype(int)
    df["holiday"] = df["holiday"].fillna(0).map(lambda x: 0 if str(x).strip() == "0" else 1).astype(int)

    # Ensure non-negative sales
    df["sales"] = df["sales"].clip(lower=0)

    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split.

    The most recent ``test_ratio`` fraction of dates goes to the test set,
    keeping temporal ordering intact (no data leakage from the future).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (must contain a ``date`` column).
    test_ratio : float
        Fraction of unique dates to reserve for testing.

    Returns
    -------
    (train_df, test_df)
    """
    unique_dates = sorted(df["date"].unique())
    split_idx = int(len(unique_dates) * (1 - test_ratio))
    cutoff_date = unique_dates[split_idx]

    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    print(f"Cutoff date: {pd.Timestamp(cutoff_date).date()}")

    return train_df, test_df


# ── Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sales.csv")
    df = load_data(csv_path)
    df = clean_data(df)
    print(f"Loaded & cleaned: {len(df):,} rows, {df['store'].nunique()} stores")
    train, test = split_data(df)
