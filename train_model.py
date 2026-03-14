"""
Model training module for the Demand Forecasting System.

ANTI-LEAKAGE DESIGN
-------------------
1. The dataset is split chronologically FIRST (time-based, no shuffle).
2. Feature engineering (lag / rolling) runs on the FULL sorted series but
   each feature only looks BACKWARD via shift(), so no future information
   leaks into the training rows.
3. After feature engineering the dataset is re-split at the same cutoff
   date, guaranteeing train rows never see future sales.

BASELINE MODEL
--------------
A naive "yesterday's sales" baseline is evaluated alongside ML models so we
can prove the ML approach actually adds value.

Trains Linear Regression, Random Forest, and LightGBM on engineered features,
evaluates each with RMSE and MAE, then saves the best model.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Allow running from project root or from src/
sys.path.insert(0, os.path.dirname(__file__))
from data_processing import load_data, clean_data, split_data
from feature_engineering import prepare_features

# ── Configuration ───────────────────────────────────────────────────────────

TARGET = "sales"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, ".cache", "matplotlib")

# Features to drop before training (non-predictive)
DROP_COLS = ["date", TARGET]

# ── Metrics ─────────────────────────────────────────────────────────────────


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE and MAE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2)}


def configure_runtime() -> None:
    """Set writable runtime paths for optional plotting dependencies."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", CACHE_DIR)


# ── Baseline ────────────────────────────────────────────────────────────────


def evaluate_baseline(test_df: pd.DataFrame) -> dict:
    """Naive baseline: predict sales = yesterday's sales (sales_lag_7 as proxy).

    If ``sales_lag_7`` is not available, we fall back to the mean of the
    training set, which is still a valid (weak) baseline.
    """
    y_true = test_df[TARGET].values

    if "sales_lag_7" in test_df.columns:
        y_pred = test_df["sales_lag_7"].values
        print("\nBaseline: predict next sales = sales 7 days ago (naive lag-7)")
    else:
        mean_val = y_true.mean()
        y_pred = np.full_like(y_true, mean_val, dtype=float)
        print("\nBaseline: predict next sales = historical mean")

    metrics = evaluate(y_true, y_pred)
    print(f"  RMSE = {metrics['RMSE']:.2f}  |  MAE = {metrics['MAE']:.2f}")
    return metrics


# ── Training ────────────────────────────────────────────────────────────────


def get_models() -> dict:
    """Return a dict of model name -> estimator."""
    configure_runtime()

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=1,
        ),
    }

    try:
        import lightgbm as lgb
    except ImportError:
        print("Warning: LightGBM is not installed. Skipping that model.")
    else:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )

    return models


def train_and_evaluate(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[dict, object, str, list[str], pd.DataFrame]:
    """Train all models, evaluate, and return results.

    Returns
    -------
    results : dict[str, dict]
        Model name -> {RMSE, MAE}.
    best_model : estimator
        The model with the lowest RMSE.
    best_name : str
    feature_names : list[str]
    predictions_df : pd.DataFrame
        Test-set dates with actual and predicted values (best model).
    """
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET].values

    # -- Baseline --
    baseline_metrics = evaluate_baseline(test_df)

    # -- ML Models --
    models = get_models()
    results = {"Baseline (lag-7)": baseline_metrics}
    best_rmse = float("inf")
    best_model = None
    best_name = None
    best_preds = None

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        print(f"  RMSE = {metrics['RMSE']:.2f}  |  MAE = {metrics['MAE']:.2f}")

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model = model
            best_name = name
            best_preds = preds

    print(f"\n* Best model: {best_name} (RMSE={best_rmse:.2f})")

    # Build actual vs predicted DataFrame
    predictions_df = pd.DataFrame({
        "date": test_df["date"].values,
        "actual": y_test,
        "predicted": best_preds,
    })

    return results, best_model, best_name, feature_cols, predictions_df


# ── Save artifacts ──────────────────────────────────────────────────────────


def build_metadata(df: pd.DataFrame) -> dict:
    """Build small reference metadata for inference and UI layers."""
    return {
        "stores": sorted(df["store"].dropna().astype(str).unique().tolist()),
        "products": sorted(df["product"].dropna().astype(str).unique().tolist()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "row_count": int(len(df)),
    }


def save_artifacts(
    model,
    feature_names: list[str],
    results: dict,
    best_name: str,
    metadata: dict,
    predictions_df: pd.DataFrame,
) -> None:
    """Persist the best model, feature names, metrics, metadata, and predictions."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    predictions_path = os.path.join(MODELS_DIR, "predictions.csv")

    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)

    metrics_out = {"best_model": best_name, "results": results}
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    predictions_df.to_csv(predictions_path, index=False)

    print(f"\nModel saved       -> {model_path}")
    print(f"Features saved    -> {features_path}")
    print(f"Metrics saved     -> {metrics_path}")
    print(f"Metadata saved    -> {metadata_path}")
    print(f"Predictions saved -> {predictions_path}")


# ── Entry-point ──────────────────────────────────────────────────────────────

def main():
    configure_runtime()

    # 1. Load & clean
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sales.csv")
    print("=" * 60)
    print("  DEMAND FORECASTING - MODEL TRAINING")
    print("=" * 60)

    df = load_data(csv_path)
    df = clean_data(df)
    metadata = build_metadata(df)

    # 2. Time-based split FIRST (Mistake #1 & #2 prevention)
    #    We split chronologically BEFORE feature engineering.
    #    Then we feature-engineer on the full sorted dataset (all features
    #    only look backward via shift()), and re-split at the same cutoff.
    raw_train, _ = split_data(df)
    cutoff_date = raw_train["date"].max()
    print(f"Anti-leakage: splitting at {cutoff_date.date()} BEFORE feature engineering")

    # 3. Feature engineering on full (sorted) dataset
    #    Lag/rolling features use shift() so they never peek into the future.
    df = prepare_features(df)

    # 4. Re-split at the same cutoff
    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()
    print(f"Post-FE split: Train {len(train_df):,} | Test {len(test_df):,}")

    # 5. Train, evaluate (including baseline), compare
    results, best_model, best_name, feature_cols, predictions_df = train_and_evaluate(
        train_df, test_df
    )

    # 6. Save
    save_artifacts(best_model, feature_cols, results, best_name, metadata, predictions_df)

    # 7. Print summary table
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (includes baseline)")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    summary.index.name = "Model"
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
