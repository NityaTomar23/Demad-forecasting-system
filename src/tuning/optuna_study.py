"""
Hyperparameter tuning script using Optuna for the LightGBM model.
"""

import os
import sys
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# Add src to path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from data_processing import load_data, clean_data, split_data
from feature_engineering import prepare_features

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM must be installed to run this tuning script.")
    sys.exit(1)

# -- Config --
TARGET = "sales"
DROP_COLS = ["date", TARGET]

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

def run_tuning():
    csv_path = os.path.join(BASE_DIR, "data", "sales.csv")
    if not os.path.exists(csv_path):
        print(f"Data not found at {csv_path}. Run data preparation first.")
        sys.exit(1)

    print("Loading data for Optuna tuning...")
    df = load_data(csv_path)
    df = clean_data(df)

    raw_train, _ = split_data(df)
    cutoff_date = raw_train["date"].max()
    
    df = prepare_features(df)
    
    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()
    
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET].values

    study = optuna.create_study(direction="minimize", study_name="lgbm_tuning")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    print("\n" + "="*50)
    print("Optimization finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (RMSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)
    
if __name__ == "__main__":
    run_tuning()
