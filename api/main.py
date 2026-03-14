"""
Demand Forecasting — Prediction API (FastAPI).

Endpoints
---------
GET  /             → Health check and artifact readiness
GET  /model-info   → Model name, metrics, feature list, and metadata
POST /predict      → Predict sales from engineered feature values
POST /predict-next → Predict the next day for a store/product using history
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
CACHE_DIR = os.path.join(BASE_DIR, ".cache", "matplotlib")

os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", CACHE_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import build_next_day_features, load_sales_history

# ── Load artifacts ──────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _missing_prediction_artifacts() -> list[str]:
    required = [MODEL_PATH, FEATURES_PATH, METRICS_PATH, METADATA_PATH]
    return [path for path in required if not os.path.exists(path)]


def _load_prediction_artifacts() -> dict:
    missing = _missing_prediction_artifacts()
    if missing:
        missing_names = ", ".join(os.path.basename(path) for path in missing)
        raise RuntimeError(
            "Prediction artifacts are missing: "
            f"{missing_names}. Run `python src/train_model.py` first."
        )

    return {
        "model": joblib.load(MODEL_PATH),
        "feature_names": joblib.load(FEATURES_PATH),
        "metrics": _load_json(METRICS_PATH),
        "metadata": _load_json(METADATA_PATH),
    }


def _get_prediction_artifacts() -> dict:
    artifacts = getattr(app.state, "prediction_artifacts", None)
    if artifacts is None:
        artifacts = _load_prediction_artifacts()
        app.state.prediction_artifacts = artifacts
    return artifacts


def _get_sales_history():
    sales_df = getattr(app.state, "sales_history", None)
    if sales_df is None:
        if not os.path.exists(DATA_PATH):
            raise RuntimeError(
                f"Sales data not found at {DATA_PATH}. Run `python data/generate_dataset.py` first."
            )
        sales_df = load_sales_history(DATA_PATH)
        app.state.sales_history = sales_df
    return sales_df


def _require_prediction_artifacts() -> dict:
    try:
        return _get_prediction_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _coerce_feature_value(feature_name: str, value, metadata: dict) -> float:
    if feature_name in {"store", "product"} and isinstance(value, str):
        lookup_key = "stores" if feature_name == "store" else "products"
        values = metadata.get(lookup_key, [])
        if value not in values:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown {feature_name} '{value}'. Expected one of: {values}",
            )
        return float(values.index(value))

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Feature '{feature_name}' must be numeric or a known label.",
        ) from exc


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Demand Forecasting API",
    description="Predict future product sales based on engineered features.",
    version="1.1.0",
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Feature values in the same order as training features."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "store": "Store_A",
                    "product": "Product_1",
                    "promotion": 1,
                    "holiday": 0,
                    "sales_lag_7": 50,
                    "sales_lag_14": 48,
                    "sales_lag_28": 52,
                    "sales_roll_mean_7": 49.5,
                    "sales_roll_std_7": 3.2,
                    "sales_roll_mean_14": 50.1,
                    "sales_roll_std_14": 4.0,
                    "sales_roll_mean_30": 51.0,
                    "sales_roll_std_30": 5.1,
                    "day_of_week": 2,
                    "month": 6,
                    "week_of_year": 25,
                    "is_weekend": 0,
                }
            }
        }
    )

    features: dict[str, float | int | str]


class PredictionResponse(BaseModel):
    predicted_sales: float
    model_used: str


class NextDayPredictionRequest(BaseModel):
    store: str = Field(..., examples=["Store_A"])
    product: str = Field(..., examples=["Product_1"])
    promotion: int = Field(default=0, ge=0, le=1)
    holiday: int | None = Field(default=None, ge=0, le=1)
    forecast_date: date | None = None


class NextDayPredictionResponse(BaseModel):
    forecast_date: date
    store: str
    product: str
    predicted_sales: float
    model_used: str
    derived_features: dict[str, float | int]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/")
def health_check():
    """Health check that also reports model readiness."""
    missing = _missing_prediction_artifacts()
    return {
        "status": "healthy",
        "service": "Demand Forecasting API",
        "model_ready": not missing,
        "missing_artifacts": [os.path.basename(path) for path in missing],
        "data_available": os.path.exists(DATA_PATH),
    }


@app.get("/model-info")
def model_info():
    """Return model metadata and evaluation metrics."""
    artifacts = _require_prediction_artifacts()
    return {
        "best_model": artifacts["metrics"].get("best_model"),
        "results": artifacts["metrics"].get("results"),
        "features": artifacts["feature_names"],
        "stores": artifacts["metadata"].get("stores", []),
        "products": artifacts["metadata"].get("products", []),
        "trained_on": {
            "date_min": artifacts["metadata"].get("date_min"),
            "date_max": artifacts["metadata"].get("date_max"),
            "row_count": artifacts["metadata"].get("row_count"),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict sales given engineered feature values."""
    artifacts = _require_prediction_artifacts()

    try:
        feature_vector = []
        for feature_name in artifacts["feature_names"]:
            if feature_name not in request.features:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Missing feature: '{feature_name}'. "
                        f"Required features: {artifacts['feature_names']}"
                    ),
                )
            feature_vector.append(
                _coerce_feature_value(
                    feature_name,
                    request.features[feature_name],
                    artifacts["metadata"],
                )
            )

        X = pd.DataFrame([feature_vector], columns=artifacts["feature_names"])
        prediction = artifacts["model"].predict(X)[0]

        return PredictionResponse(
            predicted_sales=round(float(prediction), 2),
            model_used=artifacts["metrics"].get("best_model", "unknown"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-next", response_model=NextDayPredictionResponse)
def predict_next(request: NextDayPredictionRequest):
    """Predict the next day using the latest stored history."""
    artifacts = _require_prediction_artifacts()

    try:
        sales_df = _get_sales_history()
        forecast_timestamp, features = build_next_day_features(
            sales_df=sales_df,
            metadata=artifacts["metadata"],
            store=request.store,
            product=request.product,
            promotion=request.promotion,
            holiday=request.holiday,
            forecast_date=request.forecast_date,
        )

        feature_vector = pd.DataFrame(
            [[features[name] for name in artifacts["feature_names"]]],
            columns=artifacts["feature_names"],
        )
        prediction = artifacts["model"].predict(feature_vector)[0]

        return NextDayPredictionResponse(
            forecast_date=forecast_timestamp.date(),
            store=request.store,
            product=request.product,
            predicted_sales=round(float(prediction), 2),
            model_used=artifacts["metrics"].get("best_model", "unknown"),
            derived_features=features,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
