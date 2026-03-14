# Demand Forecasting System

> Predict future product sales using LightGBM with time-series feature engineering, a FastAPI prediction API, and a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

---

## Architecture

```
Sales Dataset (synthetic, 3 stores x 5 products x 2 years)
   |
   v
Data Processing (load, clean, validate)
   |
   v
Chronological Train/Test Split (anti-leakage)
   |
   v
Feature Engineering
   |-- Lag features (7, 14, 28 days)
   |-- Rolling statistics (mean/std over 7, 14, 30 days)
   |-- Calendar features (day_of_week, month, is_weekend)
   |
   v
Model Training
   |-- Baseline (naive lag-7)
   |-- Linear Regression
   |-- Random Forest
   |-- LightGBM  <-- best model
   |
   v
Evaluation (RMSE / MAE)
   |
   v
Prediction API (FastAPI)  +  Dashboard (Streamlit)
```

---

## Results

| Model | RMSE | MAE |
|-------|------|-----|
| Baseline (lag-7) | 23.80 | 18.46 |
| Linear Regression | 16.79 | 13.00 |
| Random Forest | 16.84 | 13.08 |
| **LightGBM** | **16.44** | **12.82** |

**LightGBM improves over the naive baseline by 31% (RMSE).**

---

## Project Structure

```
demand-forecasting-system/
|
|-- data/
|   |-- generate_dataset.py   # Synthetic data generator
|   +-- sales.csv              # Generated dataset (10,965 rows)
|
|-- src/
|   |-- data_processing.py     # Load, clean, time-based split
|   |-- feature_engineering.py  # Lag, rolling, calendar features
|   |-- train_model.py          # Multi-model training + baseline
|   +-- inference.py            # Next-day feature derivation
|
|-- api/
|   +-- main.py                # FastAPI prediction endpoints
|
|-- dashboard/
|   +-- app.py                 # Streamlit interactive dashboard
|
|-- models/                    # Saved model artifacts (generated)
|-- tests/
|   +-- test_smoke.py          # Smoke tests
|
|-- requirements.txt
|-- Dockerfile
+-- README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate dataset and train models

```bash
python data/generate_dataset.py
python src/train_model.py
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

API endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + artifact readiness |
| GET | `/model-info` | Model metrics, features, metadata |
| POST | `/predict` | Predict from engineered features |
| POST | `/predict-next` | Next-day forecast from store/product |

Example request:

```bash
curl -X POST http://localhost:8000/predict-next \
  -H "Content-Type: application/json" \
  -d '{"store": "Store_A", "product": "Product_1", "promotion": 0}'
```

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard pages:
- **Sales Overview** -- historical trends with store/product filters
- **Model Performance** -- RMSE/MAE comparison (including baseline)
- **Actual vs Predicted** -- overlay chart, scatter plot, residual distribution
- **Predict Sales** -- interactive next-day prediction form
- **Feature Importance** -- LightGBM feature importance chart

### 5. Run tests

```bash
python -B -m unittest discover -s tests
```

### 6. Docker (optional)

```bash
docker build -t demand-forecasting .
docker run -p 8000:8000 demand-forecasting
```

---

## Key Technical Decisions

### Preventing Data Leakage
The dataset is split **chronologically before feature engineering**. All lag and rolling features use `shift()` (backward-looking only), so no future information leaks into training data.

### Baseline Model
A naive lag-7 baseline (predict sales = sales 7 days ago) is evaluated alongside ML models. This proves the ML approach adds real value (31% RMSE improvement).

### Time-based Split (Not Random)
Time-series data requires respecting temporal order. The last 20% of dates form the test set -- matching real-world forecasting where you predict the future, not shuffled data.

### Feature Engineering
| Feature Type | Examples | Purpose |
|---|---|---|
| Lag features | `sales_lag_7`, `sales_lag_14`, `sales_lag_28` | Capture recent sales patterns |
| Rolling stats | `sales_roll_mean_7`, `sales_roll_std_14` | Smooth out noise, capture trends |
| Calendar | `day_of_week`, `month`, `is_weekend` | Capture seasonality |
| Business | `promotion`, `holiday` | External demand drivers |



## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| ML | scikit-learn, LightGBM |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Serialization | Joblib |
| Containerization | Docker |

---

## License

This project is for educational and portfolio purposes.
