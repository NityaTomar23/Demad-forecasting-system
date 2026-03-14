"""
Demand Forecasting - Streamlit Dashboard.

Provides:
- Historical sales trend visualization
- Model performance comparison (including baseline)
- Actual vs Predicted chart (Mistake #7 fix)
- Interactive sales prediction form
- Feature importance chart
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "models", "predictions.csv")
CACHE_DIR = os.path.join(BASE_DIR, ".cache", "matplotlib")

os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", CACHE_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import build_next_day_features

# ── Helpers ──────────────────────────────────────────────────────────────────


@st.cache_data
def load_sales_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names() -> list[str]:
    return joblib.load(FEATURES_PATH)


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    if os.path.exists(PREDICTIONS_PATH):
        return pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    return None


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Demand Forecasting")
st.sidebar.markdown("Predict future product sales using ML.")

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Model Performance", "Actual vs Predicted", "Predict Sales", "Feature Importance"],
)

# ── Load data ────────────────────────────────────────────────────────────────

try:
    sales_df = load_sales_data()
    model = load_model()
    feature_names = load_feature_names()
    metrics = load_metrics()
    metadata = load_metadata()
    predictions_df = load_predictions()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Could not load data/model: {e}")
    st.info("Run `python data/generate_dataset.py` and `python src/train_model.py` first.")
    st.stop()

# ── Page: Sales Overview ─────────────────────────────────────────────────────

if page == "Sales Overview":
    st.title("Sales Overview")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(sales_df):,}")
    col2.metric("Stores", sales_df["store"].nunique())
    col3.metric("Products", sales_df["product"].nunique())
    col4.metric("Avg Daily Sales", f"{sales_df['sales'].mean():.0f} units")

    st.markdown("---")

    # Filters
    c1, c2 = st.columns(2)
    selected_store = c1.selectbox("Store", ["All"] + sorted(sales_df["store"].unique().tolist()))
    selected_product = c2.selectbox("Product", ["All"] + sorted(sales_df["product"].unique().tolist()))

    filtered = sales_df.copy()
    if selected_store != "All":
        filtered = filtered[filtered["store"] == selected_store]
    if selected_product != "All":
        filtered = filtered[filtered["product"] == selected_product]

    # Aggregate daily sales
    daily = filtered.groupby("date")["sales"].sum().reset_index()

    fig = px.line(
        daily, x="date", y="sales",
        title="Daily Sales Trend",
        labels={"sales": "Total Sales (units)", "date": "Date"},
    )
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    # Monthly breakdown
    monthly = filtered.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly_agg = monthly.groupby("month")["sales"].sum().reset_index()

    fig2 = px.bar(
        monthly_agg, x="month", y="sales",
        title="Monthly Sales",
        labels={"sales": "Total Sales", "month": "Month"},
        color="sales",
        color_continuous_scale="Viridis",
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, width="stretch")

# ── Page: Model Performance ──────────────────────────────────────────────────

elif page == "Model Performance":
    st.title("Model Performance Comparison")

    best_name = metrics.get("best_model", "N/A")
    results = metrics.get("results", {})

    st.success(f"**Best model:** {best_name}")

    # Check baseline improvement
    baseline_rmse = results.get("Baseline (lag-7)", {}).get("RMSE")
    best_rmse = results.get(best_name, {}).get("RMSE")
    if baseline_rmse and best_rmse and baseline_rmse > 0:
        improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
        st.info(f"ML model improves over naive baseline by **{improvement:.1f}%** (RMSE)")

    # Metrics table
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    st.dataframe(results_df.style.highlight_min(axis=0, color="#2ecc71"), width="stretch")

    # Bar chart
    melted = results_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(
        melted, x="Model", y="Value", color="Metric",
        barmode="group", title="RMSE & MAE by Model (including baseline)",
        color_discrete_sequence=["#e74c3c", "#3498db"],
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

# ── Page: Actual vs Predicted ─────────────────────────────────────────────────

elif page == "Actual vs Predicted":
    st.title("Actual vs Predicted Sales")
    st.caption("Test-set predictions from the best model compared to actual values.")

    if predictions_df is not None and not predictions_df.empty:
        # Aggregate by date for clarity
        agg = predictions_df.groupby("date")[["actual", "predicted"]].sum().reset_index()

        # Line chart: actual vs predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agg["date"], y=agg["actual"],
            mode="lines", name="Actual",
            line=dict(color="#3498db", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=agg["date"], y=agg["predicted"],
            mode="lines", name="Predicted",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        ))
        fig.update_layout(
            title="Actual vs Predicted (Test Set)",
            xaxis_title="Date",
            yaxis_title="Total Sales (units)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, width="stretch")

        # Scatter: predicted vs actual
        fig2 = px.scatter(
            predictions_df, x="actual", y="predicted",
            title="Predicted vs Actual (each point = one test sample)",
            labels={"actual": "Actual Sales", "predicted": "Predicted Sales"},
            opacity=0.4,
        )
        # Perfect-prediction line
        max_val = max(predictions_df["actual"].max(), predictions_df["predicted"].max())
        fig2.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect Prediction",
            line=dict(color="#2ecc71", dash="dash"),
        ))
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, width="stretch")

        # Residual distribution
        residuals = predictions_df["actual"] - predictions_df["predicted"]
        fig3 = px.histogram(
            residuals, nbins=50,
            title="Residual Distribution (Actual - Predicted)",
            labels={"value": "Residual (units)", "count": "Frequency"},
            color_discrete_sequence=["#9b59b6"],
        )
        fig3.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig3, width="stretch")

        # Error stats
        st.markdown("### Error Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Residual", f"{residuals.mean():.1f}")
        c2.metric("Std Residual", f"{residuals.std():.1f}")
        c3.metric("Median Abs Error", f"{residuals.abs().median():.1f}")
    else:
        st.warning("No predictions found. Run `python src/train_model.py` first.")

# ── Page: Predict Sales ──────────────────────────────────────────────────────

elif page == "Predict Sales":
    st.title("Predict Next-Day Sales")
    st.caption(
        "This flow derives lag and rolling features from the latest available history. "
        "Only next-day forecasts are supported right now."
    )

    store_options = metadata.get("stores", sorted(sales_df["store"].unique().tolist()))
    product_options = metadata.get("products", sorted(sales_df["product"].unique().tolist()))

    select_col1, select_col2 = st.columns(2)
    selected_store = select_col1.selectbox("Store", store_options)
    selected_product = select_col2.selectbox("Product", product_options)

    series_df = sales_df[
        (sales_df["store"] == selected_store) & (sales_df["product"] == selected_product)
    ].sort_values("date")
    latest_date = series_df["date"].max()
    next_forecast_date = (latest_date + pd.Timedelta(days=1)).date()

    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Latest observed date", str(latest_date.date()))
    info_col2.metric("Forecast date", str(next_forecast_date))

    with st.form("predict_form"):
        cols = st.columns(3)

        input_values = {}
        for i, feat in enumerate(["promotion", "holiday"]):
            col = cols[i % 3]
            default_index = int(feat == "holiday" and next_forecast_date.weekday() >= 5)
            input_values[feat] = col.selectbox(feat, [0, 1], index=default_index)

        submitted = st.form_submit_button("Predict next day", width="stretch")

    if submitted:
        try:
            forecast_timestamp, derived_features = build_next_day_features(
                sales_df=sales_df,
                metadata=metadata,
                store=selected_store,
                product=selected_product,
                promotion=int(input_values["promotion"]),
                holiday=int(input_values["holiday"]),
                forecast_date=next_forecast_date,
            )
            vec = pd.DataFrame(
                [[derived_features[feature] for feature in feature_names]],
                columns=feature_names,
            )
            prediction = model.predict(vec)[0]
            st.markdown("---")
            st.metric("Predicted Sales", f"{prediction:.0f} units")
            with st.expander("Derived model features"):
                st.json(
                    {
                        "forecast_date": forecast_timestamp.date().isoformat(),
                        "store": selected_store,
                        "product": selected_product,
                        "features": derived_features,
                    }
                )
        except ValueError as exc:
            st.error(str(exc))

# ── Page: Feature Importance ─────────────────────────────────────────────────

elif page == "Feature Importance":
    st.title("Feature Importance")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=True)
        )
        fig = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            title="Feature Importance (Best Model)",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Feature importance is not available for the current best model (e.g. Linear Regression).")

    # Feature descriptions
    st.markdown("### Feature Descriptions")
    descriptions = {
        "store": "Encoded store identifier",
        "product": "Encoded product identifier",
        "promotion": "Whether a promotion was active (0/1)",
        "holiday": "Whether the day was a holiday (0/1)",
        "sales_lag_7": "Sales 7 days ago",
        "sales_lag_14": "Sales 14 days ago",
        "sales_lag_28": "Sales 28 days ago",
        "sales_roll_mean_7": "7-day rolling average of sales",
        "sales_roll_std_7": "7-day rolling std of sales",
        "sales_roll_mean_14": "14-day rolling average of sales",
        "sales_roll_std_14": "14-day rolling std of sales",
        "sales_roll_mean_30": "30-day rolling average of sales",
        "sales_roll_std_30": "30-day rolling std of sales",
        "day_of_week": "Day of week (0=Mon, 6=Sun)",
        "month": "Month (1-12)",
        "week_of_year": "ISO week number",
        "is_weekend": "Whether the day is Saturday or Sunday (0/1)",
    }
    desc_df = pd.DataFrame(
        [(f, descriptions.get(f, "--")) for f in feature_names],
        columns=["Feature", "Description"],
    )
    st.table(desc_df)
