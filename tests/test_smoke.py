import json
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import build_next_day_features, load_sales_history
from api.main import (
    NextDayPredictionRequest,
    PredictionRequest,
    health_check,
    predict,
    predict_next,
)


def load_metadata(sales_df):
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)

    return {
        "stores": sorted(sales_df["store"].dropna().astype(str).unique().tolist()),
    }


class InferenceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sales_df = load_sales_history(DATA_PATH)
        cls.metadata = load_metadata(cls.sales_df)

    def test_build_next_day_features_contains_expected_fields(self):
        forecast_date, features = build_next_day_features(
            sales_df=self.sales_df,
            metadata=self.metadata,
            store=self.sales_df["store"].iloc[0],
        )

        self.assertEqual(features["store"], 0)
        self.assertIn("sales_lag_7", features)
        self.assertIn("sales_roll_mean_30", features)
        self.assertIn("week_of_year", features)

    def test_rejects_non_next_day_requests(self):
        with self.assertRaises(ValueError):
            build_next_day_features(
                sales_df=self.sales_df,
                metadata=self.metadata,
                store=self.sales_df["store"].iloc[0],
                forecast_date="2026-01-02",
            )


class ApiSmokeTests(unittest.TestCase):
    def setUp(self):
        health = health_check()
        if not health["model_ready"]:
            self.skipTest("Model artifacts are missing. Run `python src/train_model.py` first.")

    def test_predict_next_matches_predict_from_derived_features(self):
        next_day = predict_next(
            NextDayPredictionRequest(store=self.sales_df["store"][:1].values[0])
        )
        direct = predict(PredictionRequest(features=next_day.derived_features))

        self.assertAlmostEqual(next_day.predicted_sales, direct.predicted_sales, places=2)


if __name__ == "__main__":
    unittest.main()
