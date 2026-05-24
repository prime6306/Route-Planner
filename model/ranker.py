"""
XGBoost model that scores how efficient a route configuration is,
based on learned patterns from historical trips.

The score (0-1) becomes the "confidence" value in API responses.
Higher = the model thinks this route matches historically efficient patterns
for this driver, time of day, and geography.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from model.features import FEATURE_COLS

MODEL_PATH = "model/saved/xgb_ranker.joblib"


def train(training_df):
    """
    Trains on route-level features. Target is normalized efficiency score.
    XGBoost handles the non-linear interactions between time, geography,
    and driver behavior pretty well without much tuning.
    """
    X = training_df[FEATURE_COLS]
    y = training_df["efficiency_score"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=7)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"  Ranker val MAE: {mae:.4f}")

    os.makedirs("model/saved", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  Saved → {MODEL_PATH}")

    return model


def load():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained ranker found at {MODEL_PATH}. Run scripts/train.py first.")
    return joblib.load(MODEL_PATH)


def score_route(model, feature_dict):
    """
    Score a single route. Returns a float between 0 and 1.
    Clipped just in case the model extrapolates outside training range.
    """
    row = pd.DataFrame([feature_dict])[FEATURE_COLS]
    raw = float(model.predict(row)[0])
    return round(float(np.clip(raw, 0.0, 1.0)), 3)


def feature_importance(model):
    """Useful for README / evaluation — shows what the model actually learned."""
    scores = model.get_booster().get_fscore()
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked
