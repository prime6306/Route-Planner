"""
Small neural network for predicting travel time per stop.
Uses sklearn's MLPRegressor — clean, fast to train, no GPU needed.

Input features: distance, time of day (sin/cos), day of week (sin/cos), traffic zone.
Output: estimated travel time in minutes.

Could swap this for an LSTM to capture sequential dependencies across the
full day's route, but for leg-by-leg ETA the MLP gets within ~2-3 minutes
which is more than good enough for scheduling purposes.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from model.features import haversine_km, cyclic_encode, TRAFFIC_ZONE_SCORE

MODEL_PATH = "model/saved/eta_model.joblib"

ETA_FEATURES = [
    "dist_km", "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_weekend", "traffic_zone_score", "traffic_mult_est",
]


def build_eta_samples(trips_df):
    """
    Each row = one trip leg (travel from prev stop to current stop).
    We approximate the leg distance using haversine on consecutive stops.
    """
    samples = []
    for (driver, date), grp in trips_df.groupby(["driver_id", "date"]):
        grp = grp.sort_values("stop_sequence").reset_index(drop=True)

        for i in range(1, len(grp)):
            prev = grp.iloc[i - 1]
            curr = grp.iloc[i]

            dist = haversine_km(prev["lat"], prev["lng"], curr["lat"], curr["lng"])
            hour = int(curr["visit_hour"])
            dow  = int(curr["day_of_week"])
            h_sin, h_cos = cyclic_encode(hour, 24)
            d_sin, d_cos = cyclic_encode(dow, 7)

            samples.append({
                "dist_km":            round(dist, 3),
                "hour_sin":           h_sin,
                "hour_cos":           h_cos,
                "day_sin":            d_sin,
                "day_cos":            d_cos,
                "is_weekend":         int(dow >= 5),
                "traffic_zone_score": TRAFFIC_ZONE_SCORE.get(curr["traffic_zone"], 1),
                "traffic_mult_est":   float(curr["traffic_mult"]),
                "travel_time_min":    float(curr["travel_time_min"]),
            })

    return pd.DataFrame(samples)


def train(trips_df):
    samples = build_eta_samples(trips_df)

    X = samples[ETA_FEATURES]
    y = samples["travel_time_min"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=7)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ))
    ])

    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_val, model.predict(X_val))
    print(f"  ETA model val MAE: {mae:.2f} minutes")

    os.makedirs("model/saved", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  Saved → {MODEL_PATH}")

    return model


def load():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No ETA model found at {MODEL_PATH}. Run scripts/train.py first.")
    return joblib.load(MODEL_PATH)


def predict_leg(model, dist_km, hour, day_of_week, traffic_zone, traffic_mult=1.1):
    """Predict travel time for a single leg."""
    h_sin, h_cos = cyclic_encode(hour, 24)
    d_sin, d_cos = cyclic_encode(day_of_week, 7)

    row = pd.DataFrame([{
        "dist_km":            dist_km,
        "hour_sin":           h_sin,
        "hour_cos":           h_cos,
        "day_sin":            d_sin,
        "day_cos":            d_cos,
        "is_weekend":         int(day_of_week >= 5),
        "traffic_zone_score": TRAFFIC_ZONE_SCORE.get(traffic_zone, 1),
        "traffic_mult_est":   traffic_mult,
    }])

    pred = float(model.predict(row)[0])
    return round(max(pred, 1.0), 1)  # at least 1 minute


def predict_route_eta(model, ordered_lats, ordered_lngs, ordered_zones,
                      start_hour, day_of_week, avg_visit_min=20):
    """
    Predict total time for a full route: sum of leg travel times + visit durations.
    Returns (total_hours, per_stop_minutes).
    """
    from model.features import haversine_km

    leg_times = []
    current_hour = start_hour

    for i in range(1, len(ordered_lats)):
        dist = haversine_km(
            ordered_lats[i-1], ordered_lngs[i-1],
            ordered_lats[i],   ordered_lngs[i]
        )
        t = predict_leg(model, dist, current_hour, day_of_week, ordered_zones[i])
        leg_times.append(t)
        current_hour = min(current_hour + int((t + avg_visit_min) / 60), 22)

    total_travel = sum(leg_times)
    total_visits = avg_visit_min * len(ordered_lats)
    total_hours  = round((total_travel + total_visits) / 60, 2)

    return total_hours, leg_times
