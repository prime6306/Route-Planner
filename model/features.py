"""
Feature engineering for both training and inference.
Keeps everything in one place so training and prediction
use the exact same logic — a mistake that bites hard otherwise.
"""

import numpy as np
import pandas as pd


TRAFFIC_ZONE_SCORE = {
    "high_traffic":   2,
    "medium_traffic": 1,
    "low_traffic":    0,
}


def cyclic_encode(value, max_value):
    """Encode circular values (hour, day) as sin/cos so 23→0 wraps smoothly."""
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


def haversine_km(lat1, lng1, lat2, lng2):
    """Straight-line distance between two coords. Good enough for feature computation."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def route_spread(lats, lngs):
    """How geographically scattered are the stops? High spread = harder route."""
    if len(lats) < 2:
        return 0.0
    center_lat = np.mean(lats)
    center_lng = np.mean(lngs)
    dists = [haversine_km(lat, lng, center_lat, center_lng) for lat, lng in zip(lats, lngs)]
    return float(np.std(dists))


def total_route_distance(lats, lngs):
    """Sum of straight-line distances between consecutive stops."""
    total = 0.0
    for i in range(len(lats) - 1):
        total += haversine_km(lats[i], lngs[i], lats[i+1], lngs[i+1])
    return round(total, 2)


def build_driver_profiles(trips_df):
    """
    Aggregate per-driver stats from historical data.
    These become features when predicting for a specific driver.
    """
    grp = trips_df.groupby(["driver_id", "date"]).agg(
        total_travel = ("travel_time_min", "sum"),
        total_stops  = ("stop_id", "count"),
        avg_traffic  = ("traffic_mult", "mean"),
    ).reset_index()

    grp["stops_per_hour"] = grp["total_stops"] / (grp["total_travel"] / 60 + 0.1)

    profiles = grp.groupby("driver_id").agg(
        avg_stops_per_hour = ("stops_per_hour", "mean"),
        avg_daily_stops    = ("total_stops", "mean"),
        avg_traffic_mult   = ("avg_traffic", "mean"),
    ).reset_index()

    return profiles


def make_route_features(driver_id, date_str, stop_ids, lats, lngs,
                        traffic_zones, hour, driver_profiles):
    """
    Builds the feature vector for one route prediction request.
    Returns a dict that can be turned into a DataFrame row.
    """
    dow = pd.Timestamp(date_str).dayofweek
    hour_sin, hour_cos = cyclic_encode(hour, 24)
    day_sin,  day_cos  = cyclic_encode(dow, 7)

    avg_traffic_score = np.mean([TRAFFIC_ZONE_SCORE.get(z, 1) for z in traffic_zones])
    spread  = route_spread(lats, lngs)
    dist_km = total_route_distance(lats, lngs)

    # driver history — fall back to global averages if driver is new
    dp = driver_profiles[driver_profiles["driver_id"] == driver_id]
    if len(dp) > 0:
        avg_speed  = float(dp["avg_stops_per_hour"].iloc[0])
        avg_stops  = float(dp["avg_daily_stops"].iloc[0])
        hist_mult  = float(dp["avg_traffic_mult"].iloc[0])
    else:
        avg_speed  = 3.5
        avg_stops  = 6.0
        hist_mult  = 1.1

    return {
        "n_stops":          len(stop_ids),
        "total_dist_km":    dist_km,
        "area_spread":      spread,
        "avg_traffic_score": avg_traffic_score,
        "hour_sin":         hour_sin,
        "hour_cos":         hour_cos,
        "day_sin":          day_sin,
        "day_cos":          day_cos,
        "is_weekend":       int(dow >= 5),
        "driver_speed":     avg_speed,
        "driver_avg_stops": avg_stops,
        "driver_hist_mult": hist_mult,
    }


def build_training_data(trips_df, driver_profiles):
    """
    Group trips by driver+date and compute route-level features + efficiency target.
    Efficiency = stops per hour (simple, interpretable, works well in practice).
    """
    rows = []

    for (driver, date), grp in trips_df.groupby(["driver_id", "date"]):
        grp = grp.sort_values("stop_sequence")

        lats  = grp["lat"].tolist()
        lngs  = grp["lng"].tolist()
        zones = grp["traffic_zone"].tolist()
        hour  = int(grp["visit_hour"].iloc[0])

        feats = make_route_features(
            driver, date,
            grp["stop_id"].tolist(),
            lats, lngs, zones, hour,
            driver_profiles
        )

        total_time = grp["travel_time_min"].sum() + grp["visit_duration_min"].sum()
        efficiency = feats["n_stops"] / (total_time / 60 + 0.1)

        feats["efficiency"] = round(efficiency, 4)
        rows.append(feats)

    df = pd.DataFrame(rows)

    # normalize efficiency to 0-1
    mn, mx = df["efficiency"].min(), df["efficiency"].max()
    df["efficiency_score"] = (df["efficiency"] - mn) / (mx - mn + 1e-9)

    return df


FEATURE_COLS = [
    "n_stops", "total_dist_km", "area_spread", "avg_traffic_score",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_weekend", "driver_speed", "driver_avg_stops", "driver_hist_mult",
]
