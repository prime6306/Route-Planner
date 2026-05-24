"""
POST /predict/weekly
Given a driver and an ISO week string (e.g. "2026-W20"),
returns a suggested daily schedule for the whole week.

Logic:
1. Pull the driver's historically visited locations
2. Cluster them geographically (one cluster per work day)
3. Run TSP within each cluster
4. Estimate weekly total distance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sklearn.cluster import KMeans

from api.main import app_state
from model.tsp import solve as tsp_solve, estimate_drive_time
from model.features import total_route_distance

router = APIRouter()

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]


class WeeklyRequest(BaseModel):
    driver_id: str
    week: str   # ISO format: "2026-W20"


class WeeklyResponse(BaseModel):
    monday:           List[str]
    tuesday:          List[str]
    wednesday:        List[str]
    thursday:         List[str]
    friday:           List[str]
    saturday:         List[str]
    weekly_distance:  str
    estimated_hours:  str


def parse_week(week_str):
    """Turn '2026-W20' into the Monday date of that week."""
    try:
        monday = datetime.strptime(f"{week_str}-1", "%G-W%V-%u")
        return monday
    except Exception:
        raise HTTPException(status_code=400, detail="Week should be 'YYYY-Wnn' format, e.g. '2026-W20'.")


def get_driver_locations(driver_id, trips_df, locations_df):
    """
    Figure out which locations this driver typically visits.
    Use the past 60 days if available, otherwise all history.
    """
    driver_trips = trips_df[trips_df["driver_id"] == driver_id]

    if len(driver_trips) == 0:
        # unknown driver — fall back to a spread of locations
        return locations_df.head(30)

    # weight by visit frequency
    freq = driver_trips["stop_id"].value_counts()
    common = freq.head(30).index.tolist()

    locs = locations_df[locations_df["location_id"].isin(common)].copy()
    locs["visit_count"] = locs["location_id"].map(freq).fillna(0)
    return locs.sort_values("visit_count", ascending=False)


def cluster_by_day(locations, n_days=6):
    """
    K-Means cluster stops geographically so nearby stops end up on the same day.
    Falls back gracefully if there aren't enough locations.
    """
    if len(locations) <= n_days:
        # fewer stops than days — just put one per day
        clusters = {i: [row] for i, (_, row) in enumerate(locations.iterrows())}
        return clusters

    coords = locations[["lat", "lng"]].values
    k = min(n_days, len(locations))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    clusters = {}
    for day_idx in range(k):
        mask = labels == day_idx
        clusters[day_idx] = locations[mask].to_dict("records")

    return clusters


@router.post("/weekly", response_model=WeeklyResponse)
async def predict_weekly(req: WeeklyRequest):
    monday = parse_week(req.week)

    trips_df     = app_state["trips"]
    locations_df = app_state["locations"]

    driver_locs = get_driver_locations(req.driver_id, trips_df, locations_df)

    if len(driver_locs) == 0:
        raise HTTPException(status_code=404, detail=f"No historical data for driver {req.driver_id}.")

    clusters = cluster_by_day(driver_locs, n_days=6)

    schedule    = {d: [] for d in DAYS}
    total_km    = 0.0
    total_hours = 0.0

    for day_idx, day_name in enumerate(DAYS):
        if day_idx not in clusters:
            continue

        stops = clusters[day_idx]
        if not stops:
            continue

        ids  = [s["location_id"] for s in stops]
        lats = [s["lat"]         for s in stops]
        lngs = [s["lng"]         for s in stops]

        ordered_ids, day_km = tsp_solve(ids, lats, lngs)
        schedule[day_name] = ordered_ids

        total_km    += day_km
        total_hours += estimate_drive_time(day_km)

        # add visit time estimate (20 min avg per stop)
        total_hours += len(ordered_ids) * 20 / 60

    return WeeklyResponse(
        monday    = schedule["monday"],
        tuesday   = schedule["tuesday"],
        wednesday = schedule["wednesday"],
        thursday  = schedule["thursday"],
        friday    = schedule["friday"],
        saturday  = schedule["saturday"],
        weekly_distance = f"{total_km:.1f}km",
        estimated_hours = f"{total_hours:.1f}h",
    )
