"""
POST /predict/daily
Takes a driver, date, and list of locations.
Returns an optimized stop order, estimated total time, and a confidence score.
"""

import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from api.main import app_state
from api import cache as api_cache
from api import google_client
from model.tsp import solve as tsp_solve
from model.features import make_route_features, TRAFFIC_ZONE_SCORE
from model import ranker, eta_model

router = APIRouter()


class DailyRequest(BaseModel):
    driver_id: str
    date: str          # e.g. "2026-05-20"
    locations: List[str]


class DailyResponse(BaseModel):
    recommended_route: List[str]
    predicted_time: str
    confidence: float
    total_distance_km: float
    per_stop_eta_min: List[float]
    google_api_used: bool


async def resolve_location(loc_name, locations_df):
    """
    Try to match by location_id or name first.
    Fall back to geocoding if it looks like a real address.
    """
    row = locations_df[
        (locations_df["location_id"] == loc_name) |
        (locations_df["name"].str.lower() == loc_name.lower())
    ]

    if len(row) > 0:
        r = row.iloc[0]
        return {
            "id":    loc_name,
            "lat":   r["lat"],
            "lng":   r["lng"],
            "zone":  r["traffic_zone"],
            "visit_min": int(r.get("avg_visit_min", 20)),
        }

    # not in our db — geocode it
    cached = api_cache.get("geocode", {"address": loc_name})
    if cached:
        lat, lng = cached["lat"], cached["lng"]
    else:
        lat, lng = await google_client.geocode(loc_name)
        if lat:
            api_cache.set("geocode", {"address": loc_name}, {"lat": lat, "lng": lng})

    return {
        "id":    loc_name,
        "lat":   lat or 26.8467,
        "lng":   lng or 80.9462,
        "zone":  "medium_traffic",
        "visit_min": 20,
    }


@router.post("/daily", response_model=DailyResponse)
async def predict_daily(req: DailyRequest):
    if len(req.locations) < 1:
        raise HTTPException(status_code=400, detail="Need at least one location.")

    # parse date
    try:
        date_obj = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date should be YYYY-MM-DD format.")

    locations_df = app_state["locations"]
    driver_profiles = app_state["driver_profiles"]

    # resolve all locations to lat/lng
    resolved = []
    for loc in req.locations:
        info = await resolve_location(loc, locations_df)
        resolved.append(info)

    ids   = [r["id"]   for r in resolved]
    lats  = [r["lat"]  for r in resolved]
    lngs  = [r["lng"]  for r in resolved]
    zones = [r["zone"] for r in resolved]
    avg_visit = int(sum(r["visit_min"] for r in resolved) / len(resolved))

    # optimise route order with TSP
    ordered_ids, total_km = tsp_solve(ids, lats, lngs)

    # reorder lats/lngs to match optimised order
    order_map   = {oid: i for i, oid in enumerate(ids)}
    ordered_idx = [order_map[oid] for oid in ordered_ids]
    o_lats  = [lats[i]  for i in ordered_idx]
    o_lngs  = [lngs[i]  for i in ordered_idx]
    o_zones = [zones[i] for i in ordered_idx]

    # ETA prediction
    start_hour = 9  # assume 9 AM start if not specified
    total_hours, leg_times = eta_model.predict_route_eta(
        app_state["eta"], o_lats, o_lngs, o_zones,
        start_hour=start_hour, day_of_week=date_obj.weekday(),
        avg_visit_min=avg_visit,
    )

    # route confidence from XGBoost ranker
    feat = make_route_features(
        req.driver_id, req.date,
        ordered_ids, o_lats, o_lngs, o_zones,
        start_hour, driver_profiles,
    )
    confidence = ranker.score_route(app_state["ranker"], feat)

    return DailyResponse(
        recommended_route  = ordered_ids,
        predicted_time     = f"{total_hours:.1f} hours",
        confidence         = confidence,
        total_distance_km  = total_km,
        per_stop_eta_min   = leg_times,
        google_api_used    = google_client._has_key(),
    )
