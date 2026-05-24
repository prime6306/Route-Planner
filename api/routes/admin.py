"""
Admin endpoints: health check, model retraining, and cache/metrics info.
"""

import time
import os
import subprocess
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from api import cache as api_cache
from api.main import app_state
from model import ranker, eta_model

router = APIRouter()

startup_time = time.time()
retrain_log  = []   # keeps last few retrain runs in memory


@router.get("/health")
def health_check():
    return {
        "status":         "ok",
        "uptime_seconds": round(time.time() - startup_time),
        "models_loaded":  "ranker" in app_state and "eta" in app_state,
        "cache_stats":    api_cache.stats(),
    }


@router.get("/metrics")
def metrics():
    """Basic monitoring data — used by the dashboard."""
    trips = app_state.get("trips")
    profiles = app_state.get("driver_profiles")

    stats = {
        "total_trips":   len(trips) if trips is not None else 0,
        "total_drivers": trips["driver_id"].nunique() if trips is not None else 0,
        "model_files": {
            "ranker":    os.path.exists("model/saved/xgb_ranker.joblib"),
            "eta_model": os.path.exists("model/saved/eta_model.joblib"),
        },
        "cache":         api_cache.stats(),
        "retrain_history": retrain_log[-5:],
    }

    if profiles is not None:
        stats["driver_summary"] = profiles.to_dict("records")

    return stats


class RetrainResponse(BaseModel):
    status:  str
    message: str


def _run_retrain():
    """Runs in background so the API doesn't block."""
    start = time.time()
    try:
        import pandas as pd
        from model.features import build_driver_profiles, build_training_data

        trips = pd.read_csv("data/trips.csv")
        profiles = build_driver_profiles(trips)
        training = build_training_data(trips, profiles)

        new_ranker = ranker.train(training)
        new_eta    = eta_model.train(trips)

        # hot-swap models without restarting
        app_state["ranker"]          = new_ranker
        app_state["eta"]             = new_eta
        app_state["driver_profiles"] = profiles
        app_state["trips"]           = trips

        elapsed = round(time.time() - start, 1)
        retrain_log.append({"status": "success", "duration_sec": elapsed, "at": time.strftime("%Y-%m-%d %H:%M")})

    except Exception as e:
        retrain_log.append({"status": "failed", "error": str(e), "at": time.strftime("%Y-%m-%d %H:%M")})


@router.post("/retrain", response_model=RetrainResponse)
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_retrain)
    return RetrainResponse(
        status  = "started",
        message = "Retraining running in background. Check /metrics for progress."
    )


@router.delete("/cache")
def clear_cache():
    removed = api_cache.clear_expired()
    return {"removed_stale_entries": removed}
