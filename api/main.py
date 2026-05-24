"""
Route Predictor API
Handles startup, model loading, and route registration.
"""

import os
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model import ranker, eta_model
from model.features import build_driver_profiles

# global state — loaded once at startup, reused across requests
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    app_state["ranker"]   = ranker.load()
    app_state["eta"]      = eta_model.load()
    app_state["locations"] = pd.read_csv("data/locations.csv")

    trips = pd.read_csv("data/trips.csv")
    app_state["driver_profiles"] = build_driver_profiles(trips)
    app_state["trips"] = trips

    print("Ready.")
    yield
    app_state.clear()


app = FastAPI(
    title="Route Predictor API",
    description="AI-powered route optimization for field sales drivers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# register routes
from api.routes import daily, weekly, admin
app.include_router(daily.router,   prefix="/predict")
app.include_router(weekly.router,  prefix="/predict")
app.include_router(admin.router)


@app.get("/")
def root():
    return {
        "service": "Route Predictor",
        "status":  "running",
        "docs":    "/docs",
    }
