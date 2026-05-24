"""
Train all models from scratch.
Run this once after generating data, and again whenever you want to retrain.

Usage:
    python -m scripts.train
    # or from project root:
    python scripts/train.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from model.features import build_driver_profiles, build_training_data
from model import ranker, eta_model


def main():
    print("=" * 50)
    print("Route Predictor — Model Training")
    print("=" * 50)

    if not os.path.exists("data/trips.csv"):
        print("\nNo trip data found. Generating it first...")
        os.system("python data/generate_data.py")

    print("\nLoading data...")
    trips = pd.read_csv("data/trips.csv")
    print(f"  {len(trips)} trip records, {trips['driver_id'].nunique()} drivers")

    print("\nBuilding driver profiles...")
    profiles = build_driver_profiles(trips)

    print("\nEngineering route-level features...")
    training_data = build_training_data(trips, profiles)
    print(f"  {len(training_data)} route records for training")

    print("\nTraining XGBoost route ranker...")
    t0 = time.time()
    ranker.train(training_data)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nTraining ETA neural network...")
    t0 = time.time()
    eta_model.train(trips)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n✓ All models saved to model/saved/")
    print("You can now start the API with: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
