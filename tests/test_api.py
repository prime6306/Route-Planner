"""
Tests for the route predictor API.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from model.tsp import solve, route_length, two_opt, distance_matrix
from model.features import (
    haversine_km, cyclic_encode, route_spread,
    total_route_distance, build_driver_profiles, make_route_features, FEATURE_COLS
)
from api.cache import get as cache_get, set as cache_set, _make_key


# ─── TSP tests ───────────────────────────────────────────────────────────────

class TestTSP:

    def test_single_stop(self):
        ids, km = solve(["A"], [26.84], [80.94])
        assert ids == ["A"]
        assert km == 0.0

    def test_two_stops(self):
        ids, km = solve(["A", "B"], [26.84, 26.85], [80.94, 80.95])
        assert set(ids) == {"A", "B"}
        assert km > 0

    def test_four_stops_all_visited(self):
        lats = [26.84, 26.87, 26.82, 26.86]
        lngs = [80.94, 80.97, 80.92, 80.95]
        ids  = ["A", "B", "C", "D"]
        ordered, km = solve(ids, lats, lngs)
        assert set(ordered) == set(ids)
        assert km > 0

    def test_two_opt_does_not_make_route_longer(self):
        lats = [26.84, 26.87, 26.82, 26.86, 26.83]
        lngs = [80.94, 80.97, 80.92, 80.95, 80.90]
        mat  = distance_matrix(lats, lngs)
        greedy_len = route_length([0,1,2,3,4], mat)
        _, opt_len = two_opt([0,1,2,3,4], mat)
        assert opt_len <= greedy_len + 1e-9


# ─── Feature engineering tests ───────────────────────────────────────────────

class TestFeatures:

    def test_haversine_same_point(self):
        assert haversine_km(26.84, 80.94, 26.84, 80.94) == 0.0

    def test_haversine_reasonable_distance(self):
        # roughly 1 km apart
        d = haversine_km(26.84, 80.94, 26.849, 80.94)
        assert 0.5 < d < 2.0

    def test_cyclic_encode_range(self):
        for h in range(24):
            s, c = cyclic_encode(h, 24)
            assert -1.0 <= s <= 1.0
            assert -1.0 <= c <= 1.0

    def test_route_spread_single(self):
        assert route_spread([26.84], [80.94]) == 0.0

    def test_make_route_features_has_all_cols(self):
        dummy_profiles = pd.DataFrame([{
            "driver_id": "D1",
            "avg_stops_per_hour": 3.5,
            "avg_daily_stops": 6.0,
            "avg_traffic_mult": 1.1,
        }])
        feat = make_route_features(
            "D1", "2026-05-20",
            ["L01", "L02"], [26.84, 26.85], [80.94, 80.95],
            ["medium_traffic", "low_traffic"], 9, dummy_profiles
        )
        for col in FEATURE_COLS:
            assert col in feat, f"Missing feature: {col}"


# ─── Cache tests ──────────────────────────────────────────────────────────────

class TestCache:

    def test_miss_returns_none(self):
        result = cache_get("test_ns", {"key": "definitely_not_cached_xyz"})
        assert result is None

    def test_set_then_get(self):
        payload = {"lat": 26.84, "lng": 80.94}
        cache_set("test_geocode", payload, {"result": "ok"})
        result = cache_get("test_geocode", payload)
        assert result == {"result": "ok"}

    def test_key_is_deterministic(self):
        k1 = _make_key("ns", {"a": 1, "b": 2})
        k2 = _make_key("ns", {"b": 2, "a": 1})
        assert k1 == k2   # sort_keys=True in json.dumps


# ─── Data integrity tests ─────────────────────────────────────────────────────

class TestData:

    @pytest.fixture(scope="class")
    def trips(self):
        if not os.path.exists("data/trips.csv"):
            pytest.skip("Run generate_data.py first")
        return pd.read_csv("data/trips.csv")

    @pytest.fixture(scope="class")
    def locations(self):
        if not os.path.exists("data/locations.csv"):
            pytest.skip("Run generate_data.py first")
        return pd.read_csv("data/locations.csv")

    def test_enough_records(self, trips):
        assert len(trips) >= 1000, f"Only {len(trips)} records — need at least 1000"

    def test_enough_drivers(self, trips):
        assert trips["driver_id"].nunique() >= 10

    def test_enough_locations(self, locations):
        assert len(locations) >= 50

    def test_no_null_coords(self, trips):
        assert trips["lat"].isna().sum() == 0
        assert trips["lng"].isna().sum() == 0

    def test_stop_sequence_starts_at_one(self, trips):
        assert trips["stop_sequence"].min() == 1
