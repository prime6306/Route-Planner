"""
Wraps the Google Maps APIs we need: Distance Matrix, Places, and Geocoding.

If GOOGLE_MAPS_API_KEY is set in the environment, this makes real API calls.
If not, it falls back to mock responses that are structurally identical —
so all downstream code works the same either way.

The mock uses haversine distance + traffic zone heuristics to produce
realistic (not random) numbers, which makes the demo look credible.
"""

import os
import math
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
BASE_MAPS  = "https://maps.googleapis.com/maps/api"


def _has_key():
    return bool(API_KEY and API_KEY != "your_api_key_here")


# ─── haversine for mock calculations ────────────────────────────────────────

def _haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _mock_travel_time(dist_km, traffic_condition="normal"):
    # city driving ~25-35 km/h depending on traffic
    speed = {"light": 35, "normal": 28, "heavy": 18}.get(traffic_condition, 28)
    minutes = (dist_km / speed) * 60
    return max(int(minutes), 2)


# ─── distance matrix ────────────────────────────────────────────────────────

async def distance_matrix(origins, destinations, departure_time="now"):
    """
    origins / destinations: list of "lat,lng" strings or place names.
    Returns a matrix of {distance_km, duration_min, status}.
    """
    if _has_key():
        return await _real_distance_matrix(origins, destinations, departure_time)
    return _mock_distance_matrix(origins, destinations)


async def _real_distance_matrix(origins, destinations, departure_time):
    params = {
        "origins":          "|".join(origins),
        "destinations":     "|".join(destinations),
        "mode":             "driving",
        "departure_time":   departure_time,
        "traffic_model":    "best_guess",
        "key":              API_KEY,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{BASE_MAPS}/distancematrix/json", params=params)
        data = resp.json()

    results = []
    for row in data.get("rows", []):
        for elem in row.get("elements", []):
            if elem["status"] == "OK":
                results.append({
                    "distance_km":  round(elem["distance"]["value"] / 1000, 2),
                    "duration_min": round(elem["duration_in_traffic"]["value"] / 60, 1),
                    "status":       "OK",
                })
            else:
                results.append({"distance_km": 0, "duration_min": 0, "status": elem["status"]})
    return results


def _mock_distance_matrix(origins, destinations):
    """Parse 'lat,lng' strings and compute realistic mock distances."""
    results = []
    for orig in origins:
        try:
            olat, olng = map(float, orig.split(","))
        except Exception:
            olat, olng = 26.8467, 80.9462  # default to city center

        for dest in destinations:
            try:
                dlat, dlng = map(float, dest.split(","))
            except Exception:
                dlat, dlng = 26.8467, 80.9462

            dist = _haversine(olat, olng, dlat, dlng)
            mins = _mock_travel_time(dist)
            results.append({
                "distance_km":  round(dist, 2),
                "duration_min": mins,
                "status":       "OK (mock)",
            })
    return results


# ─── geocoding ──────────────────────────────────────────────────────────────

async def geocode(address):
    """Returns (lat, lng) for a place name or address."""
    if _has_key():
        return await _real_geocode(address)
    return _mock_geocode(address)


async def _real_geocode(address):
    params = {"address": address, "key": API_KEY}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{BASE_MAPS}/geocode/json", params=params)
        data = resp.json()

    if data["status"] == "OK":
        loc = data["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None


def _mock_geocode(address):
    """
    Deterministic mock — same address always gets the same coords.
    Uses a hash so it at least spreads locations around rather than stacking them.
    """
    seed = sum(ord(c) for c in address) % 1000
    lat = 26.8467 + (seed % 100 - 50) * 0.002
    lng = 80.9462 + (seed % 70  - 35) * 0.002
    return round(lat, 6), round(lng, 6)


# ─── nearby places ──────────────────────────────────────────────────────────

async def nearby_places(lat, lng, radius=500, place_type="store"):
    if _has_key():
        return await _real_nearby(lat, lng, radius, place_type)
    return _mock_nearby(lat, lng)


async def _real_nearby(lat, lng, radius, place_type):
    params = {
        "location":  f"{lat},{lng}",
        "radius":    radius,
        "type":      place_type,
        "key":       API_KEY,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{BASE_MAPS}/place/nearbysearch/json", params=params)
        data = resp.json()

    places = []
    for r in data.get("results", [])[:5]:
        places.append({
            "name":     r.get("name"),
            "place_id": r.get("place_id"),
            "lat":      r["geometry"]["location"]["lat"],
            "lng":      r["geometry"]["location"]["lng"],
            "rating":   r.get("rating", "N/A"),
        })
    return places


def _mock_nearby(lat, lng):
    return [
        {"name": f"Nearby Store {i}", "place_id": f"mock_place_{i}",
         "lat": lat + i * 0.001, "lng": lng + i * 0.001, "rating": 4.0}
        for i in range(1, 4)
    ]
