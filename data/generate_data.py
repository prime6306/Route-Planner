"""
Generates synthetic trip history for 10 drivers across 50 locations.
The data is loosely based on a mid-size Indian city layout (Lucknow area).
Patterns are baked in intentionally — morning rush, driver territories,
day-of-week variation — so the model actually has something useful to learn.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

BASE_LAT = 26.8467
BASE_LNG = 80.9462

store_names = [
    "Metro Cash & Carry Hazratganj", "Big Bazaar Gomti Nagar", "Reliance Fresh Aliganj",
    "D-Mart Saharaganj", "Spencer's Indiranagar", "Star Bazaar Rajajipuram",
    "More Supermarket Mahanagar", "V-Mart Chinhat", "Vishal Mega Mart Golf City",
    "Lulu Hypermarket Faizabad Road", "Walmart Best Price Kursi Road",
    "Grofers Hub Vikas Nagar", "Swiggy Dark Store Gomti Extension",
    "Blinkit Hub Hazratganj", "Zepto Store Aliganj", "Quick Commerce Indiranagar",
    "Apollo Pharmacy Rajajipuram", "MedPlus Mahanagar", "Netmeds Hub Chinhat",
    "HealthKart Golf City", "Monginis Bakery Hazratganj",
    "Cake Hub Gomti Nagar", "Bread Talk Aliganj", "French Loaf Saharaganj",
    "Barista Indiranagar", "Cafe Coffee Day Rajajipuram", "Third Wave Coffee Mahanagar",
    "Tea Trunk Chinhat", "Restaurant Depot Golf City", "FoodHub Faizabad Road",
    "Croma Hazratganj", "Vijay Sales Gomti Nagar", "Reliance Digital Aliganj",
    "Samsung Plaza Saharaganj", "Apple Reseller Indiranagar", "Lenovo Hub Rajajipuram",
    "HP Store Mahanagar", "Boat Service Chinhat", "Noise Hub Golf City",
    "Cables & More Faizabad Road", "Fabindia Hazratganj", "Westside Gomti Nagar",
    "Lifestyle Store Aliganj", "Max Fashion Saharaganj", "Zudio Indiranagar",
    "Pantaloons Rajajipuram", "Manyavar Mahanagar", "Raymond Shop Chinhat",
    "Bata Showroom Golf City", "Sports Station Faizabad Road"
]

def assign_traffic_zone(lat, lng):
    # central areas tend to be heavier
    dist_from_center = abs(lat - BASE_LAT) + abs(lng - BASE_LNG)
    if dist_from_center < 0.05:
        return "high_traffic"
    elif dist_from_center < 0.10:
        return "medium_traffic"
    return "low_traffic"


def build_locations():
    rows = []
    for i, name in enumerate(store_names):
        lat = BASE_LAT + np.random.uniform(-0.13, 0.13)
        lng = BASE_LNG + np.random.uniform(-0.13, 0.13)
        rows.append({
            "location_id": f"L{i+1:02d}",
            "name": name,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "traffic_zone": assign_traffic_zone(lat, lng),
            "avg_visit_min": random.randint(12, 40),
        })
    return pd.DataFrame(rows)


# each driver has a home territory — makes the learned patterns more realistic
DRIVER_ZONES = {
    "D1":  [f"L{i:02d}" for i in range(1, 11)],
    "D2":  [f"L{i:02d}" for i in range(11, 21)],
    "D3":  [f"L{i:02d}" for i in range(21, 31)],
    "D4":  [f"L{i:02d}" for i in range(31, 41)],
    "D5":  [f"L{i:02d}" for i in range(41, 51)],
    "D6":  ["L01","L11","L21","L31","L41","L02","L12","L22","L32","L42"],
    "D7":  ["L05","L15","L25","L35","L45","L06","L16","L26","L36","L46"],
    "D8":  ["L07","L17","L27","L37","L47","L08","L18","L28","L38","L48"],
    "D9":  ["L03","L13","L23","L33","L43","L09","L19","L29","L39","L49"],
    "D10": ["L04","L14","L24","L34","L44","L10","L20","L30","L40","L50"],
}


def traffic_multiplier(hour):
    # morning and evening rush are noticeably slower
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        return round(random.uniform(1.3, 1.8), 2)
    elif 13 <= hour <= 14:
        return round(random.uniform(1.1, 1.3), 2)
    return round(random.uniform(0.85, 1.05), 2)


def generate_trips(locations_df):
    loc = {r["location_id"]: r for _, r in locations_df.iterrows()}
    all_ids = list(loc.keys())

    records = []
    start = datetime(2026, 1, 2)
    end   = datetime(2026, 4, 30)

    for driver, home_zone in DRIVER_ZONES.items():
        day = start
        while day <= end:
            if day.weekday() == 6:  # skip sundays
                day += timedelta(days=1)
                continue

            n_stops = random.randint(3, 5) if day.weekday() == 5 else random.randint(5, 8)

            # mostly home zone, occasionally elsewhere
            pool = home_zone.copy()
            random.shuffle(pool)
            stops = pool[:min(n_stops, len(pool))]
            if len(stops) < n_stops:
                extras = random.sample([x for x in all_ids if x not in pool], n_stops - len(stops))
                stops += extras

            clock = datetime(day.year, day.month, day.day, random.randint(8, 9), random.randint(0, 45))

            for seq, stop_id in enumerate(stops, start=1):
                l = loc[stop_id]
                tmult = traffic_multiplier(clock.hour)
                travel_min = round(random.randint(8, 30) * tmult, 1)
                visit_min  = int(l["avg_visit_min"] + random.randint(-4, 8))

                records.append({
                    "driver_id":         driver,
                    "date":              day.strftime("%Y-%m-%d"),
                    "day_of_week":       day.weekday(),
                    "stop_id":           stop_id,
                    "stop_name":         l["name"],
                    "lat":               l["lat"],
                    "lng":               l["lng"],
                    "visit_time":        clock.strftime("%H:%M"),
                    "visit_hour":        clock.hour,
                    "visit_duration_min": visit_min,
                    "travel_time_min":   travel_min,
                    "traffic_mult":      tmult,
                    "traffic_zone":      l["traffic_zone"],
                    "stop_sequence":     seq,
                    "total_stops":       n_stops,
                })

                clock += timedelta(minutes=int(visit_min + travel_min))

            day += timedelta(days=1)

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    locs = build_locations()
    trips = generate_trips(locs)

    locs.to_csv("locations.csv", index=False)
    trips.to_csv("trips.csv", index=False)

    print(f"Locations : {len(locs)}")
    print(f"Trip records : {len(trips)}")
    print(f"Drivers  : {trips['driver_id'].nunique()}")
    print(f"Date range : {trips['date'].min()} → {trips['date'].max()}")
