# Route Predictor

An AI-powered route optimization system for field sales drivers.
Learns from historical trip data to suggest efficient daily and weekly visit schedules.

---

## What It Does

Field sales drivers waste significant time on inefficient routes — this system addresses that by:

- **Learning historical patterns**: which drivers visit which locations, at what times, under what traffic conditions
- **Optimizing stop order**: using a greedy nearest-neighbor algorithm refined with 2-opt improvement
- **Predicting travel time**: a trained neural network estimates ETA per leg based on distance, time of day, and traffic zone
- **Scoring route confidence**: an XGBoost model rates how well a proposed route matches historically efficient patterns for that driver

---

## Model Architecture

Three components, each with a distinct role:

| Component | Type | Role |
|---|---|---|
| **TSP Solver** | Greedy NN + 2-opt | Finds the shortest stop sequence |
| **Route Ranker** | XGBoost Regressor | Scores efficiency, produces confidence value |
| **ETA Predictor** | MLP Neural Network | Estimates travel time per leg |

**Why this combination?**
TSP handles the combinatorial sequencing problem efficiently without needing training data. XGBoost learns driver-specific and time-specific efficiency patterns that pure distance minimization misses. The MLP captures non-linear relationships between distance, traffic, and actual travel time.

---

## Google API Integration

The system integrates with:
- **Distance Matrix API** — real traffic-aware travel durations
- **Places API** — nearby place discovery and place details
- **Geocoding API** — address → coordinate resolution

**No API key?** The system runs in mock mode — distances use haversine geometry, travel times use traffic-zone heuristics. All endpoints work identically. Add a key to `.env` to switch to live data automatically.

---

## Quick Start

```bash
# clone and install
git clone <repo>
cd route_predictor
pip install -r requirements.txt

# optional: add Google API key
cp .env.example .env
# edit .env and set GOOGLE_MAPS_API_KEY

# generate data and train models
python data/generate_data.py
python scripts/train.py

# start the API
uvicorn api.main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## Docker

```bash
# builds image, generates data, trains models, starts API + dashboard
docker-compose up --build
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

---

## API Reference

### `POST /predict/daily`
```json
{
  "driver_id": "D1",
  "date": "2026-05-20",
  "locations": ["L01", "L05", "L10", "L15"]
}
```
```json
{
  "recommended_route": ["L01", "L10", "L05", "L15"],
  "predicted_time": "3.2 hours",
  "confidence": 0.87,
  "total_distance_km": 18.4,
  "per_stop_eta_min": [12.3, 18.7, 9.1],
  "google_api_used": false
}
```

### `POST /predict/weekly`
```json
{
  "driver_id": "D1",
  "week": "2026-W20"
}
```
```json
{
  "monday": ["L01", "L02", "L03"],
  "tuesday": ["L11", "L12"],
  "wednesday": ["L21", "L22", "L23"],
  "thursday": ["L31", "L32"],
  "friday": ["L41", "L42", "L43"],
  "saturday": ["L04", "L05"],
  "weekly_distance": "187.3km",
  "estimated_hours": "42.1h"
}
```

### `POST /retrain`
Triggers background retraining on current data. Returns immediately.

### `GET /health`
System health, uptime, and cache stats.

### `GET /metrics`
Model status, driver profiles, retrain history.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Feature Engineering

Features used by the route ranker:

| Feature | Description |
|---|---|
| `n_stops` | Number of stops in the route |
| `total_dist_km` | Straight-line distance across all stops |
| `area_spread` | Geographic spread (std dev from centroid) |
| `avg_traffic_score` | Average traffic zone severity |
| `hour_sin / hour_cos` | Time of day (cyclic encoded) |
| `day_sin / day_cos` | Day of week (cyclic encoded) |
| `is_weekend` | Saturday flag |
| `driver_speed` | Driver's historical stops-per-hour |
| `driver_avg_stops` | Driver's typical daily stop count |
| `driver_hist_mult` | Driver's average experienced traffic multiplier |

---

## Project Structure

```
route_predictor/
├── data/
│   └── generate_data.py     synthetic dataset generator
├── model/
│   ├── features.py          feature engineering (shared by train + predict)
│   ├── tsp.py               greedy nearest-neighbor + 2-opt solver
│   ├── ranker.py            XGBoost route efficiency scorer
│   ├── eta_model.py         neural network travel time predictor
│   └── saved/               trained model files (gitignored)
├── api/
│   ├── main.py              FastAPI app + startup
│   ├── google_client.py     Google Maps wrapper with mock fallback
│   ├── cache.py             SQLite cache for API responses
│   └── routes/
│       ├── daily.py         POST /predict/daily
│       ├── weekly.py        POST /predict/weekly
│       └── admin.py         health, metrics, retrain
├── scripts/
│   └── train.py             one-shot training script
├── tests/
│   └── test_api.py          unit tests
├── monitoring/
│   └── dashboard.py         Streamlit monitoring dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
