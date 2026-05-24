"""
Monitoring dashboard — run with: streamlit run monitoring/dashboard.py
Shows model health, driver stats, prediction history, and cache usage.
Make sure the API is running on localhost:8000 first.
"""

import streamlit as st
import pandas as pd
import requests
import os

API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Route Predictor Dashboard", layout="wide")
st.title("Route Predictor — Model Dashboard")


def fetch_metrics():
    try:
        return requests.get(f"{API_BASE}/metrics", timeout=5).json()
    except Exception:
        return None


def fetch_health():
    try:
        return requests.get(f"{API_BASE}/health", timeout=5).json()
    except Exception:
        return None


# ─── sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("API Connection")
    st.text(f"Endpoint: {API_BASE}")

    health = fetch_health()
    if health:
        st.success("API is up")
        st.metric("Uptime", f"{health['uptime_seconds']}s")
        st.metric("Models loaded", "Yes" if health["models_loaded"] else "No")
    else:
        st.error("Can't reach the API. Is it running?")

    if st.button("Trigger Retrain"):
        try:
            r = requests.post(f"{API_BASE}/retrain", timeout=5)
            st.info(r.json().get("message", "Retrain started"))
        except Exception as e:
            st.error(str(e))


# ─── main content ─────────────────────────────────────────────────────────────

metrics = fetch_metrics()

if metrics is None:
    st.warning("No metrics data — is the API running?")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trips",   metrics.get("total_trips",   "—"))
col2.metric("Drivers",       metrics.get("total_drivers", "—"))
cache = metrics.get("cache", {})
col3.metric("Cache Entries", cache.get("fresh_entries", "—"))
col4.metric("Stale Cache",   cache.get("stale", "—"))

# model file status
st.subheader("Model Files")
model_files = metrics.get("model_files", {})
c1, c2 = st.columns(2)
c1.success("✓ XGBoost Ranker")    if model_files.get("ranker")    else c1.error("✗ XGBoost Ranker missing")
c2.success("✓ ETA Neural Network") if model_files.get("eta_model") else c2.error("✗ ETA model missing")

# driver performance table
st.subheader("Driver Profiles")
driver_data = metrics.get("driver_summary", [])
if driver_data:
    df = pd.DataFrame(driver_data)
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True)
else:
    st.info("No driver profiles loaded yet.")

# retrain history
st.subheader("Recent Retrain History")
history = metrics.get("retrain_history", [])
if history:
    st.table(pd.DataFrame(history))
else:
    st.info("No retrains recorded in this session.")

# quick test prediction
st.subheader("Quick Test — Daily Prediction")
with st.form("test_predict"):
    driver = st.selectbox("Driver", [f"D{i}" for i in range(1, 11)])
    date   = st.date_input("Date")
    locs   = st.text_input("Locations (comma-separated)", "L01,L05,L10,L15")
    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "driver_id": driver,
        "date": str(date),
        "locations": [l.strip() for l in locs.split(",") if l.strip()]
    }
    try:
        resp = requests.post(f"{API_BASE}/predict/daily", json=payload, timeout=10)
        result = resp.json()
        if resp.status_code == 200:
            st.success(f"Route: {' → '.join(result['recommended_route'])}")
            st.metric("Predicted Time",   result["predicted_time"])
            st.metric("Confidence",       result["confidence"])
            st.metric("Total Distance",   f"{result['total_distance_km']} km")
        else:
            st.error(f"Error: {result}")
    except Exception as e:
        st.error(str(e))
