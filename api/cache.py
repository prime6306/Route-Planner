"""
Simple SQLite cache for Google API responses.
Saves money and makes repeated requests instant.
Cache entries expire after 24 hours since traffic conditions change.
"""

import json
import sqlite3
import hashlib
import time
import os

DB_PATH = "data/api_cache.db"
TTL_SECONDS = 86400  # 24 hours


def _connect():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key       TEXT PRIMARY KEY,
            value     TEXT NOT NULL,
            saved_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _make_key(namespace, payload):
    raw = f"{namespace}:{json.dumps(payload, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get(namespace, payload):
    key = _make_key(namespace, payload)
    conn = _connect()
    row = conn.execute(
        "SELECT value, saved_at FROM cache WHERE key = ?", (key,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    value, saved_at = row
    if time.time() - saved_at > TTL_SECONDS:
        return None  # stale

    return json.loads(value)


def set(namespace, payload, result):
    key = _make_key(namespace, payload)
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, value, saved_at) VALUES (?, ?, ?)",
        (key, json.dumps(result), time.time())
    )
    conn.commit()
    conn.close()


def clear_expired():
    conn = _connect()
    deleted = conn.execute(
        "DELETE FROM cache WHERE saved_at < ?", (time.time() - TTL_SECONDS,)
    ).rowcount
    conn.commit()
    conn.close()
    return deleted


def stats():
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    fresh = conn.execute(
        "SELECT COUNT(*) FROM cache WHERE saved_at >= ?", (time.time() - TTL_SECONDS,)
    ).fetchone()[0]
    conn.close()
    return {"total_entries": total, "fresh_entries": fresh, "stale": total - fresh}
