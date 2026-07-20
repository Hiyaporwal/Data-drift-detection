"""
app/db.py
----------
PostgreSQL connection + logging for the deployed income model.

Uses a DATABASE_URL environment variable (this is exactly what Railway
gives you when you add a Postgres service to your project -- just copy
its connection string into your environment).

If DATABASE_URL isn't set (e.g., you're testing FastAPI locally without
Postgres running yet), logging is skipped with a warning instead of
crashing the app -- so you can develop the API first, wire up the real
database second.
"""

import os
import json
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL) if DATABASE_URL else None


def init_db():
    """Creates the tables if they don't already exist. Call this once at
    app startup."""
    if engine is None:
        print("[db] DATABASE_URL not set -- skipping DB init (running without logging).")
        return

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                model_name TEXT NOT NULL,
                input_features JSONB NOT NULL,
                prediction TEXT NOT NULL,
                prediction_proba FLOAT NOT NULL
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS drift_checks (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                window_label TEXT NOT NULL,
                method TEXT NOT NULL,
                column_checked TEXT NOT NULL,
                score FLOAT NOT NULL,
                drifted BOOLEAN NOT NULL
            )
        """))
    print("[db] Tables ready: predictions, drift_checks")


def log_prediction(model_name: str, input_features: dict, prediction: str, prediction_proba: float):
    if engine is None:
        print("[db] Skipping prediction log -- no DATABASE_URL set.")
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO predictions (timestamp, model_name, input_features, prediction, prediction_proba)
                    VALUES (:ts, :model_name, CAST(:features AS JSONB), :prediction, :proba)
                """),
                {
                    "ts": datetime.now(timezone.utc),
                    "model_name": model_name,
                    "features": json.dumps(input_features),
                    "prediction": prediction,
                    "proba": prediction_proba,
                },
            )
        print("[db] Prediction logged successfully.")
    except Exception as e:
        print(f"[db] ERROR logging prediction: {e}")
        raise


def log_drift_check(window_label: str, method: str, column_checked: str, score: float, drifted: bool):
    if engine is None:
        print("[db] Skipping drift check log -- no DATABASE_URL set.")
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO drift_checks (timestamp, window_label, method, column_checked, score, drifted)
                    VALUES (:ts, :window_label, :method, :column_checked, :score, :drifted)
                """),
                {
                    "ts": datetime.now(timezone.utc),
                    "window_label": window_label,
                    "method": method,
                    "column_checked": column_checked,
                    "score": score,
                    "drifted": drifted,
                },
            )
        print(f"[db] Drift check ({method}) logged successfully.")
    except Exception as e:
        print(f"[db] ERROR logging drift check: {e}")
        raise


def fetch_recent_predictions(limit: int = 50):
    if engine is None:
        return []
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT :limit"),
            {"limit": limit},
        )
        return [dict(row._mapping) for row in result]


def fetch_recent_drift_checks(limit: int = 50):
    if engine is None:
        return []
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT * FROM drift_checks ORDER BY timestamp DESC LIMIT :limit"),
            {"limit": limit},
        )
        return [dict(row._mapping) for row in result]