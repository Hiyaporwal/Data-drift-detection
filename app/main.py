"""
app/main.py
------------
The deployed income-prediction service.

Endpoints:
    GET  /                   -> health check
    POST /predict            -> predict income class for one record, logs to Postgres
    POST /check-drift        -> compare a batch of new records against the
                                 reference window using all 4 detectors,
                                 logs results to Postgres
    GET  /recent-predictions -> last 50 logged predictions (for the dashboard)
    GET  /recent-drift-checks-> last 50 logged drift checks (for the dashboard)

Run locally with:
    uvicorn app.main:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive API docs.
"""

import os
import sys
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Make src/ importable so we can reuse drift_detectors.py without duplicating logic
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from drift_detectors import ks_test_drift, calculate_psi, calculate_kl_divergence, classifier_drift_detector  # noqa: E402

from app import db

MODEL_DIR = "models"
REFERENCE_WINDOW_PATH = "data/window_0.csv"
DEFAULT_MODEL_NAME = "random_forest"  # which trained model /predict uses by default

app = FastAPI(title="Income Drift Detection API", version="1.0")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class IncomeRecord(BaseModel):
    """One row of Adult Income features (everything except the target)."""
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        # Lets us accept the real dataset's dotted column names via alias
        populate_by_name = True


def record_to_dataframe(record: IncomeRecord) -> pd.DataFrame:
    """Converts the API's underscore field names back to the dataset's
    original dotted column names, since that's what the trained pipeline
    expects (it was fit on columns like 'education.num', 'capital.gain')."""
    row = {
        "age": record.age,
        "workclass": record.workclass,
        "fnlwgt": record.fnlwgt,
        "education": record.education,
        "education.num": record.education_num,
        "marital.status": record.marital_status,
        "occupation": record.occupation,
        "relationship": record.relationship,
        "race": record.race,
        "sex": record.sex,
        "capital.gain": record.capital_gain,
        "capital.loss": record.capital_loss,
        "hours.per.week": record.hours_per_week,
        "native.country": record.native_country,
    }
    return pd.DataFrame([row])


class DriftCheckRequest(BaseModel):
    records: List[IncomeRecord]
    window_label: Optional[str] = "unlabeled_batch"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    db.init_db()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "service": "income-drift-detection-api"}


@app.post("/predict")
def predict(record: IncomeRecord, model_name: str = DEFAULT_MODEL_NAME):
    """
    Predicts income class (<=50K / >50K) for one record using the trained
    pipeline (already includes preprocessing), and logs the prediction.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found at {model_path}")

    model = joblib.load(model_path)
    X = record_to_dataframe(record)

    pred_label = model.predict(X)[0]
    pred_proba = float(model.predict_proba(X)[0][1])
    prediction_str = ">50K" if pred_label == 1 else "<=50K"

    db.log_prediction(
        model_name=model_name,
        input_features=record.dict(),
        prediction=prediction_str,
        prediction_proba=pred_proba,
    )

    return {
        "model_used": model_name,
        "prediction": prediction_str,
        "probability_over_50k": round(pred_proba, 4),
    }


@app.post("/check-drift")
def check_drift(request: DriftCheckRequest):
    """
    Compares a batch of new records against the reference window (window_0)
    using all 4 detection methods, on the same columns your research
    pipeline uses (age numeric, education categorical, plus the
    classifier-based multi-feature check). Logs every result to Postgres.
    """
    if not os.path.exists(REFERENCE_WINDOW_PATH):
        raise HTTPException(status_code=500, detail=f"Reference window not found at {REFERENCE_WINDOW_PATH}")

    reference_df = pd.read_csv(REFERENCE_WINDOW_PATH)
    comparison_df = pd.concat([record_to_dataframe(r) for r in request.records], ignore_index=True)

    results = []

    # Numeric: age
    for result in [
        ks_test_drift(reference_df["age"], comparison_df["age"]),
        calculate_psi(reference_df["age"], comparison_df["age"]),
        calculate_kl_divergence(reference_df["age"], comparison_df["age"]),
    ]:
        results.append({**result, "column_checked": "age"})

    # Categorical: education
    for result in [
        calculate_psi(reference_df["education"], comparison_df["education"]),
        calculate_kl_divergence(reference_df["education"], comparison_df["education"]),
    ]:
        results.append({**result, "column_checked": "education"})

    # Classifier-based, multi-feature
    numeric_cols = ["age", "hours.per.week", "capital.gain", "capital.loss"]
    for model_type in ["random_forest", "logistic_regression"]:
        clf_result = classifier_drift_detector(reference_df, comparison_df, numeric_cols, model_type=model_type)
        results.append({**clf_result, "column_checked": "multi_feature"})

    # Log every result to Postgres
    for r in results:
        db.log_drift_check(
            window_label=request.window_label,
            method=r["method"],
            column_checked=r["column_checked"],
            score=r["score"],
            drifted=r["drifted"],
        )

    return {
        "window_label": request.window_label,
        "n_records": len(request.records),
        "results": results,
    }


@app.get("/recent-predictions")
def recent_predictions(limit: int = 50):
    return db.fetch_recent_predictions(limit=limit)


@app.get("/recent-drift-checks")
def recent_drift_checks(limit: int = 50):
    return db.fetch_recent_drift_checks(limit=limit)