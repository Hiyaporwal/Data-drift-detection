"""
evaluate_dl_model.py
----------------------
Evaluates the trained MLP (deep learning) income predictor under every
drift scenario, exactly like evaluate_model.py does for LR/RF -- so all
three models end up in the SAME results/model_metrics.csv, ready for
analyze_results.py and the model_accuracy_by_scenario plot to pick up
automatically with zero changes.

Run train_dl_model.py first to produce the saved model + preprocessor
this script loads.
"""

import os
import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from drift_injection import (
    inject_covariate_drift,
    inject_label_drift,
    inject_concept_drift,
)

TARGET = "income"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

preprocessor = joblib.load(os.path.join(MODEL_DIR, "mlp_preprocessor.joblib"))
model = keras.models.load_model(os.path.join(MODEL_DIR, "mlp_model.keras"))

results = []

for window in range(1, 6):
    print(f"\n=== Window {window} (MLP) ===")
    df = pd.read_csv(f"data/window_{window}.csv")

    no_drift = df.copy()

    covariate_age, _ = inject_covariate_drift(df, column="age", shift_std=1.5, random_state=window)

    if pd.api.types.is_numeric_dtype(df[TARGET]):
        target_value = 1
    else:
        target_value = ">50K"
    label_income, _ = inject_label_drift(
        df, target_col=TARGET, target_value=target_value, new_positive_rate=0.60, random_state=window
    )

    education_value = "Bachelors" if "Bachelors" in df["education"].unique() else df["education"].mode()[0]
    concept_education, _ = inject_concept_drift(
        df, target_col=TARGET, condition_col="education", condition_value=education_value,
        flip_fraction=0.50, random_state=window,
    )

    scenarios = {
        "no_drift": no_drift,
        "covariate_age": covariate_age,
        "label_income": label_income,
        "concept_education": concept_education,
    }

    for scenario_name, scenario_df in scenarios.items():
        print(f"   -> {scenario_name}")

        X = scenario_df.drop(columns=[TARGET])
        y = scenario_df[TARGET]
        if not pd.api.types.is_numeric_dtype(y):
            y = y.map({"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1})

        X_processed = preprocessor.transform(X)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        y_proba = model.predict(X_processed, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)

        results.append({
            "window": window,
            "scenario": scenario_name,
            "model": "MLP (Deep Learning)",
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
        })

results_df = pd.DataFrame(results)

# --- Merge into the existing model_metrics.csv (don't overwrite LR/RF rows) ---
metrics_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
if os.path.exists(metrics_path):
    existing = pd.read_csv(metrics_path)
    existing = existing[existing["model"] != "MLP (Deep Learning)"]  # avoid duplicates on re-run
    combined = pd.concat([existing, results_df], ignore_index=True)
else:
    combined = results_df

combined.to_csv(metrics_path, index=False)
print(f"\nSaved {len(results_df)} new MLP rows to {metrics_path} (total rows now: {len(combined)})")

print("\n=== MLP Average Accuracy by Scenario ===")
print(results_df.groupby("scenario")[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean().round(4))