import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from drift_injection import (
    inject_covariate_drift,
    inject_label_drift,
    inject_concept_drift,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TARGET = "income"

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
}

results = []

# --------------------------------------------------
# Evaluate all windows
# --------------------------------------------------

for window in range(1, 7):

    print(f"\n=== Window {window} ===")

    df = pd.read_csv(f"data/window_{window}.csv")

    # --------------------------------------------------
    # Create drift scenarios
    # --------------------------------------------------

    no_drift = df.copy()

    covariate_age, _ = inject_covariate_drift(
        df,
        column="age",
        shift_std=1.5,
        random_state=window,
    )

    # IMPORTANT:
    # If income is encoded as 0/1 use target_value=1
    # If income contains ">50K"/"<=50K" keep target_value=">50K"

    if pd.api.types.is_numeric_dtype(df[TARGET]):
        target_value = 1
    else:
        target_value = ">50K"

    label_income, _ = inject_label_drift(
        df,
        target_col=TARGET,
        target_value=target_value,
        new_positive_rate=0.60,
        random_state=window,
    )

    # Choose an education category that actually exists
    if "Bachelors" in df["education"].unique():
        education_value = "Bachelors"
    else:
        education_value = df["education"].mode()[0]

    concept_education, _ = inject_concept_drift(
        df,
        target_col=TARGET,
        condition_col="education",
        condition_value=education_value,
        flip_fraction=0.50,
        random_state=window,
    )

    scenarios = {
        "no_drift": no_drift,
        "covariate_age": covariate_age,
        "label_income": label_income,
        "concept_education": concept_education,
    }

    # --------------------------------------------------
    # Evaluate every scenario
    # --------------------------------------------------

    for scenario_name, scenario_df in scenarios.items():

        print(f"   -> {scenario_name}")

        X = scenario_df.drop(columns=[TARGET])
        y = scenario_df[TARGET]

        if y.dtype == object:
            y = y.map({
                "<=50K": 0,
                ">50K": 1,
                "<=50K.": 0,
                ">50K.": 1,
            })

        for model_name, filename in MODELS.items():

            model = joblib.load(os.path.join(MODEL_DIR, filename))

            pred = model.predict(X)
            prob = model.predict_proba(X)[:, 1]

            results.append({
                "window": window,
                "scenario": scenario_name,
                "model": model_name,
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "roc_auc": roc_auc_score(y, prob),
            })

# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_metrics.csv",
    index=False,
)

print(f"\nSaved {len(results_df)} rows to results/model_metrics.csv")

print("\n=== Average Accuracy ===")
print(
    results_df.groupby(["model", "scenario"])[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    .mean()
    .round(4)
)