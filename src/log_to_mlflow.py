"""
log_to_mlflow.py
------------------
Phase 5: EXPERIMENT TRACKING.

Rather than rewriting run_experiment.py / train_model.py / analyze_results.py
to log inline (which would mean re-running everything), this script logs
your ALREADY-GENERATED results into MLflow as one clean, reproducible run.

Run this AFTER you've run, in order:
    python src/run_experiment.py
    python src/train_model.py
    python src/evaluate_model.py
    python src/analyze_results.py

It logs:
    - PARAMS: dataset, number of windows, drift types, methods, models used
    - METRICS: detection performance per method, model accuracy per scenario,
      the concept-drift blind-spot numbers
    - ARTIFACTS: every results CSV, every plot, and the trained model files

To view everything in MLflow's UI afterward, run:
    mlflow ui
then open http://localhost:5000 in your browser.
"""

import os
import mlflow
import pandas as pd

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
MODEL_DIR = "models"

EXPERIMENT_NAME = "data_drift_detection"


def log_params_and_config(run):
    """Static config describing THIS experiment design -- useful for
    comparing runs later if you change drift parameters and re-run."""
    mlflow.log_param("dataset", "UCI Adult Income (Kaggle: uciml/adult-census-income)")
    mlflow.log_param("n_windows", 6)
    mlflow.log_param("reference_window", "window_0")
    mlflow.log_param("drift_types", "covariate_age, label_income, concept_education")
    mlflow.log_param("covariate_drift_column", "age")
    mlflow.log_param("covariate_drift_shift_std", 1.5)
    mlflow.log_param("label_drift_target_col", "income")
    mlflow.log_param("label_drift_new_positive_rate", 0.6)
    mlflow.log_param("concept_drift_condition_col", "education")
    mlflow.log_param("concept_drift_condition_value", "Bachelors")
    mlflow.log_param("concept_drift_flip_fraction", 0.5)
    mlflow.log_param("detection_methods", "KS, PSI, KL, Classifier_RF, Classifier_LR")
    mlflow.log_param("prediction_models", "LogisticRegression, RandomForestClassifier")


def log_detection_performance_metrics():
    """One metric per method: overall_accuracy, precision, recall, FPR."""
    path = f"{RESULTS_DIR}/detection_performance_summary.csv"
    if not os.path.exists(path):
        print(f"Skipping detection performance metrics -- {path} not found.")
        return

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        method = row["method"]
        mlflow.log_metric(f"{method}_precision", row["precision"])
        mlflow.log_metric(f"{method}_recall_TPR", row["recall_TPR"])
        mlflow.log_metric(f"{method}_false_positive_rate", row["false_positive_rate"])
        mlflow.log_metric(f"{method}_overall_accuracy", row["overall_accuracy"])
    print(f"Logged detection performance metrics for {len(df)} methods.")


def log_model_performance_metrics():
    """One metric per (model, scenario): average accuracy under that drift type."""
    path = f"{RESULTS_DIR}/model_metrics.csv"
    if not os.path.exists(path):
        print(f"Skipping model performance metrics -- {path} not found.")
        return

    df = pd.read_csv(path)
    summary = df.groupby(["model", "scenario"])["accuracy"].mean().reset_index()
    for _, row in summary.iterrows():
        model_name = row["model"].lower().replace(" ", "_")
        metric_name = f"accuracy_{model_name}_{row['scenario']}"
        mlflow.log_metric(metric_name, row["accuracy"])
    print(f"Logged model accuracy metrics for {len(summary)} (model, scenario) pairs.")


def log_key_finding_metrics():
    """The headline concept-drift blind-spot numbers, logged explicitly so
    they're easy to find later without digging through CSVs."""
    path = f"{RESULTS_DIR}/scenario_score_vs_accuracy_summary.csv"
    if not os.path.exists(path):
        print(f"Skipping key finding metrics -- {path} not found.")
        return

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        scenario = row["scenario"]
        mlflow.log_metric(f"drift_score_{scenario}", row["drift_score"])
        mlflow.log_metric(f"accuracy_drop_{scenario}", row["accuracy_drop"])
    print(f"Logged scenario-level drift score / accuracy drop for {len(df)} scenarios.")


def log_artifacts():
    """Attach every results CSV, plot, and trained model file to this run
    so the full evidence trail is browsable inside MLflow's UI."""
    if os.path.isdir(RESULTS_DIR):
        for fname in os.listdir(RESULTS_DIR):
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.isfile(fpath):
                mlflow.log_artifact(fpath, artifact_path="results")

    if os.path.isdir(PLOTS_DIR):
        for fname in os.listdir(PLOTS_DIR):
            fpath = os.path.join(PLOTS_DIR, fname)
            mlflow.log_artifact(fpath, artifact_path="results/plots")

    if os.path.isdir(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            fpath = os.path.join(MODEL_DIR, fname)
            mlflow.log_artifact(fpath, artifact_path="models")

    print("Logged all results CSVs, plots, and model files as MLflow artifacts.")


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full_pipeline_run") as run:
        print(f"Started MLflow run: {run.info.run_id}")

        log_params_and_config(run)
        log_detection_performance_metrics()
        log_model_performance_metrics()
        log_key_finding_metrics()
        log_artifacts()

        print(f"\nDone. Run ID: {run.info.run_id}")
        print("View results with:  mlflow ui")
        print("Then open: http://localhost:5000")


if __name__ == "__main__":
    main()