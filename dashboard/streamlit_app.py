"""
dashboard/streamlit_app.py
----------------------------
Phase 7: Interactive dashboard tying together everything built so far --
your saved experiment results, your 6 trained drift detectors, and your
3 trained income-prediction models (LR, RF, MLP).

Run from the project root with:
    streamlit run dashboard/streamlit_app.py

Four tabs:
    1. Experiment Results  -- your existing saved plots/tables (static, from results/)
    2. Live Drift Check    -- upload a CSV, compare it against window_0 with all 6 detectors
    3. Live Prediction     -- fill in one person's details, get predictions from all 3 models
    4. Live Database       -- (optional) recent predictions/drift checks from your deployed
                               Postgres database, if DATABASE_URL is set
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make src/ and app/ importable from this subfolder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(PROJECT_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

st.set_page_config(page_title="Data Drift Detection Dashboard", layout="wide")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REFERENCE_WINDOW_PATH = os.path.join(DATA_DIR, "window_0.csv")

NUMERIC_FEATURE_COLS = ["age", "hours.per.week", "capital.gain", "capital.loss"]
AUTOENCODER_FEATURE_COLS = NUMERIC_FEATURE_COLS + ["education"]
TARGET = "income"


# ---------------------------------------------------------------------------
# Cached loaders -- models and data load once per session, not on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detectors():
    from drift_detectors import (
        ks_test_drift, calculate_psi, calculate_kl_divergence,
        classifier_drift_detector, autoencoder_drift_detector,
    )
    return {
        "ks_test_drift": ks_test_drift,
        "calculate_psi": calculate_psi,
        "calculate_kl_divergence": calculate_kl_divergence,
        "classifier_drift_detector": classifier_drift_detector,
        "autoencoder_drift_detector": autoencoder_drift_detector,
    }


@st.cache_resource
def load_reference_window():
    if not os.path.exists(REFERENCE_WINDOW_PATH):
        return None
    return pd.read_csv(REFERENCE_WINDOW_PATH)


@st.cache_resource
def load_prediction_models():
    models = {}
    lr_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    if os.path.exists(lr_path):
        models["Logistic Regression"] = joblib.load(lr_path)
    if os.path.exists(rf_path):
        models["Random Forest"] = joblib.load(rf_path)

    mlp_model_path = os.path.join(MODEL_DIR, "mlp_model.keras")
    mlp_prep_path = os.path.join(MODEL_DIR, "mlp_preprocessor.joblib")
    if os.path.exists(mlp_model_path) and os.path.exists(mlp_prep_path):
        from tensorflow import keras
        models["MLP (Deep Learning)"] = {
            "model": keras.models.load_model(mlp_model_path),
            "preprocessor": joblib.load(mlp_prep_path),
        }
    return models


def predict_with_all_models(models: dict, record_df: pd.DataFrame):
    """Returns {model_name: (prediction_label, probability_over_50k)}"""
    results = {}
    for name, model in models.items():
        try:
            if name == "MLP (Deep Learning)":
                X = model["preprocessor"].transform(record_df)
                if hasattr(X, "toarray"):
                    X = X.toarray()
                proba = float(model["model"].predict(X, verbose=0).flatten()[0])
                label = ">50K" if proba > 0.5 else "<=50K"
            else:
                proba = float(model.predict_proba(record_df)[0][1])
                label = ">50K" if proba > 0.5 else "<=50K"
            results[name] = (label, proba)
        except Exception as e:
            results[name] = (f"Error: {e}", None)
    return results


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📊 Data Drift Detection Dashboard")
st.caption(
    "A Comparative Study of Data Drift Detection Techniques for Deployed ML Models — "
    "statistical, classical ML, and deep learning methods, side by side."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Experiment Results", "🔍 Live Drift Check", "🤖 Live Prediction", "🗄️ Live Database",
])


# ---------------------------------------------------------------------------
# TAB 1: Experiment Results (static, from your saved results/)
# ---------------------------------------------------------------------------
with tab1:
    st.header("Saved Experiment Results")

    perf_path = os.path.join(RESULTS_DIR, "detection_performance_summary.csv")
    if os.path.exists(perf_path):
        st.subheader("Detection Performance by Method")
        st.dataframe(pd.read_csv(perf_path), width='stretch')
    else:
        st.info("Run `python src/analyze_results.py` first to generate detection_performance_summary.csv")

    col1, col2 = st.columns(2)
    heatmap_path = os.path.join(PLOTS_DIR, "detection_rate_heatmap.png")
    accuracy_path = os.path.join(PLOTS_DIR, "model_accuracy_by_scenario.png")
    with col1:
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Detection Rate by Method x Scenario", width='stretch')
    with col2:
        if os.path.exists(accuracy_path):
            st.image(accuracy_path, caption="Model Accuracy by Scenario", width='stretch')

    comparison_path = os.path.join(PLOTS_DIR, "classical_vs_dl_comparison.png")
    if os.path.exists(comparison_path):
        st.subheader("Classical vs. Deep Learning Detector")
        st.image(comparison_path, width='stretch')

    scenario_summary_path = os.path.join(RESULTS_DIR, "scenario_score_vs_accuracy_summary.csv")
    if os.path.exists(scenario_summary_path):
        st.subheader("Drift Score vs. Real Accuracy Drop, by Scenario")
        st.dataframe(pd.read_csv(scenario_summary_path), width='stretch')

    model_metrics_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
    if os.path.exists(model_metrics_path):
        st.subheader("Full Model Performance Table")
        st.dataframe(pd.read_csv(model_metrics_path), width='stretch')


# ---------------------------------------------------------------------------
# TAB 2: Live Drift Check -- upload a CSV, run all 6 detectors
# ---------------------------------------------------------------------------
with tab2:
    st.header("Check a New Batch of Data for Drift")
    st.write(
        "Upload a CSV with the same columns as your reference window "
        "(age, hours.per.week, capital.gain, capital.loss, education, ...) "
        "to compare it against `window_0` using all 6 detection methods."
    )

    reference_df = load_reference_window()
    if reference_df is None:
        st.error(f"Reference window not found at {REFERENCE_WINDOW_PATH}")
    else:
        uploaded_file = st.file_uploader("Upload a CSV batch to check", type=["csv"], key="drift_upload")

        if uploaded_file is not None:
            comparison_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(comparison_df)} rows.")
            st.dataframe(comparison_df.head(), width='stretch')

            missing_cols = [c for c in AUTOENCODER_FEATURE_COLS if c not in comparison_df.columns]
            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {missing_cols}")
            elif st.button("Run Drift Check (all 6 methods)", type="primary"):
                detectors = load_detectors()
                with st.spinner("Running KS, PSI, KL, Classifier (RF + LR), and Autoencoder..."):
                    results = []

                    for r in [
                        detectors["ks_test_drift"](reference_df["age"], comparison_df["age"]),
                        detectors["calculate_psi"](reference_df["age"], comparison_df["age"]),
                        detectors["calculate_kl_divergence"](reference_df["age"], comparison_df["age"]),
                    ]:
                        results.append({"method": r["method"], "column": "age", "score": r["score"], "drifted": r["drifted"]})

                    for r in [
                        detectors["calculate_psi"](reference_df["education"], comparison_df["education"]),
                        detectors["calculate_kl_divergence"](reference_df["education"], comparison_df["education"]),
                    ]:
                        results.append({"method": r["method"], "column": "education", "score": r["score"], "drifted": r["drifted"]})

                    for model_type in ["random_forest", "logistic_regression"]:
                        r = detectors["classifier_drift_detector"](reference_df, comparison_df, NUMERIC_FEATURE_COLS, model_type=model_type)
                        results.append({"method": r["method"], "column": "multi-feature", "score": r["score"], "drifted": r["drifted"]})

                    r = detectors["autoencoder_drift_detector"](reference_df, comparison_df, AUTOENCODER_FEATURE_COLS, epochs=50)
                    results.append({"method": r["method"], "column": "multi-feature", "score": r["score"], "drifted": r["drifted"]})

                results_df = pd.DataFrame(results)

                n_flagged = results_df["drifted"].sum()
                if n_flagged >= 3:
                    st.error(f"⚠️ Drift detected by {n_flagged} of 6 methods")
                elif n_flagged >= 1:
                    st.warning(f"⚠️ Drift detected by {n_flagged} of 6 methods")
                else:
                    st.success("✅ No drift detected by any method")

                st.dataframe(
                    results_df.style.apply(
                        lambda row: ["background-color: #ffcccc" if row["drifted"] else "" for _ in row],
                        axis=1,
                    ),
                    width='stretch',
                )
                st.bar_chart(results_df.set_index("method")["score"])


# ---------------------------------------------------------------------------
# TAB 3: Live Prediction -- fill in a form, get predictions from all 3 models
# ---------------------------------------------------------------------------
with tab3:
    st.header("Predict Income for One Person")
    st.write("Enter details below to get a live prediction from all three trained models.")

    reference_df = load_reference_window()
    models = load_prediction_models()

    if not models:
        st.error("No trained models found in `models/`. Run train_model.py and train_dl_model.py first.")
    else:
        st.caption(f"Loaded models: {', '.join(models.keys())}")

        def options_for(col, fallback):
            if reference_df is not None and col in reference_df.columns:
                return sorted(reference_df[col].dropna().unique().tolist())
            return fallback

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=17, max_value=90, value=39)
                workclass = st.selectbox("Workclass", options_for("workclass", ["Private"]))
                fnlwgt = st.number_input("fnlwgt", value=77516)
                education = st.selectbox("Education", options_for("education", ["Bachelors"]))
                education_num = st.number_input("Education-num", min_value=1, max_value=16, value=13)
            with c2:
                marital_status = st.selectbox("Marital Status", options_for("marital.status", ["Never-married"]))
                occupation = st.selectbox("Occupation", options_for("occupation", ["Tech-support"]))
                relationship = st.selectbox("Relationship", options_for("relationship", ["Not-in-family"]))
                race = st.selectbox("Race", options_for("race", ["White"]))
                sex = st.selectbox("Sex", options_for("sex", ["Male"]))
            with c3:
                capital_gain = st.number_input("Capital Gain", min_value=0.0, value=0.0)
                capital_loss = st.number_input("Capital Loss", min_value=0.0, value=0.0)
                hours_per_week = st.number_input("Hours per Week", min_value=1.0, max_value=99.0, value=40.0)
                native_country = st.selectbox("Native Country", options_for("native.country", ["United-States"]))

            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            record_df = pd.DataFrame([{
                "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
                "education": education, "education.num": education_num,
                "marital.status": marital_status, "occupation": occupation,
                "relationship": relationship, "race": race, "sex": sex,
                "capital.gain": capital_gain, "capital.loss": capital_loss,
                "hours.per.week": hours_per_week, "native.country": native_country,
            }])

            with st.spinner("Running all 3 models..."):
                predictions = predict_with_all_models(models, record_df)

            cols = st.columns(len(predictions))
            for col, (name, (label, proba)) in zip(cols, predictions.items()):
                with col:
                    st.metric(label=name, value=label, delta=f"{proba:.1%} prob. >50K" if proba is not None else None)


# ---------------------------------------------------------------------------
# TAB 4: Live Database -- optional, only if DATABASE_URL is set
# ---------------------------------------------------------------------------
with tab4:
    st.header("Live Deployed Database")
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        st.info(
            "No `DATABASE_URL` environment variable set, so this tab can't connect. "
            "Set it (same connection string your FastAPI app uses) and restart Streamlit "
            "to see recent predictions and drift checks logged by your live deployed API."
        )
    else:
        try:
            from app import db
            st.subheader("Recent Predictions (from your deployed API)")
            preds = db.fetch_recent_predictions(limit=25)
            if preds:
                st.dataframe(pd.DataFrame(preds), width='stretch')
            else:
                st.write("No predictions logged yet.")

            st.subheader("Recent Drift Checks (from your deployed API)")
            checks = db.fetch_recent_drift_checks(limit=25)
            if checks:
                st.dataframe(pd.DataFrame(checks), width='stretch')
            else:
                st.write("No drift checks logged yet.")
        except Exception as e:
            st.error(f"Could not connect to database: {e}")