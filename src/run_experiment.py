"""
run_experiment.py
------------------
Phase 3: THE MAIN EXPERIMENT.

This script ties together drift_injection.py and drift_detectors.py to
answer the actual research question:

    "How well does each detection method (KS, PSI, KL, Classifier)
     catch different TYPES of drift (covariate, label, concept), and
     how quickly?"

WORKFLOW:
1. Load your real time windows (window_0.csv ... window_5.csv).
   window_0 is always the REFERENCE (undrifted baseline).
2. For each later window, create TWO versions:
     a) a "clean" copy (no drift injected) -- tests FALSE POSITIVE RATE
     b) a "drifted" copy (one drift type injected) -- tests TRUE POSITIVE
        RATE / detection accuracy
3. Run all 4 detectors on both versions vs. the reference window.
4. Record every result (with ground truth) into one tidy results table.
5. Save to results/experiment_results.csv for analysis in Phase 4.

This is intentionally a flat, readable script rather than a fancy class
hierarchy -- easy to read start to finish, easy to explain in your paper.
"""

import pandas as pd

from drift_injection import (
    inject_covariate_drift,
    inject_label_drift,
    inject_concept_drift,
)
from drift_detectors import (
    ks_test_drift,
    calculate_psi,
    calculate_kl_divergence,
    classifier_drift_detector,
)

# ---------------------------------------------------------------------------
# CONFIG -- tweak these to control the experiment
# ---------------------------------------------------------------------------
N_WINDOWS = 7                    # must match what you used in data_prep.py
REFERENCE_WINDOW_PATH = "data/window_0.csv"

NUMERIC_COL_FOR_COVARIATE_DRIFT = "age"
TARGET_COL = "income"
POSITIVE_CLASS = ">50K"
CONCEPT_DRIFT_CONDITION_COL = "education"
CONCEPT_DRIFT_CONDITION_VALUE = "Bachelors"

# Feature columns the classifier-based detector will use
CLASSIFIER_FEATURE_COLS = ["age", "hours.per.week", "capital.gain", "capital.loss"]


def run_numeric_detectors(reference_col: pd.Series, comparison_col: pd.Series):
    """Runs KS, PSI, KL on a single numeric column. Returns a list of result dicts."""
    return [
        ks_test_drift(reference_col, comparison_col),
        calculate_psi(reference_col, comparison_col),
        calculate_kl_divergence(reference_col, comparison_col),
    ]


def run_categorical_detectors(reference_col: pd.Series, comparison_col: pd.Series):
    """Runs PSI, KL on a single categorical column (KS doesn't apply to categoricals)."""
    return [
        calculate_psi(reference_col, comparison_col),
        calculate_kl_divergence(reference_col, comparison_col),
    ]


def evaluate_window(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                     window_id: int, scenario: str, ground_truth_drift: bool,
                     drift_metadata: dict = None):
    """
    Runs ALL detectors relevant to this scenario against one comparison window,
    and returns a list of flat result rows ready to append to our results table.

    `scenario` is a label like "no_drift", "covariate_age", "label_income",
    "concept_education" -- describes WHAT we did to this window.
    `ground_truth_drift` is True/False -- did we actually inject drift here?
    """
    rows = []

    # --- Numeric column checks (age) ---
    for result in run_numeric_detectors(reference_df[NUMERIC_COL_FOR_COVARIATE_DRIFT],
                                         comparison_df[NUMERIC_COL_FOR_COVARIATE_DRIFT]):
        rows.append({
            "window_id": window_id,
            "scenario": scenario,
            "column_checked": NUMERIC_COL_FOR_COVARIATE_DRIFT,
            "method": result["method"],
            "score": result["score"],
            "detected_drift": result["drifted"],
            "ground_truth_drift": ground_truth_drift,
            "drift_metadata": drift_metadata,
        })

    # --- Categorical column checks (education) ---
    for result in run_categorical_detectors(reference_df[CONCEPT_DRIFT_CONDITION_COL],
                                             comparison_df[CONCEPT_DRIFT_CONDITION_COL]):
        rows.append({
            "window_id": window_id,
            "scenario": scenario,
            "column_checked": CONCEPT_DRIFT_CONDITION_COL,
            "method": result["method"],
            "score": result["score"],
            "detected_drift": result["drifted"],
            "ground_truth_drift": ground_truth_drift,
            "drift_metadata": drift_metadata,
        })

    # --- Classifier-based checks (multi-feature): both RF and LR variants ---
    for model_type in ["random_forest", "logistic_regression"]:
        clf_result = classifier_drift_detector(reference_df, comparison_df, CLASSIFIER_FEATURE_COLS,
                                                 model_type=model_type)
        rows.append({
            "window_id": window_id,
            "scenario": scenario,
            "column_checked": "multi_feature",
            "method": clf_result["method"],
            "score": clf_result["score"],
            "detected_drift": clf_result["drifted"],
            "ground_truth_drift": ground_truth_drift,
            "drift_metadata": drift_metadata,
        })

    return rows


def main():
    reference_df = pd.read_csv(REFERENCE_WINDOW_PATH)
    all_results = []

    for i in range(1, N_WINDOWS):
        window_path = f"data/window_{i}.csv"
        window_df = pd.read_csv(window_path)
        print(f"\n=== Processing window_{i} ===")

        # --- Scenario A: no drift injected (control / false-positive check) ---
        print("  -> scenario: no_drift (control)")
        all_results.extend(
            evaluate_window(reference_df, window_df, window_id=i,
                             scenario="no_drift", ground_truth_drift=False)
        )

        # --- Scenario B: covariate drift on the numeric column ---
        print("  -> scenario: covariate drift (age)")
        drifted_cov, meta_cov = inject_covariate_drift(
            window_df, column=NUMERIC_COL_FOR_COVARIATE_DRIFT, shift_std=1.5
        )
        all_results.extend(
            evaluate_window(reference_df, drifted_cov, window_id=i,
                             scenario="covariate_age", ground_truth_drift=True,
                             drift_metadata=meta_cov)
        )

        # --- Scenario C: label drift on the target column ---
        print("  -> scenario: label drift (income)")
        drifted_label, meta_label = inject_label_drift(
            window_df, target_col=TARGET_COL, target_value=POSITIVE_CLASS,
            new_positive_rate=0.6
        )
        all_results.extend(
            evaluate_window(reference_df, drifted_label, window_id=i,
                             scenario="label_income", ground_truth_drift=True,
                             drift_metadata=meta_label)
        )

        # --- Scenario D: concept drift on education -> income relationship ---
        print("  -> scenario: concept drift (education)")
        drifted_concept, meta_concept = inject_concept_drift(
            window_df, target_col=TARGET_COL,
            condition_col=CONCEPT_DRIFT_CONDITION_COL,
            condition_value=CONCEPT_DRIFT_CONDITION_VALUE,
            flip_fraction=0.5
        )
        all_results.extend(
            evaluate_window(reference_df, drifted_concept, window_id=i,
                             scenario="concept_education", ground_truth_drift=True,
                             drift_metadata=meta_concept)
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/experiment_results.csv", index=False)
    print(f"\nSaved {len(results_df)} result rows to results/experiment_results.csv")

    # Quick sanity summary -- per method, how often did it flag drift
    # correctly vs incorrectly? Full analysis happens in Phase 4.
    summary = results_df.groupby(["method", "scenario"])["detected_drift"].mean()
    print("\n=== Quick summary: detection rate by method x scenario ===")
    print(summary)


if __name__ == "__main__":
    main()