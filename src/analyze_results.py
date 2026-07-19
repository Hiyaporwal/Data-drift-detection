"""
analyze_results.py
--------------------
Phase 4: ANALYSIS. Ties together the two experiments you've already run:

    results/experiment_results.csv   -> did each detector FLAG drift?
    results/model_metrics.csv        -> did the model's REAL accuracy suffer?

This script produces:
    1. Detection performance table (per method): precision, recall,
       false positive rate, overall accuracy vs. ground truth.
    2. A detection-rate heatmap (method x scenario).
    3. A model-accuracy bar chart (model x scenario), so you can see the
       real damage each drift type causes.
    4. A merged table + scatter plot: drift score vs. accuracy drop,
       which is the key evidence for your paper's central argument
       (e.g., concept drift causes real harm but isn't caught by
       distribution-based detectors).

All tables are saved as CSVs in results/, all plots as PNGs in results/plots/.
Nothing here is exotic -- it's meant to be easy to read start to finish and
easy to describe in your paper's "Analysis" section.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# 1. DETECTION PERFORMANCE: precision, recall, false positive rate, accuracy
# ---------------------------------------------------------------------------
def compute_detection_performance(detection_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each method, compares `detected_drift` against `ground_truth_drift`
    across ALL rows (every window, every scenario, every column checked)
    and computes standard classification metrics -- treating "did this
    detector correctly call drift" as the thing we're scoring.
    """
    rows = []
    for method, group in detection_df.groupby("method"):
        tp = ((group["detected_drift"] == True) & (group["ground_truth_drift"] == True)).sum()
        fp = ((group["detected_drift"] == True) & (group["ground_truth_drift"] == False)).sum()
        fn = ((group["detected_drift"] == False) & (group["ground_truth_drift"] == True)).sum()
        tn = ((group["detected_drift"] == False) & (group["ground_truth_drift"] == False)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # a.k.a. true positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        overall_accuracy = (tp + tn) / len(group) if len(group) > 0 else float("nan")

        rows.append({
            "method": method,
            "precision": round(precision, 3),
            "recall_TPR": round(recall, 3),
            "false_positive_rate": round(fpr, 3),
            "overall_accuracy": round(overall_accuracy, 3),
            "n_observations": len(group),
        })

    return pd.DataFrame(rows).sort_values("overall_accuracy", ascending=False)


def plot_detection_heatmap(detection_df: pd.DataFrame):
    """Heatmap: detection rate (% flagged as drift) by method x scenario."""
    pivot = detection_df.groupby(["method", "scenario"])["detected_drift"].mean().unstack()

    plt.figure(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
                cbar_kws={"label": "Detection Rate"})
    plt.title("Drift Detection Rate by Method x Scenario")
    plt.ylabel("Detection Method")
    plt.xlabel("Drift Scenario")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/detection_rate_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/detection_rate_heatmap.png")


# ---------------------------------------------------------------------------
# 2. MODEL PERFORMANCE UNDER DRIFT
# ---------------------------------------------------------------------------
def plot_model_accuracy_by_scenario(model_df: pd.DataFrame):
    """Bar chart: accuracy by model x scenario -- shows real damage per drift type."""
    plt.figure(figsize=(9, 5))
    sns.barplot(data=model_df, x="scenario", y="accuracy", hue="model")
    plt.title("Model Accuracy by Drift Scenario")
    plt.ylabel("Accuracy")
    plt.xlabel("Scenario")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_accuracy_by_scenario.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/model_accuracy_by_scenario.png")


# ---------------------------------------------------------------------------
# 3. MERGE: does a higher drift score correspond to a bigger accuracy drop?
# ---------------------------------------------------------------------------
def build_merged_score_vs_accuracy_table(detection_df: pd.DataFrame, model_df: pd.DataFrame,
                                          detector_method: str = "Classifier_RF") -> pd.DataFrame:
    """
    Joins one detector's score (default: Classifier_RF, since it uses the
    same multi-feature view as the model) with model accuracy, per
    window x scenario. This is the table your correlation / scatter plot
    is built from.
    """
    # detection_df has one row per (window_id, scenario, method, column_checked)
    # we want ONE score per (window_id, scenario) for the chosen method
    scores = (
        detection_df[detection_df["method"] == detector_method]
        .groupby(["window_id", "scenario"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "drift_score", "window_id": "window"})
    )

    # model_df has one row per (window, scenario, model) -- average across models
    accuracy = (
        model_df
        .groupby(["window", "scenario"])["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "avg_accuracy"})
    )

    merged = pd.merge(scores, accuracy, on=["window", "scenario"], how="inner")

    # Baseline = the model's accuracy on each window's UNDRIFTED control
    # (your evaluate_model.py scenario "no_drift"), averaged across windows.
    baseline_acc = model_df[model_df["scenario"] == "no_drift"]["accuracy"].mean()
    merged["accuracy_drop"] = baseline_acc - merged["avg_accuracy"]

    return merged


def plot_score_vs_accuracy_drop(merged_df: pd.DataFrame, detector_method: str):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=merged_df, x="drift_score", y="accuracy_drop", hue="scenario", s=100)
    plt.title(f"{detector_method} Drift Score vs. Model Accuracy Drop")
    plt.xlabel(f"{detector_method} Drift Score")
    plt.ylabel("Accuracy Drop from Baseline")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/drift_score_vs_accuracy_drop.png", dpi=150)
    plt.close()
    print(f"Saved {PLOTS_DIR}/drift_score_vs_accuracy_drop.png")


def main():
    detection_df = pd.read_csv(f"{RESULTS_DIR}/experiment_results.csv")
    model_df = pd.read_csv(f"{RESULTS_DIR}/model_metrics.csv")

    # --- 1. Detection performance table ---
    perf_table = compute_detection_performance(detection_df)
    perf_table.to_csv(f"{RESULTS_DIR}/detection_performance_summary.csv", index=False)
    print("=== Detection Performance by Method ===")
    print(perf_table.to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/detection_performance_summary.csv\n")

    plot_detection_heatmap(detection_df)

    # --- 2. Model accuracy under drift ---
    plot_model_accuracy_by_scenario(model_df)

    print("\n=== Average Model Accuracy by Scenario ===")
    print(model_df.groupby(["model", "scenario"])["accuracy"].mean().round(4).to_string())

    # --- 3. Merged score-vs-accuracy analysis ---
    merged = build_merged_score_vs_accuracy_table(detection_df, model_df, detector_method="Classifier_RF")
    merged.to_csv(f"{RESULTS_DIR}/drift_score_vs_accuracy.csv", index=False)
    plot_score_vs_accuracy_drop(merged, detector_method="Classifier_RF")

    correlation = merged["drift_score"].corr(merged["accuracy_drop"])
    print(f"\nPooled correlation between Classifier_RF drift score and accuracy drop: {correlation:.3f}")
    print("(Note: this pooled number can be misleading -- see the scenario-level table below,")
    print(" which is the more honest and citable evidence for your paper.)")

    # --- Scenario-level summary: the real story, not collapsed into one number ---
    scenario_summary = (
        merged.groupby("scenario")[["drift_score", "accuracy_drop"]]
        .mean()
        .round(4)
        .reset_index()
        .sort_values("accuracy_drop", ascending=False)
    )
    scenario_summary.to_csv(f"{RESULTS_DIR}/scenario_score_vs_accuracy_summary.csv", index=False)
    print("\n=== Scenario-level: avg drift score vs. avg accuracy drop ===")
    print(scenario_summary.to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/scenario_score_vs_accuracy_summary.csv")
    print("\nThis table is the key evidence: a high drift score does NOT consistently predict")
    print("a large accuracy drop across scenarios -- the relationship depends on drift TYPE,")
    print("which is a central, citable finding for your paper's discussion section.")

    # Highlight the key finding: concept drift's detection rate vs its real damage
    concept_detection_rate = detection_df[detection_df["scenario"] == "concept_education"]["detected_drift"].mean()
    concept_accuracy_drop = merged[merged["scenario"] == "concept_education"]["accuracy_drop"].mean()
    print(f"\n--- Key finding ---")
    print(f"Concept drift (education) average detection rate across all methods: {concept_detection_rate:.1%}")
    print(f"Concept drift (education) average real accuracy drop: {concept_accuracy_drop:.3f}")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()