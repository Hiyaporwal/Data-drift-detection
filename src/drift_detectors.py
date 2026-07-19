"""
drift_detectors.py
Four drift detection methods: KS-test, PSI, KL Divergence, Classifier-based.
Each function compares a reference window to a comparison window and returns:
    {"score": float, "drifted": bool, "method": str}
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# 1. KS-TEST (numeric features only)
# ---------------------------------------------------------------------------
def ks_test_drift(reference: pd.Series, comparison: pd.Series, alpha: float = 0.05):
    """
    Kolmogorov-Smirnov test on a single numeric column.
    Null hypothesis: both samples come from the same distribution.
    We flag drift if p-value < alpha (we reject the null).
    """
    reference = reference.dropna()
    comparison = comparison.dropna()

    stat, p_value = ks_2samp(reference, comparison)

    return {
        "method": "KS",
        "score": float(stat),        # KS statistic: 0 = identical, 1 = max difference
        "p_value": float(p_value),
        "drifted": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# 2. PSI (Population Stability Index) — numeric or categorical
# ---------------------------------------------------------------------------
def calculate_psi(reference: pd.Series, comparison: pd.Series, buckets: int = 10,
                   threshold: float = 0.2):
    """
    PSI buckets the reference distribution, then compares the % of comparison
    data falling into each bucket.
    Rule-of-thumb interpretation (industry standard in credit risk):
        < 0.1  -> no significant shift
        0.1-0.2 -> moderate shift, worth watching
        > 0.2  -> significant shift (we treat this as "drifted")
    """
    reference = reference.dropna()
    comparison = comparison.dropna()

    # If categorical, treat each category as its own bucket
    if reference.dtype == object or str(reference.dtype).startswith("category"):
        categories = set(reference.unique()) | set(comparison.unique())
        ref_counts = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        comp_counts = comparison.value_counts(normalize=True).reindex(categories, fill_value=0)
    else:
        # Numeric: bucket using reference quantiles so bins are meaningful
        breakpoints = np.quantile(reference, np.linspace(0, 1, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        breakpoints = np.unique(breakpoints)  # avoid duplicate edges

        ref_binned = pd.cut(reference, bins=breakpoints)
        comp_binned = pd.cut(comparison, bins=breakpoints)

        ref_counts = ref_binned.value_counts(normalize=True).sort_index()
        comp_counts = comp_binned.value_counts(normalize=True).sort_index()

    # Avoid division by zero / log(0) by flooring at a tiny epsilon
    eps = 1e-4
    ref_counts = ref_counts.replace(0, eps)
    comp_counts = comp_counts.replace(0, eps)

    psi_value = np.sum((comp_counts - ref_counts) * np.log(comp_counts / ref_counts))

    return {
        "method": "PSI",
        "score": float(psi_value),
        "drifted": bool(psi_value > threshold),
    }


# ---------------------------------------------------------------------------
# 3. KL DIVERGENCE — numeric or categorical
# ---------------------------------------------------------------------------
def calculate_kl_divergence(reference: pd.Series, comparison: pd.Series, buckets: int = 10,
                             threshold: float = 0.1):
    """
    KL(P || Q): how much comparison distribution (Q) diverges from
    reference distribution (P). Uses the same bucket setup as PSI so the
    two methods are directly comparable in your results table.
    """
    reference = reference.dropna()
    comparison = comparison.dropna()

    if reference.dtype == object or str(reference.dtype).startswith("category"):
        categories = set(reference.unique()) | set(comparison.unique())
        p = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
        q = comparison.value_counts(normalize=True).reindex(categories, fill_value=0)
    else:
        breakpoints = np.quantile(reference, np.linspace(0, 1, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        breakpoints = np.unique(breakpoints)

        p = pd.cut(reference, bins=breakpoints).value_counts(normalize=True).sort_index()
        q = pd.cut(comparison, bins=breakpoints).value_counts(normalize=True).sort_index()

    eps = 1e-4
    p = p.replace(0, eps)
    q = q.replace(0, eps)

    kl_value = np.sum(p * np.log(p / q))

    return {
        "method": "KL",
        "score": float(kl_value),
        "drifted": bool(kl_value > threshold),
    }


# ---------------------------------------------------------------------------
# 4. CLASSIFIER-BASED DRIFT DETECTOR — works across multiple columns at once
# ---------------------------------------------------------------------------
def classifier_drift_detector(reference_df: pd.DataFrame, comparison_df: pd.DataFrame,
                               feature_cols: list, auc_threshold: float = 0.6,
                               random_state: int = 42):
    """
    Trains a classifier to distinguish 'old' (reference) vs 'new' (comparison)
    rows using the given feature columns.
    If the classifier can tell them apart well (AUC well above 0.5), the
    underlying data distribution has shifted -> drift.
    AUC ~0.5 = can't tell them apart = no drift.
    """
    ref = reference_df[feature_cols].copy()
    comp = comparison_df[feature_cols].copy()

    ref["__label__"] = 0
    comp["__label__"] = 1

    combined = pd.concat([ref, comp], axis=0, ignore_index=True)

    # One-hot encode any categorical columns so RandomForest can use them
    combined = pd.get_dummies(combined, columns=[
        c for c in feature_cols if combined[c].dtype == object
    ])

    X = combined.drop(columns="__label__")
    y = combined["__label__"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=6)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return {
        "method": "Classifier",
        "score": float(auc),         # 0.5 = indistinguishable, 1.0 = perfectly distinguishable
        "drifted": bool(auc > auc_threshold),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: run all 4 methods on one numeric/categorical column
# ---------------------------------------------------------------------------
def run_all_detectors_on_column(reference_df, comparison_df, column):
    ref_col = reference_df[column]
    comp_col = comparison_df[column]

    results = [
        ks_test_drift(ref_col, comp_col) if ref_col.dtype != object else None,
        calculate_psi(ref_col, comp_col),
        calculate_kl_divergence(ref_col, comp_col),
    ]
    return [r for r in results if r is not None]


if __name__ == "__main__":
    # Quick smoke test using your existing windows
    ref_df = pd.read_csv("data/window_0.csv")
    comp_df = pd.read_csv("data/window_1.csv")

    print("=== age (numeric) ===")
    for r in run_all_detectors_on_column(ref_df, comp_df, "age"):
        print(r)

    print("\n=== education (categorical) ===")
    print(calculate_psi(ref_df["education"], comp_df["education"]))
    print(calculate_kl_divergence(ref_df["education"], comp_df["education"]))

    print("\n=== classifier-based (multi-feature) ===")
    numeric_cols = ["age", "hours.per.week", "capital.gain", "capital.loss"]
    print(classifier_drift_detector(ref_df, comp_df, numeric_cols))