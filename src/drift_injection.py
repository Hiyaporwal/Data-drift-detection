"""
drift_injection.py
-------------------
Step 2 of the pipeline: artificially inject different TYPES of drift
into later time windows, so we know the "ground truth" of when and
where drift happened. This ground truth is what lets us score each
detection method (accuracy, false positives, detection lag).

Three drift types, explained simply:

1. COVARIATE DRIFT
   The distribution of an input feature shifts, but the relationship
   between features and the target stays the same.
   Example: "age" values start skewing older across incoming batches.

2. LABEL DRIFT
   The proportion of target classes shifts.
   Example: suddenly 70% of records are ">50K income" instead of the
   original ~25%.

3. CONCEPT DRIFT
   The RELATIONSHIP between features and target changes, even if the
   feature distributions look the same.
   Example: "high education" used to strongly predict ">50K income",
   but now it doesn't (we simulate this by shuffling/flipping labels
   conditioned on a feature).

Each function returns a NEW dataframe (does not mutate the original)
plus a small metadata dict describing what was done -- store this
metadata, it becomes your ground-truth label for scoring later.
"""

import numpy as np
import pandas as pd


def inject_covariate_drift(df: pd.DataFrame, column: str, shift_std: float = 1.5,
                            random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Shifts a numeric column's distribution by `shift_std` standard
    deviations (computed from the column's own original std).

    Example: inject_covariate_drift(df, "age", shift_std=1.0)
    shifts the whole "age" column upward by 1 std dev.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    col_std = df[column].std()
    shift_amount = shift_std * col_std

    # add a little noise too, so it's not a perfectly uniform shift
    noise = rng.normal(loc=0, scale=col_std * 0.05, size=len(df))
    df[column] = df[column] + shift_amount + noise

    metadata = {
        "drift_type": "covariate",
        "column": column,
        "shift_std": shift_std,
    }
    return df, metadata


def inject_label_drift(df: pd.DataFrame, target_col: str, target_value,
                        new_positive_rate: float, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Resamples the dataframe so the target class distribution matches
    `new_positive_rate` for `target_value`.

    Example: inject_label_drift(df, "class", ">50K", new_positive_rate=0.6)
    resamples rows so 60% of the window has class ">50K".
    """
    rng = np.random.default_rng(random_state)

    positives = df[df[target_col] == target_value]
    negatives = df[df[target_col] != target_value]

    n_total = len(df)
    n_pos_target = int(n_total * new_positive_rate)
    n_neg_target = n_total - n_pos_target

    # sample with replacement if we need more rows than exist
    pos_sample = positives.sample(n=n_pos_target, replace=(n_pos_target > len(positives)),
                                   random_state=random_state)
    neg_sample = negatives.sample(n=n_neg_target, replace=(n_neg_target > len(negatives)),
                                   random_state=random_state)

    new_df = pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    metadata = {
        "drift_type": "label",
        "target_col": target_col,
        "new_positive_rate": new_positive_rate,
    }
    return new_df, metadata


def inject_concept_drift(df: pd.DataFrame, target_col: str, condition_col: str,
                          condition_value, flip_fraction: float = 0.5,
                          random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Flips the target label for a fraction of rows that match a
    condition, simulating a change in the feature-target relationship.

    Example: inject_concept_drift(df, "class", "education", "Bachelors",
                                   flip_fraction=0.5)
    flips the income label for 50% of rows where education == "Bachelors",
    simulating "having a Bachelors no longer predicts income the same way".

    NOTE: this only works cleanly for binary targets. Adjust the flip
    logic if your target has more than 2 classes.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    mask = df[condition_col] == condition_value
    matching_idx = df[mask].index

    n_to_flip = int(len(matching_idx) * flip_fraction)
    flip_idx = rng.choice(matching_idx, size=n_to_flip, replace=False)

    classes = df[target_col].unique()
    if len(classes) != 2:
        raise ValueError("inject_concept_drift currently supports binary targets only.")

    class_a, class_b = classes[0], classes[1]

    def flip(val):
        return class_b if val == class_a else class_a

    df.loc[flip_idx, target_col] = df.loc[flip_idx, target_col].apply(flip)

    metadata = {
        "drift_type": "concept",
        "condition_col": condition_col,
        "condition_value": condition_value,
        "flip_fraction": flip_fraction,
    }
    return df, metadata


if __name__ == "__main__":
    # Quick smoke test with dummy data
    dummy = pd.DataFrame({
        "age": np.random.normal(35, 10, 200),
        "education": np.random.choice(["Bachelors", "HS-grad", "Masters"], 200),
        "class": np.random.choice(["<=50K", ">50K"], 200),
    })

    drifted_cov, meta1 = inject_covariate_drift(dummy, "age", shift_std=1.0)
    print("Covariate drift metadata:", meta1)
    print(f"Original age mean: {dummy['age'].mean():.2f} -> Drifted: {drifted_cov['age'].mean():.2f}")

    drifted_label, meta2 = inject_label_drift(dummy, "class", ">50K", new_positive_rate=0.6)
    print("Label drift metadata:", meta2)
    print(f"New positive rate: {(drifted_label['class'] == '>50K').mean():.2f}")

    drifted_concept, meta3 = inject_concept_drift(dummy, "class", "education", "Bachelors", flip_fraction=0.5)
    print("Concept drift metadata:", meta3)