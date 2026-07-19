"""
data_prep.py
------------
Step 1 of the pipeline: load the dataset and split it into
sequential "time windows" that simulate batches of data arriving
over time (e.g., Week 1, Week 2, Week 3...).

We don't have a real timestamp column, so we simulate time by
simply chunking the shuffled dataset into equal-sized windows.
Window 0 = "reference" / training-time data.
Windows 1..N = "incoming production data" that we'll later inject
drift into (see drift_injection.py).
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/adult.csv"


def load_adult_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the Adult Census Income dataset from a local CSV
    (Kaggle: uciml/adult-census-income).

    Target column: 'income' (<=50K or >50K)

    Note: this Kaggle version uses '?' as its missing-value marker
    instead of NaN, so we handle that explicitly.
    """
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)

    # Clean up column names (strip whitespace, standardize case)
    df.columns = [c.strip() for c in df.columns]

    # Kaggle's version marks missing values as '?' instead of NaN
    df = df.replace("?", np.nan)

    # Drop rows with missing values for a clean first pass.
    # (You can revisit this later -- missingness itself can be a
    # feature of drift, but let's keep Phase 1 simple.)
    df = df.dropna().reset_index(drop=True)

    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def create_time_windows(df: pd.DataFrame, n_windows: int = 6, random_state: int = 42):
    """
    Shuffles the dataset and splits it into n_windows equal-sized
    chunks, simulating sequential batches of incoming data.

    Returns a list of DataFrames: [window_0, window_1, ..., window_N]
    window_0 is treated as the REFERENCE window (i.e., what the
    model was originally trained/validated on).
    """
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    windows = np.array_split(shuffled, n_windows)
    return windows


if __name__ == "__main__":
    df = load_adult_dataset()
    windows = create_time_windows(df, n_windows=6)

    for i, w in enumerate(windows):
        path = f"data/window_{i}.csv"
        w.to_csv(path, index=False)
        print(f"Saved {path} with {len(w)} rows")