"""
train_dl_model.py
-------------------
Trains a deep learning (MLP) income predictor, as a third model alongside
the existing Logistic Regression and Random Forest from train_model.py.

Uses the SAME preprocessing pattern (ColumnTransformer: median-imputed +
scaled numeric, most-frequent-imputed + one-hot categorical) so the
comparison between models is fair -- all three see identically prepared
features.

Saves:
    models/mlp_preprocessor.joblib   -- the fitted ColumnTransformer
    models/mlp_model.keras            -- the trained Keras model
    results/training_metrics.csv      -- appends an "MLP (Deep Learning)" row
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRAIN_FILE = "data/window_0.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_mlp(input_dim: int, random_state: int = RANDOM_STATE) -> keras.Model:
    """
    A small, standard feedforward classifier: two hidden layers with
    dropout for regularization, sigmoid output for binary classification.
    Deliberately modest in size (this is tabular data with a moderate
    number of features, not images -- a huge network would just overfit).
    """
    tf.random.set_seed(random_state)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    df = pd.read_csv(TRAIN_FILE)
    TARGET = "income"

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # ColumnTransformer output can be sparse depending on OneHotEncoder --
    # Keras needs dense arrays.
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    print("Training MLP (deep learning) income predictor...")
    model = build_mlp(input_dim=X_train_processed.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train_processed, y_train,
        epochs=100, batch_size=32, validation_split=0.15,
        callbacks=[early_stop], verbose=0,
    )

    y_proba = model.predict(X_test_processed, verbose=0).flatten()
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nMLP (Deep Learning) test performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")

    # --- Save model + preprocessor for evaluate_dl_model.py ---
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "mlp_preprocessor.joblib"))
    model.save(os.path.join(MODEL_DIR, "mlp_model.keras"))
    print(f"\nSaved preprocessor and model to {MODEL_DIR}/")

    # --- Append to training_metrics.csv (create if missing, else append) ---
    new_row = pd.DataFrame([{
        "Model": "MLP (Deep Learning)",
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": auc,
    }])
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        existing = existing[existing["Model"] != "MLP (Deep Learning)"]  # avoid duplicate rows on re-run
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(metrics_path, index=False)
    print(f"Updated {metrics_path}")


if __name__ == "__main__":
    main()