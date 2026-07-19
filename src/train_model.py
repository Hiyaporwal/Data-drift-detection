import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_FILE = "data/window_0.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(TRAIN_FILE)

TARGET = "income"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Convert target if needed
if y.dtype == object:
    y = y.map({
        "<=50K":0,
        ">50K":1,
        "<=50K.":0,
        ">50K.":1
    })

# --------------------------------------------------
# Column types
# --------------------------------------------------

numeric_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()

categorical_features = X.select_dtypes(include=["object","category"]).columns.tolist()

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --------------------------------------------------
# Models
# --------------------------------------------------

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": lr_pipeline,
    "Random Forest": rf_pipeline
}

metrics = []

# --------------------------------------------------
# Training
# --------------------------------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,prob)

    metrics.append([
        name,
        acc,
        prec,
        rec,
        f1,
        auc
    ])

    filename = name.lower().replace(" ","_") + ".pkl"

    joblib.dump(model, os.path.join(MODEL_DIR,filename))

# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metrics,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]
)

metrics_df.to_csv(
    "results/training_metrics.csv",
    index=False
)

print("\nTraining Complete\n")
print(metrics_df)