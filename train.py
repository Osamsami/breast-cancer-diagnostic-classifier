"""Train a K-Nearest Neighbors classifier on the Wisconsin Diagnostic Breast
Cancer dataset and persist it to disk.

This mirrors the modeling steps in project.ipynb (drop unused columns,
encode the target, scale features, fit a KNN classifier with k=9) as a
standalone, reproducible script that the Streamlit app and CI pipeline
can rely on.

Usage:
    python train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "knn_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"

N_NEIGHBORS = 9
TEST_SIZE = 0.3
RANDOM_STATE = 70


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and drop columns that aren't model features."""
    df = pd.read_csv(path)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def train(df: pd.DataFrame):
    """Split, scale, and fit the KNN classifier. Returns the fitted model,
    scaler, and the test split for evaluation."""
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test, list(X.columns)


def evaluate(model, X_test_scaled, y_test) -> None:
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print()
    print(
        classification_report(
            y_test, y_pred, target_names=["Benign", "Malignant"]
        )
    )


def main() -> None:
    df = load_data()
    model, scaler, X_test_scaled, y_test, feature_names = train(df)
    evaluate(model, X_test_scaled, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
