"""Streamlit app for interactive breast cancer diagnosis predictions.

Loads the KNN model and scaler produced by train.py and lets a user enter
values for the 30 clinical features (pre-filled with dataset averages) to
get a live malignant/benign prediction.

Usage:
    python train.py   # once, to produce models/*.joblib
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "knn_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"

# (label, default/mean, min, max) for each of the 30 WDBC features, grouped
# by the three measurement types the dataset provides.
FEATURE_GROUPS = {
    "Mean": [
        ("radius_mean", 14.13, 6.98, 28.11),
        ("texture_mean", 19.29, 9.71, 39.28),
        ("perimeter_mean", 91.97, 43.79, 188.50),
        ("area_mean", 654.89, 143.50, 2501.00),
        ("smoothness_mean", 0.0964, 0.0526, 0.1634),
        ("compactness_mean", 0.1043, 0.0194, 0.3454),
        ("concavity_mean", 0.0888, 0.0, 0.4268),
        ("concave points_mean", 0.0489, 0.0, 0.2012),
        ("symmetry_mean", 0.1812, 0.1060, 0.3040),
        ("fractal_dimension_mean", 0.0628, 0.0500, 0.0974),
    ],
    "Standard Error": [
        ("radius_se", 0.4052, 0.1115, 2.8730),
        ("texture_se", 1.2169, 0.3602, 4.8850),
        ("perimeter_se", 2.8661, 0.7570, 21.9800),
        ("area_se", 40.3371, 6.8020, 542.2000),
        ("smoothness_se", 0.0070, 0.0017, 0.0311),
        ("compactness_se", 0.0255, 0.0023, 0.1354),
        ("concavity_se", 0.0319, 0.0, 0.3960),
        ("concave points_se", 0.0118, 0.0, 0.0528),
        ("symmetry_se", 0.0205, 0.0079, 0.0790),
        ("fractal_dimension_se", 0.0038, 0.0009, 0.0298),
    ],
    "Worst": [
        ("radius_worst", 16.2692, 7.9300, 36.0400),
        ("texture_worst", 25.6772, 12.0200, 49.5400),
        ("perimeter_worst", 107.2612, 50.4100, 251.2000),
        ("area_worst", 880.5831, 185.2000, 4254.0000),
        ("smoothness_worst", 0.1324, 0.0712, 0.2226),
        ("compactness_worst", 0.2543, 0.0273, 1.0580),
        ("concavity_worst", 0.2722, 0.0, 1.2520),
        ("concave points_worst", 0.1146, 0.0, 0.2910),
        ("symmetry_worst", 0.2901, 0.1565, 0.6638),
        ("fractal_dimension_worst", 0.0839, 0.0550, 0.2075),
    ],
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, feature_names


def render_form() -> dict:
    values = {}
    tabs = st.tabs(list(FEATURE_GROUPS.keys()))
    for tab, (group_name, features) in zip(tabs, FEATURE_GROUPS.items()):
        with tab:
            for name, default, min_val, max_val in features:
                values[name] = st.number_input(
                    name.replace("_", " ").title(),
                    min_value=float(min_val),
                    max_value=float(max_val) * 1.5,
                    value=float(default),
                    format="%.4f",
                    key=name,
                )
    return values


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Diagnostic Classifier", page_icon="🩺")
    st.title("Breast Cancer Diagnostic Classifier")
    st.write(
        "Enter clinical measurements from a fine needle aspirate (FNA) below "
        "to get a prediction from a K-Nearest Neighbors model trained on the "
        "Wisconsin Diagnostic Breast Cancer dataset. Fields are pre-filled "
        "with dataset averages as a starting point."
    )

    if not MODEL_PATH.exists():
        st.error(
            "No trained model found. Run `python train.py` first to create "
            "models/knn_model.joblib."
        )
        return

    model, scaler, feature_names = load_artifacts()
    values = render_form()

    if st.button("Predict", type="primary"):
        ordered = np.array([[values[name] for name in feature_names]])
        scaled = scaler.transform(ordered)
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]

        if prediction == 1:
            st.error(f"Prediction: **Malignant** (confidence: {proba[1]:.1%})")
        else:
            st.success(f"Prediction: **Benign** (confidence: {proba[0]:.1%})")

        st.caption(
            "This is a portfolio demo, not a medical device. Do not use it "
            "for real diagnostic decisions."
        )


if __name__ == "__main__":
    main()
