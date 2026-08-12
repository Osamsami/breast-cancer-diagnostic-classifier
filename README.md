# Breast Cancer Diagnostic Classifier

![CI](https://github.com/Osamsami/breast-cancer-diagnostic-classifier/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

A machine learning project that classifies breast tumors as **malignant** or **benign** from digitized fine needle aspirate (FNA) measurements, using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Overview

The notebook (`project.ipynb`) walks through the full workflow: loading and cleaning the data, scaling the features, tuning a **K-Nearest Neighbors (KNN)** classifier, and evaluating it with a hold-out test set and 10-fold cross-validation. A standalone script (`train.py`) reproduces the same pipeline outside the notebook and saves a trained model that a small Streamlit app (`app.py`) uses to serve interactive predictions.

## Dataset

- **Source:** [Wisconsin Diagnostic Breast Cancer (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), provided here as `data.csv`
- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric features computed from digitized images of a breast mass FNA — mean, standard error, and "worst" (largest) values for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension
- **Target:** `diagnosis` — `M` (malignant, encoded as 1) or `B` (benign, encoded as 0)

## Model

- **Algorithm:** K-Nearest Neighbors (`sklearn.neighbors.KNeighborsClassifier`)
- **Preprocessing:** features standardized with `StandardScaler`; `id` and the empty trailing column are dropped
- **Hyperparameter selection:** `k` swept from 1-20 against test-set error rate; `k = 9` chosen as the operating point
- **Split:** 70/30 train/test, `random_state=70`

> Earlier drafts of this README described the model as a Random Forest — that was inaccurate. The notebook has always trained a KNN classifier; the docs now match the code.

## Results

Metrics from the held-out test set (30% split, 171 samples):

| Metric | Value |
|---|---|
| Accuracy | 0.99 |
| Precision | 1.00 |
| Recall | 0.98 |
| F1-score | 0.99 |
| 10-fold CV mean accuracy | 96.47% |

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 108 | 0 |
| **Actual Malignant** | 1 | 62 |

These numbers come from a single train/test split on a small (569-row) dataset, so treat them as illustrative rather than a claim of clinical-grade performance.

## Live Demo

🔗 Live demo: _coming soon_

## Project Structure

```
.
├── project.ipynb             # Exploratory notebook: EDA, preprocessing, model tuning, evaluation
├── train.py                  # Standalone script: trains the KNN model and saves it to models/
├── app.py                    # Streamlit app: interactive prediction UI backed by the saved model
├── data.csv                  # Wisconsin Diagnostic Breast Cancer dataset
├── models/                   # Serialized model artifacts (created by train.py)
├── requirements.txt          # Pinned dependencies
├── Dockerfile                # Container build for the Streamlit app
└── .github/workflows/ci.yml  # CI: installs dependencies and runs train.py
```

## Installation

```bash
git clone https://github.com/Osamsami/breast-cancer-diagnostic-classifier.git
cd breast-cancer-diagnostic-classifier
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the notebook

Open `project.ipynb` in Jupyter or Google Colab and run all cells (make sure `data.csv` is in the same directory).

### Train the model from the command line

```bash
python train.py
```

This loads `data.csv`, splits it into train/test sets, fits the KNN classifier, prints evaluation metrics, and saves the trained model to `models/knn_model.joblib`.

### Run the Streamlit demo locally

```bash
python train.py        # only needed once, to produce the saved model
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) and fill in the form to get a live prediction.

### Run with Docker

```bash
docker build -t breast-cancer-classifier .
docker run -p 8501:8501 breast-cancer-classifier
```

## License

Released under the [MIT License](LICENSE).
