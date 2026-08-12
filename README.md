# Breast Cancer Diagnostic Classification
!https://raw.githubusercontent.com/OsamaSami98/Breast-Cancer-Classification/main/banner.png

## Project Overview
This project uses **Machine Learning (K-Nearest Neighbors)** to classify breast tumors as either **Malignant** (Cancerous) or **Benign** (Non-cancerous). By analyzing 30+ clinical features, the model achieves near-perfect accuracy, making it a reliable tool for early diagnosis.

## Performance Highlights
- **Model:** K-Nearest Neighbors (KNN) Classifier, k=9
- **Accuracy:** 99% on the held-out test set
- **F1-Score:** 0.99
- **Dataset:** Wisconsin Breast Cancer (Diagnostic) Dataset

##  How it Works
1. **Data Loading:** Input clinical data from `data.csv`.
2. **Feature Analysis:** Correlating mean radius, texture, and perimeter to identify patterns.
3. **Training:** Features are standardized and a KNN classifier is fit after sweeping `k` from 1-20 to find the best-performing value.
4. **Validation:** Cross-validation performed to ensure the model doesn't overfit.



##  Files in this Project
* `project.ipynb`: The complete Python notebook containing data cleaning, EDA, and the ML model.
* `data.csv`: The dataset used for training and testing.

##  Usage
Simply open the `project.ipynb` file in Google Colab or Jupyter Notebook and run all cells. Make sure `data.csv` is in the same directory.

## 🏷️ Tags
#MachineLearning #Python #BreastCancerAwareness #DataScience #KNN #HealthTech
