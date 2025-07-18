# 🧠 Stroke Prediction Using Classical Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)

A comprehensive comparative study of classical supervised ML models for predicting stroke occurrence based on healthcare-related features. This repository contains source code, model evaluations, and experimental results aimed at early stroke detection.

---

## 📊 Summary of Model Performance

| Model                   | Best Params                                                                                                                                   | CV Accuracy | Test Accuracy | Precision (0 / 1) | Recall (0 / 1) | F1-Score (0 / 1) | Confusion Matrix     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------- | ----------------- | -------------- | ---------------- | -------------------- |
| **Logistic Regression** | `{'C': 0.001, 'max_iter': 100, 'penalty': 'l1', 'solver': 'saga'}`                                                                           | 0.9512      | 0.6901        | 0.99 / 0.12       | 0.68 / 0.87    | 0.81 / 0.21      | `[[828, 388], [8, 54]]` |
| **SVM**                 | `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf', 'shrinking': True}`                                                                             | 0.7534      | 0.7363        | 0.98 / 0.11       | 0.74 / 0.65    | 0.84 / 0.19      | `[[901, 315], [22, 40]]` |
| **Naive Bayes**         | `{'var_smoothing': 1e-09}`                                                                                                                    | 0.8617      | 0.8607        | 0.97 / 0.16       | 0.88 / 0.45    | 0.92 / 0.24      | `[[1072, 144], [34, 28]]` |
| **KNN**                 | `{'metric': 'euclidean', 'n_neighbors': 11, 'p': 1, 'weights': 'uniform'}`                                                                   | 0.9512      | 0.9507        | 0.95 / 0.00       | 1.00 / 0.00    | 0.97 / 0.00      | `[[1215, 1], [62, 0]]` |
| **Random Forest**       | `{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}`           | 0.9225      | 0.9163        | 0.96 / 0.23       | 0.95 / 0.31    | 0.96 / 0.26      | `[[1152, 64], [43, 19]]` |
| **XGBoost**             | `{'colsample_bytree': 0.8, 'gamma': 0.3, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 1.0}` | 0.9512      | **0.9515**    | 0.95 / 0.50       | 1.00 / 0.03    | 0.98 / 0.06      | `[[1214, 2], [60, 2]]` |

---

## ✅ Key Highlights

- 📌 Implemented and compared six classical models: **Logistic Regression, SVM, Naive Bayes, KNN, Random Forest, XGBoost**
- 🔍 Used **GridSearchCV** for hyperparameter tuning and **5-fold cross-validation**
- 📉 Tackled **class imbalance** problem in stroke prediction
- 📈 Performance measured using accuracy, precision, recall, F1-score, and confusion matrices

---

## 📖 Final Conclusion

This study evaluated multiple classical machine learning models for stroke prediction. **XGBoost** achieved the **highest test accuracy (95.15%)**, showing excellent overall classification performance. However, it, along with most models, **struggled to identify stroke cases (minority class)**, as seen by their low recall.

**Naïve Bayes** and **Random Forest** provided a better balance between accuracy and sensitivity to stroke cases, indicating their potential for clinical applications. The results stress the need to go **beyond accuracy** and improve **minority class recall**—crucial for medical diagnosis tasks.

Future work should incorporate **techniques to handle data imbalance**, such as **SMOTE**, **cost-sensitive learning**, or **custom loss functions**, to ensure more reliable real-world deployment of stroke prediction systems.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/stroke-prediction-ml.git
cd stroke-prediction-ml
pip install -r requirements.txt
