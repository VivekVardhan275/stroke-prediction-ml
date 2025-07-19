# 🧠 Stroke Prediction Using Classical Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)

A comprehensive comparative study of classical supervised ML models for predicting stroke occurrence based on healthcare-related features. This repository contains source code, model evaluations, and experimental results aimed at early stroke detection.

Link: https://stroke-prediction-ml-27.streamlit.app
---

## 📊 Summary of Model Performance

| **Model**               | **Best Hyperparameters**                                                                                                                      | **CV Accuracy** | **Test Accuracy** | **Precision (0 / 1)** | **Recall (0 / 1)** | **F1-Score (0 / 1)** | **Confusion Matrix**        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ----------------- | --------------------- | ------------------ | -------------------- | --------------------------- |
| **Logistic Regression** | `{'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'}`                                                                         | 0.7805          | 0.7525            | 1.00 / 0.06           | 0.75 / 0.81        | 0.86 / 0.11          | `[[8006, 2648], [37, 159]]` |
| **SVM**                 | `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}`                                                                                                | 0.8309          | 0.7621            | 0.99 / 0.05           | 0.76 / 0.69        | 0.86 / 0.09          | `[[8134, 2520], [61, 135]]` |
| **Naive Bayes**         | `{'var_smoothing': 1e-05}`                                                                                                                    | 0.7703          | 0.7345            | 0.99 / 0.05           | 0.73 / 0.77        | 0.84 / 0.09          | `[[7818, 2836], [45, 151]]` |
| **K-Nearest Neighbors** | `{'metric': 'euclidean', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}`                                                                    | 0.9370          | 0.8837            | 0.99 / 0.05           | 0.89 / 0.30        | 0.94 / 0.08          | `[[9530, 1124], [138, 58]]` |
| **Random Forest**       | `{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}`            | 0.8684          | 0.7939            | 0.99 / 0.05           | 0.80 / 0.59        | 0.88 / 0.09          | `[[8499, 2155], [81, 115]]` |
| **XGBoost**             | `{'colsample_bytree': 1.0, 'gamma': 0.3, 'learning_rate': 0.2, 'max_depth': 7, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.8}` | 0.9545          | **0.9469**        | 0.98 / 0.06           | 0.96 / 0.13        | 0.97 / 0.08          | `[[10249, 405], [171, 25]]` |


---

## ✅ Key Highlights

- 📌 Implemented and compared six classical models: **Logistic Regression, SVM, Naive Bayes, KNN, Random Forest, XGBoost**
- 🔍 Used **GridSearchCV** for hyperparameter tuning and **5-fold cross-validation**
- 📉 Tackled **class imbalance** problem in stroke prediction
- 📈 Performance measured using accuracy, precision, recall, F1-score, and confusion matrices

---

## 📖 Final Conclusion

This study provides a comprehensive comparative analysis of six classical supervised machine learning algorithms—Logistic Regression, Support Vector Machine (SVM), Naïve Bayes, k-Nearest Neighbors (k-NN), Random Forest, and XGBoost—for stroke prediction using a real-world healthcare dataset.

The evaluation was carried out using hyperparameter tuning with cross-validation and further validated on unseen test data. Each model's performance was assessed using multiple metrics, including accuracy, precision, recall, F1-score, and confusion matrices, providing a holistic view of their strengths and limitations.

📊 Key Findings

XGBoost achieved the highest test accuracy of 94.69%, showcasing its strong ability to model complex patterns in data. However, its recall for stroke cases remained low (13%), which is concerning in a healthcare context.

Random Forest also performed strongly with a test accuracy of 79.39%, offering better recall (59%) for the minority class, making it a balanced and reliable option.

k-NN showed impressive overall accuracy (88.37%), but suffered from low recall (30%) for stroke detection, indicating limited effectiveness on rare events.

Logistic Regression, with a test accuracy of 75.25%, delivered high recall (81%) for stroke cases despite extremely low precision (6%), making it useful for flagging potential cases even with noisy predictions.

Naïve Bayes achieved 73.45% accuracy with 77% recall, showing robustness in probabilistic classification even under class imbalance.

SVM offered 76.21% accuracy and 69% recall, balancing decision boundary precision with a moderate ability to detect stroke cases.

⚠️ Class Imbalance Consideration
A major limitation observed was the severe class imbalance in the dataset, where stroke cases form a very small fraction of the total data. This imbalance caused most models—especially XGBoost and KNN—to prioritize the majority class, resulting in poor recall for stroke detection.

To mitigate this, techniques like:

SMOTE (Synthetic Minority Oversampling Technique)
Class weighting
Cost-sensitive learning
were employed. These significantly affect model performance, particularly for real-world clinical deployment, where false negatives (missed strokes) can have life-threatening consequences.

✅ Final Remarks
While XGBoost stood out in terms of raw predictive accuracy, models like Logistic Regression and Random Forest provided better recall for the minority class, which is crucial in stroke prediction tasks. This study emphasizes that accuracy is not the sole metric of success in imbalanced healthcare datasets. Instead, recall and F1-score for stroke cases must be prioritized to ensure clinically actionable results.

In future work, combining these classical models with advanced imbalance-handling techniques or building ensemble systems that maximize both precision and recall could significantly enhance stroke prediction systems for real-world applications.

---

## 📦 Installation

```bash
git clone https://github.com/VivekVardhan275/stroke-prediction-ml.git
cd stroke-prediction-ml
pip install -r requirements.txt
