# 🧠 Stroke Prediction Using Classical Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)

A comprehensive comparative study of classical supervised ML models for predicting stroke occurrence based on healthcare-related features. This repository contains source code, model evaluations, and experimental results aimed at early stroke detection.

---

## 📊 Summary of Model Performance

| **Model**               | **Best Hyperparameters**                                                                                                                      | **CV Accuracy** | **Test Accuracy** | **Precision (0 / 1)** | **Recall (0 / 1)** | **F1-Score (0 / 1)** | **Confusion Matrix**        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ----------------- | --------------------- | ------------------ | -------------------- | --------------------------- |
| **Logistic Regression** | `{'C': 100, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}`                                                                              | 0.7806          | 0.7528            | 1.00 / 0.06           | 0.75 / 0.81        | 0.86 / 0.11          | `[[8009, 2645], [37, 159]]` |
| **SVM**                 | `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}`                                                                                                | 0.8354          | 0.7721            | 0.99 / 0.05           | 0.77 / 0.66        | 0.87 / 0.09          | `[[8248, 2406], [67, 129]]` |
| **Naive Bayes**         | `{'var_smoothing': 1e-09}`                                                                                                                    | 0.7665          | 0.7365            | 0.99 / 0.05           | 0.74 / 0.76        | 0.85 / 0.09          | `[[7842, 2812], [47, 149]]` |
| **K-Nearest Neighbors** | `{'metric': 'euclidean', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}`                                                                    | 0.9386          | 0.8850            | 0.99 / 0.05           | 0.90 / 0.30        | 0.94 / 0.09          | `[[9544, 1110], [138, 58]]` |
| **Random Forest**       | `{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}`            | 0.8703          | 0.7977            | 0.99 / 0.05           | 0.80 / 0.60        | 0.89 / 0.10          | `[[8537, 2117], [78, 118]]` |
| **XGBoost**             | `{'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.2, 'max_depth': 7, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.8}` | 0.9573          | **0.9465**        | 0.98 / 0.05           | 0.96 / 0.11        | 0.97 / 0.07          | `[[10248, 406], [174, 22]]` |


---

## ✅ Key Highlights

- 📌 Implemented and compared six classical models: **Logistic Regression, SVM, Naive Bayes, KNN, Random Forest, XGBoost**
- 🔍 Used **GridSearchCV** for hyperparameter tuning and **5-fold cross-validation**
- 📉 Tackled **class imbalance** problem in stroke prediction
- 📈 Performance measured using accuracy, precision, recall, F1-score, and confusion matrices

---

## 📖 Final Conclusion

## Conclusion

This study provides a comprehensive comparative analysis of six classical supervised machine learning algorithms—Logistic Regression, Support Vector Machine (SVM), Naïve Bayes, k-Nearest Neighbors (k-NN), Random Forest, and XGBoost—for stroke prediction using a real-world healthcare dataset.

The evaluation was carried out using hyperparameter tuning with cross-validation and further validated on unseen test data. Each model's performance was assessed using multiple metrics, including accuracy, precision, recall, F1-score, and confusion matrices, providing a holistic view of their strengths and limitations.

📊 Key Findings

XGBoost achieved the highest test accuracy of 94.65%, showcasing its strong ability to model complex patterns in data. However, its recall for stroke cases remained low (11%), which is concerning in a healthcare context.

Random Forest also performed strongly with a test accuracy of 79.76%, offering better recall (60%) for the minority class, making it a balanced and reliable option.

k-NN showed impressive overall accuracy (88.50%), but suffered from low recall (30%) for stroke detection, indicating limited effectiveness on rare events.

Logistic Regression, with a test accuracy of 75.28%, delivered high recall (81%) for stroke cases despite extremely low precision (6%), making it useful for flagging potential cases even with noisy predictions.

Naïve Bayes achieved 73.65% accuracy with 76% recall, showing robustness in probabilistic classification even under class imbalance.

SVM offered 77.21% accuracy and 66% recall, balancing decision boundary precision with a moderate ability to detect stroke cases.

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
