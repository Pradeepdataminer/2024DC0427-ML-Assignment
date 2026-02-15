# MTech-ML-Assignment2
Machine Learning Assignment – 2

## Problem Statement The task involves implementing various supervised learning algorithms, comparing their performance using standard evaluation metrics, and deploying the models through an interactive Streamlit web application. This project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, UI development, and deployment.

## Dataset Description

Dataset characteristics:

Problem Type: Binary Classification

Target Variable: Credit card default (0 = No Default, 1 = Default)

Number of Instances: More than 500 records

Number of Features: More than 12 features

Feature Types: Numerical and categorical attributes related to customer demographics, credit history, and payment behavior

The dataset was preprocessed by handling missing values, encoding categorical variables, and applying feature scaling where required.

## Models Used and Evaluation Metrics

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (kNN)

Naive Bayes Classifier

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

## Comparison Table of Evaluation Metrics

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|:--------:|:---:|:---------:|:------:|:--------:|:---:|
| Logistic Regression | 0.7790 | 0.7109 | 0.5032 | 0.0588 | 0.1053 | 0.1107 |
| Decision Tree | 0.7292 | 0.6154 | 0.3928 | 0.4115 | 0.4019 | 0.2271 |
| kNN | 0.7802 | 0.6820 | 0.5050 | 0.3052 | 0.3805 | 0.2686 |
| Naive Bayes | 0.5505 | 0.6600 | 0.2949 | 0.7423 | 0.4221 | 0.1991 |
| Random Forest (Ensemble) | 0.8113 | 0.7590 | 0.6349 | 0.3459 | 0.4478 | 0.3689 |
| XGBoost (Ensemble) | 0.8095 | 0.7609 | 0.6179 | 0.3632 | 0.4575 | 0.3696 |
## Model Performance Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Shows good overall accuracy but very low recall, indicating poor detection of default cases due to class imbalance. |
| Decision Tree | Provides balanced performance with moderate recall and interpretability, but slightly lower accuracy than ensemble models. |
| kNN | Achieves reasonable accuracy but struggles with recall, making it less effective for detecting minority default cases. |
| Naive Bayes | Has the highest recall among all models, making it good for detecting defaulters, but suffers from low accuracy and precision. |
| Random Forest (Ensemble) | Performs strongly across all metrics with high accuracy, AUC, and MCC, demonstrating robustness and reduced overfitting. |
| XGBoost (Ensemble) | Achieves the best overall balance between accuracy, AUC, F1 score, and MCC, making it the most effective model for this dataset. |
## Conclusion 
Naive Bayes is effective for recall-focused applications, while Logistic Regression provides a simple baseline model. The Streamlit deployment enables interactive exploration and real-time evaluation of all models.
