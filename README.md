# Predicting the likelihood of a patient developing heart disease
![Workflow](/fig/workflow.png)

## Overview

In this project, we will apply basic machine learning methods to predict whether a person is likely to develop heart disease based on the Cleveland Heart Disease dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets). 

The Cleveland dataset comprises 14 features including age, gender, chest pain type, resting blood pressure, serum cholesterol level, fasting blood sugar, resting ECG result, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, peak exercise ST segment, number of major vessels (0-3) colored by fluoroscopy, thalassemia status, and heart disease diagnosis (0 representing no disease and 1, 2, 3, 4 representing varying degrees of disease). The Cleveland dataset consists of 303 samples with 14 features.

## Results:
![Results](/fig/result.png)

| id | Algorithm       | Value |
|----|-----------------|-------|
| 1  | Decision Tree   | 75    |
| 2  | Random Forest   | 80    |
| 3  | SVM             | 67    |
| 4  | KNN             | 69    |
| 5  | Naive Bayes     | 84    |
| 6  | AdaBoost        | 84    |
| 7  | GradientBoost   | 85    |
| 8  | XGBoost         | 84    |
| 9  | **Ensemble**    | **90**   |


