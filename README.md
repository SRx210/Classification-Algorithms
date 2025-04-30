# Classification Algorithms in Machine Learning

## Overview

Classification is a type of supervised learning in machine learning where the goal is to predict a **categorical label** for a given input based on prior data. It’s used in many real-world applications like spam detection, medical diagnosis, sentiment analysis, and more.

This document provides an overview of common classification algorithms, their working principles, advantages, disadvantages, and typical use cases.

---

## What is Supervised Learning?

Supervised learning is a machine learning approach where a model is trained on a labeled dataset. This means the algorithm is given input data along with the correct output (label), and it learns to map inputs to outputs.

Supervised learning is broadly categorized into:
- **Classification** – predicting categories or classes (e.g., "spam" vs "not spam")
- **Regression** – predicting continuous values (e.g., house prices)

---

## Common Classification Algorithms

### 1. Naive Bayes Classifier
- **How it works**: Based on Bayes' Theorem with a strong assumption of independence between features.
- **Variants**: Gaussian, Multinomial, Bernoulli
- **Advantages**:
  - Fast and efficient
  - Performs well on high-dimensional data
- **Disadvantages**:
  - Assumes feature independence, which is rarely true in real-world data
- **Use cases**: Spam filtering, document classification, sentiment analysis

---

### 2. Decision Tree
- **How it works**: Splits data based on feature values to build a tree of decisions.
- **Advantages**:
  - Easy to understand and interpret
  - Works for both numerical and categorical data
- **Disadvantages**:
  - Prone to overfitting
- **Use cases**: Risk assessment, medical diagnosis

---

### 3. Random Forest
- **How it works**: An ensemble of multiple decision trees combined using averaging or voting.
- **Advantages**:
  - Reduces overfitting
  - Handles missing values and large datasets well
- **Disadvantages**:
  - More complex and slower than a single decision tree
- **Use cases**: Fraud detection, customer segmentation

---

### 4. Support Vector Machine (SVM)
- **How it works**: Finds the optimal hyperplane that separates classes in feature space.
- **Advantages**:
  - Works well in high-dimensional spaces
  - Effective when the number of dimensions is greater than the number of samples
- **Disadvantages**:
  - Not suitable for very large datasets
  - Requires careful tuning of parameters
- **Use cases**: Image classification, handwriting recognition

---

### 5. K-Nearest Neighbors (KNN)
- **How it works**: Classifies data based on the majority class of its 'k' nearest neighbors.
- **Advantages**:
  - Simple and easy to implement
  - No training phase
- **Disadvantages**:
  - Slow prediction time on large datasets
  - Sensitive to irrelevant features and scaling
- **Use cases**: Recommendation systems, pattern recognition

---

### 6. Logistic Regression
- **How it works**: Models the probability that an input belongs to a particular class using the logistic function.
- **Advantages**:
  - Interpretable and easy to implement
  - Good baseline for binary classification
- **Disadvantages**:
  - Assumes a linear relationship between input and log-odds
- **Use cases**: Binary classification problems such as churn prediction or email classification

---

## Evaluation Metrics for Classification

To evaluate classification models, several metrics are used:
- **Accuracy** – Percentage of correct predictions
- **Precision and Recall** – Especially important for imbalanced datasets
- **F1 Score** – Harmonic mean of precision and recall
- **Confusion Matrix** – Summary of prediction results
- **ROC-AUC Curve** – Trade-off between true positive rate and false positive rate

---

## Getting Started with Python

You can implement these algorithms using libraries like scikit-learn:

```bash
pip install scikit-learn
```

**Example**
```
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

~SRx210
