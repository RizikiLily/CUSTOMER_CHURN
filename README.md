# Customer Churn Prediction Using Machine Learning

## Overview

Customer churn prediction is a key business problem aimed at identifying customers who are likely to discontinue a service. This project builds and evaluates machine learning models to predict customer churn while emphasizing **proper validation, leakage prevention, and model interpretability**.

A Random Forest classifier implemented within a scikit-learn Pipeline is used as the final model after experimentation and validation.

---

## Dataset Description

Each row represents a customer with the following features:

* **CustomerID** – Unique identifier for each customer
* **Age** – Age of the customer
* **Gender** – Gender of the customer
* **Tenure** – Number of months the customer has been with the company
* **Usage Frequency** – Number of service uses in the last month
* **Support Calls** – Number of customer support calls in the last month
* **Payment Delay** – Number of days payment was delayed in the last month
* **Subscription Type** – Type of subscription chosen
* **Contract Length** – Duration of the contract
* **Total Spend** – Total amount spent by the customer
* **Last Interaction** – Days since the last interaction with the company
* **Churn (Target)** – Whether the customer churned (1) or not (0)

---

## Project Structure

```
├── Customer_Churn.ipynb
├── README.md
└── data/
    └── train_data.csv
```

---

## Methodology

### 1. Data Preprocessing

* Numerical features were scaled using `StandardScaler`
* Categorical variables were encoded using `OneHotEncoder`
* All preprocessing was performed inside a **Pipeline** to prevent data leakage
* A stratified train–test split was used to preserve class distribution

---

### 2. Model Training

Two models were trained and compared:

* Logistic Regression (baseline, interpretable model)
* Random Forest Classifier (non-linear, high-performance model)

---

### 3. Cross-Validation

* Stratified K-Fold cross-validation was applied
* ROC-AUC was used as the primary evaluation metric
* This ensured performance stability and protected against overfitting

---

### 4. Data Leakage Investigation

Initial experiments produced near-perfect ROC-AUC scores (~0.9999), raising concerns of possible data leakage.

Actions taken:

* Reviewed feature definitions and temporal relationships
* Removed features suspected of leaking target information
* Retrained all models from scratch

Post-leakage results:

* Test ROC-AUC ≈ 0.984
* Cross-validation ROC-AUC ≈ 0.985 with low variance

This confirmed the model’s performance is realistic and generalizes well.

---

### 5. Hyperparameter Tuning

* RandomizedSearchCV was used to tune the Random Forest model
* ROC-AUC was optimized
* Performance gains were marginal, indicating the baseline model was already well-calibrated

---

### 6. Feature Importance Analysis

Feature importance analysis from the final Random Forest model showed:

* Support Calls and Total Spend are the strongest churn predictors
* Contract Length and Payment Delay have moderate influence
* Demographic features contribute less compared to behavioral features

This provides actionable insights for customer retention strategies.

---

### 7. Threshold Analysis

Different probability thresholds were evaluated to analyze precision–recall trade-offs.
This allows the model to be adapted to different business objectives, such as prioritizing churn prevention or minimizing false positives.

---

## Final Model Performance

* Test ROC-AUC: ~0.984
* Cross-validation mean ROC-AUC: ~0.985
* Low variance across folds indicates strong generalization

---

## Deployment Considerations

* The full preprocessing and model pipeline can be serialized using `joblib`
* Ensures consistent preprocessing during inference

---

## Key Takeaways

* Extremely high performance metrics can indicate data leakage
* Cross-validation is essential for trustworthy evaluation
* Pipelines are critical for reproducibility and deployment
* Feature importance bridges model performance and business understanding

---

## Technologies Used

* Python
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* Jupyter Notebook
* Google colab

---

## Author

**Lillian Riziki**
Engineer & Aspiring Data Scientist

