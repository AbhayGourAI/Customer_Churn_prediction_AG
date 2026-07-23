# Customer Churn Prediction

## Overview

This project predicts whether a telecom customer is likely to churn using various machine learning classification algorithms. The project includes complete data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, cross-validation, model evaluation, and deployment using Streamlit.

The objective is to identify customers who are likely to leave the telecom service, helping businesses improve customer retention through data-driven decision making.

---

## Features

- Data Cleaning and Preprocessing
- Handling Missing Values
- Outlier Detection and Treatment
- Feature Engineering
- Exploratory Data Analysis (10 Visualizations)
- Training Multiple Machine Learning Models
- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- Model Comparison using Multiple Evaluation Metrics
- Streamlit Web Application
- Model Serialization using Joblib

---

## Machine Learning Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score

Logistic Regression was selected as the final model because it achieved the best overall balance across the evaluation metrics.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── app.py
├── customer_churn_model.pkl
├── scaler.pkl
├── features.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/AbhayGourAI/Customer_Churn_prediction_AG.git
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## Results

The trained models were compared using Accuracy, Precision, Recall, F1-score, and ROC-AUC. Logistic Regression achieved the best overall performance and was selected as the final model for deployment.

---

## Author

**Abhay Gour**
