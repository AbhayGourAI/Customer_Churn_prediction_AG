# -*- coding: utf-8 -*-
"""Customer_Churn_Notebook.ipynb"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()

df.info()

df.shape

df.describe()

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.isnull().sum()

imputer = SimpleImputer(strategy="median")
df["TotalCharges"] = imputer.fit_transform(df[["TotalCharges"]])

df.isnull().sum()

num_cols = ["tenure","MonthlyCharges","TotalCharges"]

for col in num_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3-Q1

    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR

    df[col] = np.where(df[col]<lower, lower, df[col])
    df[col] = np.where(df[col]>upper, upper, df[col])

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

sns.countplot(x="gender", hue="Churn", data=df)
plt.show()

sns.countplot(x="Contract", hue="Churn", data=df)
plt.xticks(rotation=40)
plt.show()

sns.countplot(x="InternetService", hue="Churn", data=df)
plt.show()

sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.xticks(rotation=30)
plt.show()

sns.histplot(df["tenure"], kde=True)
plt.show()

sns.histplot(df["MonthlyCharges"], kde=True)
plt.show()

sns.boxplot(y=df["TotalCharges"])
plt.show()

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.show()

temp = df.copy()

le = LabelEncoder()

for col in temp.columns:

    if temp[col].dtype=="object":
        temp[col]=le.fit_transform(temp[col])

plt.figure(figsize=(15,10))

sns.heatmap(temp.corr(), cmap="coolwarm")

plt.show()

binary_cols = ["gender",
               "Partner",
               "Dependents",
               "PhoneService",
               "PaperlessBilling",
               "Churn"]

le = LabelEncoder()

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)

y = df["Churn"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(),

    "Gradient Boosting": GradientBoostingClassifier(),

    "SVM": SVC(probability=True)
}

params = {

    "n_estimators":[100,200],

    "max_depth":[5,10,None],

    "min_samples_split":[2,5]
}

grid = GridSearchCV(

    RandomForestClassifier(),

    param_grid=params,

    cv=5,

    scoring="f1",

    n_jobs=-1
)

grid.fit(X_train,y_train)

best_rf = grid.best_estimator_

print(grid.best_params_)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']
}

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_lr.fit(X_train, y_train)

print(grid_lr.best_params_)
best_lr = grid_lr.best_estimator_

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_score = cross_val_score(

    best_rf,

    X_scaled,

    y,

    cv=cv,

    scoring="accuracy"
)

print(cv_score.mean())

results=[]

for name, model in models.items():

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    prob=model.predict_proba(X_test)[:,1]

    results.append({

        "Model":name,

        "Accuracy":accuracy_score(y_test,pred),

        "Precision":precision_score(y_test,pred),

        "Recall":recall_score(y_test,pred),

        "F1":f1_score(y_test,pred),

        "ROC_AUC":roc_auc_score(y_test,prob)
    })

results=pd.DataFrame(results)

results

pred = best_rf.predict(X_test)

prob = best_rf.predict_proba(X_test)[:,1]

print(classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

RocCurveDisplay.from_estimator(best_rf,X_test,y_test)

plt.show()

import joblib

# Save model
joblib.dump(best_lr, "customer_churn_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

joblib.dump(lr, "customer_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

joblib.dump(X.columns.tolist(), "features.pkl")

