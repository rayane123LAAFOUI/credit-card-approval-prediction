
# ============================================================
# CREDIT CARD APPROVAL PREDICTION
# Logistic Regression + GridSearchCV
# ============================================================


# 1. Import necessary libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


# 2. Load the dataset

cc_apps = pd.read_csv("data/cc_approvals.data", header=None)

print("First observations:")
print(cc_apps.head())


# 3. Explore dataset structure

print("\nDataset shape:", cc_apps.shape)

print("\nData types:")
print(cc_apps.dtypes)

print("\nStatistical summary:")
print(cc_apps.describe(include="all"))


# 4. Identify target variable

target_col = cc_apps.columns[-1]

print("\nTarget variable:", target_col)

print("\nTarget distribution:")
print(cc_apps[target_col].value_counts(dropna=False))


# 5. Identify and handle missing values

print("\nMissing values represented by '?':")
print((cc_apps == '?').sum())

# Replace '?' with NaN
cc_apps = cc_apps.replace('?', np.nan)

print("\nMissing values after replacement:")
print(cc_apps.isna().sum())

# Calculate missing value percentages
missing_summary = pd.DataFrame({
    'Missing_Count': cc_apps.isna().sum(),
    'Missing_Percentage': cc_apps.isna().mean() * 100
})

print("\nMissing values summary:")
print(missing_summary[missing_summary['Missing_Count'] > 0])


# 6. Data cleaning

# Drop observations containing missing values
# in the selected columns

cc_apps = cc_apps.dropna(subset=[0, 1, 3, 4, 5, 6])

print("\nDataset shape after cleaning:", cc_apps.shape)

print("\nRemaining missing values:")
print(cc_apps.isna().sum().sort_values())

print("\nTarget distribution after cleaning:")
print(cc_apps[target_col].value_counts(normalize=True))


# 7. Separate features and target

X = cc_apps.drop(columns=[target_col])
y = cc_apps[target_col]

# Convert column names to strings
X.columns = X.columns.astype(str)


# 8. Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. Encode categorical variables using One-Hot Encoding
# with pandas get_dummies()

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Align test features with training features
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Same columns:", X_train.columns.equals(X_test.columns))


# 10. Build the Machine Learning pipeline

steps = [
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
]

pipeline = Pipeline(steps)


# 11. Hyperparameter tuning using GridSearchCV

parameters = {
    'logreg__C': [0.01, 0.1, 1, 10]
}

cv = GridSearchCV(
    pipeline,
    param_grid=parameters,
    cv=5
)

cv.fit(X_train, y_train)


# 12. Make predictions

y_pred = cv.predict(X_test)


# 13. Display the best model results

best_score = cv.best_score_
best_parameters = cv.best_params_

print("\nBest cross-validation score:", best_score)
print("Best parameters:", best_parameters)


# 14. Model evaluation

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    y_pred,
    labels=['+', '-']
))


print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=['+', '-'],
    target_names=['Approved (+)', 'Rejected (-)']
))
