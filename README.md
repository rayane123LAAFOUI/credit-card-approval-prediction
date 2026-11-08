# credit-card-approval-prediction
Credit card approval prediction using machine learning and logistic regression.


## Project Overview

This project focuses on predicting credit card application approvals using Machine Learning techniques.

The objective is to build a classification model capable of predicting whether a credit card application should be approved or rejected based on applicant characteristics.

The project follows a complete Machine Learning workflow, including data exploration, preprocessing, model development, hyperparameter tuning, and performance evaluation.

## Business Problem

Financial institutions receive numerous credit card applications and need to assess applicants efficiently.

Automating part of this assessment process can help identify patterns in application data and support more consistent decision-making.

This project explores how Logistic Regression can be used to predict credit card approval outcomes.

## Dataset

The project uses the Credit Card Approval dataset, commonly known as the UCI Credit Approval dataset.

- Initial dataset: 690 observations.
- Original structure used: 14 columns, including the target variable.
- Target variable: Credit card application status (`+` for approved and `-` for rejected).
- Missing values are represented by `?`.

The dataset is not included in this repository. Please obtain it from the original source and place the file in the `data/` directory.

## Methodology

### 1. Data Exploration
- Examined dataset dimensions and data types.
- Generated descriptive statistics.
- Analyzed the distribution of the target variable.
- Identified missing values.

### 2. Data Cleaning
- Replaced missing value markers (`?`) with NaN.
- Removed observations containing missing values in the selected features.
- Retained 659 observations, representing approximately 95.5% of the original dataset.

### 3. Data Preprocessing
- Separated explanatory variables from the target variable.
- Applied a stratified train-test split (80% training, 20% testing).
- Used One-Hot Encoding through `pandas.get_dummies()` to transform categorical variables.
- Aligned training and testing features to ensure consistent input dimensions.
- Applied feature standardization using `StandardScaler`.

### 4. Model Development
- Implemented a Logistic Regression classifier.
- Built a preprocessing and modeling pipeline using Scikit-learn.
- Applied GridSearchCV with 5-fold cross-validation to optimize the regularization parameter `C`.

### 5. Model Evaluation

The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross-validation score

## Results

The optimized Logistic Regression model achieved the following results:

| Metric | Result |
|---|---:|
| Cross-validation accuracy | 84.25% |
| Test accuracy | 81.06% |
| Best regularization parameter (C) | 0.1 |
| Macro F1-score | 81% |

### Confusion Matrix

| | Predicted Approved (+) | Predicted Rejected (-) |
|---|---:|---:|
| Actual Approved (+) | 48 | 12 |
| Actual Rejected (-) | 13 | 59 |

The model correctly classified 107 out of 132 applications in the test set.

## Key Findings

- The target classes are relatively balanced, with approximately 45.2% approved applications and 54.8% rejected applications after data cleaning.
- Logistic Regression achieved an accuracy of approximately 81% on unseen test data.
- The best regularization parameter identified through GridSearchCV was `C = 0.1`.
- The model demonstrated relatively balanced precision and recall across both classes.

## Limitations and Future Improvements

Potential improvements include:

- Comparing Logistic Regression with other classification algorithms.
- Exploring alternative missing-value imputation strategies.
- Improving categorical and numerical feature preprocessing.
- Investigating feature importance and model interpretability.
- Evaluating additional performance metrics and classification thresholds.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Machine Learning
- Logistic Regression
- GridSearchCV

## Author

**Data Science & Statistics Student**

This project was developed as part of my Machine Learning learning journey.
