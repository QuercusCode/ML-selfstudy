# Week 20: Pipelines, Leakage, Feature Engineering, & Block Project

> **Goal:** Synthesize your classical machine learning skills. You will master scikit-learn's `Pipeline` and `ColumnTransformer` frameworks to construct leak-free workflows, implement feature engineering techniques, and begin your first comprehensive end-of-block portfolio project.

---

## Part 1: Preprocessing and Feature Engineering

Raw data is rarely ready for model training. We must clean, transform, and encode features.

### 1.1 Types of Transformations
- **Numerical Scaling:** Standardization ($z$-score) or Min-Max scaling.
- **Handling Missing Values:** Imputing missing values using the mean, median, or constant values (e.g., `SimpleImputer`).
- **Categorical Encoding:** Converting categories into numbers.
  - **One-Hot Encoding:** Creates binary columns for each category (suitable for nominal data without order).
  - **Ordinal Encoding:** Maps categories to sorted integers (suitable for ordered data like grades).
- **Polynomial Features:** Creating interaction terms (e.g., $x_1 \times x_2$, $x_1^2$) to allow linear models to fit non-linear relationships.

---

## Part 2: Scikit-Learn Pipeline Architecture

If we apply preprocessing transformations to our whole dataset before splitting or cross-validating, we cause **data leakage**. For example, computing standard deviation on the entire dataset leaks validation set properties into training.

Scikit-learn provides `Pipeline` and `ColumnTransformer` to automate preprocessing, ensuring that all transformations are fitted *only* on the training splits during cross-validation.

### 2.1 ColumnTransformer
We use `ColumnTransformer` to apply different preprocessing steps to different subsets of columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_features = ["embarked", "sex"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
```

### 2.2 Combining Preprocessor and Model into a Pipeline
We append the classifier or regressor to the preprocessing block. The resulting `Pipeline` object exposes the standard `fit` and `predict` API, encapsulating all transformations:

```python
from sklearn.ensemble import RandomForestClassifier

clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])
```

When you call `clf_pipeline.fit(X_train, y_train)`, it fits the preprocessing transformers, applies the transformations, and trains the classifier on the transformed data.

When you call `clf_pipeline.predict(X_val)`, it applies the fitted transformations to the validation data and passes the results to the classifier's prediction method.

---

## Part 3: Cross-Validated Hyperparameter Sweeps

We can pass the entire pipeline directly into `GridSearchCV` or `RandomizedSearchCV` to search for hyperparameters of both the preprocessor and the model:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10]
}

grid_search = GridSearchCV(clf_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

---

## Part 4: Block Project Guidelines

To complete the Classical Machine Learning block, you will build a comprehensive, portfolio-ready project on a dataset of your choice.

### 4.1 Recommended Datasets
- **Biological:** Peptide toxicity prediction, gene expression classification, or secondary structure propensity from sequence-derived features.
- **Kaggle Tabular:** Titanic (survival classification) or House Prices (regression).

### 4.2 Project Requirements
1. **Exploratory Data Analysis (EDA):** Diagnose missing data, distribution shapes, and pairwise feature correlations.
2. **Leak-Free Preprocessing:** Build a complete scikit-learn `Pipeline` utilizing `ColumnTransformer`.
3. **Model Sweep:** Tune and compare at least two different model classes (e.g., Random Forest vs. XGBoost) using `GridSearchCV`.
4. **Honest Evaluation:** Report the final metrics (Precision, Recall, F1, ROC-AUC for classification; RMSE, $R^2$ for regression) on a held-out test set that was never touched during training or validation.

---

## Practice Exercises

### Exercise 1: ColumnTransformer Implementation
1. Generate a toy DataFrame containing:
   - 2 numeric columns with missing values.
   - 2 categorical columns with string labels.
2. **Code it:** Construct a `ColumnTransformer` that imputes and standardizes the numeric columns, and one-hot encodes the categorical columns. Print the output shape and verify that no NaNs remain.

### Exercise 2: Grid Sweep on Pipeline
1. Load a standard classification dataset.
2. Create a pipeline containing your `ColumnTransformer` and an `XGBClassifier`.
3. **Code it:** Write a grid sweep that optimizes the imputer strategy and the XGBoost `learning_rate` simultaneously, showing the best score and parameters.

---

## Self-Test Questions

1. **Why is it critical to fit transformers inside CV loops?** *(If we fit them on the whole dataset first, properties of the validation fold (like its mean or list of categories) leak into the training process, leading to optimistic validation scores that fail to generalize.)*
2. **What does `handle_unknown="ignore"` do in OneHotEncoder?** *(It ensures that if the test set contains a category that was not present in the training set, the encoder will output all zeros for that feature instead of raising an error.)*
3. **What is the difference between GridSearchCV and RandomizedSearchCV?** *(GridSearchCV evaluates every combination in a specified grid of parameters, which is computationally expensive. RandomizedSearchCV samples a fixed number of parameter settings from specified distributions, which is much faster for large grids.)*
4. **How do you verify if a pipeline successfully prevented data leakage?** *(By checking that no step in the preprocessing pipeline called `fit` or `fit_transform` on the validation/test set, and verifying that the final test performance matches the cross-validation score.)*
