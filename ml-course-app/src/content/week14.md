# Week 15: Regularization, Bias–Variance, & Cross-Validation

> **Goal:** Develop the discipline required to train models that generalize. You will learn the structural definition of the bias–variance tradeoff, implement Ridge and Lasso regularization, understand why Lasso induces sparsity, and build a robust cross-validation pipeline from scratch to prevent data leakage.

---

## Part 1: Bias–Variance Decomposition

When training a machine learning model, we aim to minimize generalization error. The total error of a model on unseen data can be decomposed into three mathematically distinct components:

$$ \text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $$

1. **Bias:** Error due to erroneous or overly simplistic assumptions in the model. High bias leads to **underfitting** (the model is not flexible enough to capture the true underlying pattern).
2. **Variance:** Error due to sensitivity to small fluctuations in the training set. High variance leads to **overfitting** (the model fits the noise in the training data rather than the underlying distribution).
3. **Irreducible Error ($\sigma^2$):** Noise inherent to the problem itself, which no model can overcome.

### 1.1 The Tradeoff Curve
As model complexity increases:
- **Bias decreases** (the model can fit more complex shapes).
- **Variance increases** (the model becomes sensitive to individual training points).
- The validation error reaches a minimum at the optimal complexity level, after which it begins to rise as variance dominates.

---

## Part 2: Regularization — Ridge and Lasso

To combat high variance (overfitting), we can constrain the model parameters by adding a penalty term to our cost function. This is called **regularization**.

### 2.1 Ridge Regression ($L_2$ Regularization)
Ridge regression adds a penalty proportional to the sum of the squares of the weights:

$$ J(\mathbf{\theta}) = J_{\text{MSE}}(\mathbf{\theta}) + \lambda \sum_{j=1}^d \theta_j^2 = J_{\text{MSE}}(\mathbf{\theta}) + \lambda \|\mathbf{\theta}\|_2^2 $$

- The hyperparameter $\lambda \geq 0$ controls the strength of the penalty. If $\lambda = 0$, it reverts to standard OLS.
- $L_2$ regularization penalizes large weights heavily, forcing them to shrink towards zero, but it rarely makes them exactly zero.

### 2.2 Lasso Regression ($L_1$ Regularization)
Lasso (Least Absolute Shrinkage and Selection Operator) regression adds a penalty proportional to the sum of the absolute values of the weights:

$$ J(\mathbf{\theta}) = J_{\text{MSE}}(\mathbf{\theta}) + \lambda \sum_{j=1}^d |\theta_j| = J_{\text{MSE}}(\mathbf{\theta}) + \lambda \|\mathbf{\theta}\|_1 $$

- Unlike Ridge, Lasso has a unique property: it can shrink weights **exactly to zero**, effectively performing automatic feature selection.

### 2.3 Why Lasso Induces Sparsity (The Geometric Intuition)
Consider a 2D parameter space $(\theta_1, \theta_2)$. The regularization term sets a constraint boundary:
- For Ridge ($L_2$), the constraint is a circle: $\theta_1^2 + \theta_2^2 \leq t$.
- For Lasso ($L_1$), the constraint is a diamond: $|\theta_1| + |\theta_2| \leq t$.

The loss contours of the OLS cost expand outwards from the unregularized minimum. The regularized solution is the point where an OLS contour first touches the constraint boundary.
- Because the Lasso boundary has sharp corners located directly on the axes (where one parameter is exactly zero), the expanding elliptical loss contours are highly likely to touch the diamond at one of these corners, setting parameters to zero.
- The smooth circle of the Ridge constraint makes a tangent contact off the axes almost certain, keeping both values small but non-zero.

---

## Part 3: Cross-Validation & Data Leakage

Evaluating a model on the same data used to train it yields an overly optimistic estimate of performance. We must keep test data completely isolated.

### 3.1 K-Fold Cross-Validation
To maximize data usage for validation:
1. Split the dataset randomly into $K$ equal-sized folds.
2. For each fold $k \in \{1, \dots, K\}$:
   - Train the model on the other $K-1$ folds.
   - Evaluate the model on fold $k$ and record the score.
3. Compute the average validation score across all $K$ iterations.

```python
import numpy as np

def k_fold_split(X, y, k=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    return folds
```

### 3.2 Data Leakage
Data leakage occurs when information from outside the training dataset is used to train the model. This leads to high validation scores but poor generalization in production.
- **Example:** Scaling features (e.g., computing mean and std) on the *entire* dataset before splitting. The mean and std contain information from the validation fold, leaking future information into training.
- **Fix:** Always fit transformers (like scalers) only on the training folds, then apply the transformation to the validation/test folds.

---

## Part 4: Practice Exercises

### Exercise 1: K-Fold CV from Scratch
1. Generate synthetic data with 200 samples.
2. Implement $K$-fold cross-validation from scratch. Do not use `sklearn.model_selection`.
3. **Code it:** Return the average mean squared error across 5 folds for a linear regression model.

### Exercise 2: Ridge vs Lasso Coefficient Paths
1. Generate a dataset with 10 features, where only 3 features are actually correlated with the target $y$.
2. Train Ridge and Lasso models over a grid of regularization strengths ($\lambda \in [10^{-4}, 10^2]$).
3. **Code it:** Plot the values of the 10 coefficients as a function of $\lambda$ for both Ridge and Lasso. Observe how Lasso drops coefficients to zero while Ridge shrinks them asymptotically.

---

## Self-Test Questions

1. **How does adding regularization affect the bias–variance profile of a model?** *(Regularization limits parameter values, making the model less flexible. This increases bias slightly but reduces variance significantly, often leading to better overall generalization.)*
2. **If Lasso performs automatic feature selection, why would we ever use Ridge?** *(If we have many features that all have small, real effects on the target, Ridge will perform better because it keeps all features. Lasso will arbitrarily zero out useful predictors to satisfy the sparsity constraint.)*
3. **Why is cross-validation better than a single train-test split?** *(Cross-validation reduces the variance of our performance estimate. If a single train-test split happens to put outliers in the test set, the score will be unrepresentative. CV averages out these anomalies.)*
4. **How do you avoid data leakage during cross-validation when doing feature normalization?** *(By fitting the scaler on the training folds only, then transforming both the training folds and validation fold. Never fit standardizers on the whole dataset before cross-validation.)*
