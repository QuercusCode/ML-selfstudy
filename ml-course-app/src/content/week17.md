# Week 18: Gradient Boosting & XGBoost

> **Goal:** Master the concepts behind Gradient Boosting. You will understand how sequential boosting ensembles work, mathematically trace how gradient boosting fits trees to residuals (gradients of loss), learn the key enhancements of XGBoost, and build a pipeline to tune hyperparameters.

---

## Part 1: Boosting vs. Bagging

While Bagging (e.g., Random Forests) builds independent, deep trees in parallel to reduce variance, **Boosting** builds shallow trees sequentially to reduce bias.

```
Bagging:   [Bootstrapped Data 1] -> [Deep Tree 1] ---\
           [Bootstrapped Data 2] -> [Deep Tree 2] ----+--> [Average / Vote]
           [Bootstrapped Data 3] -> [Deep Tree 3] ---/

Boosting:  [Data] -> [Tree 1] -> [Predict residuals] -> [Tree 2] -> [Predict residuals] -> [Tree 3] -> Final Ensemble
```

In boosting:
- Each new base model is trained to correct the errors made by the previous models.
- Base learners are typically **weak learners** (e.g., shallow decision trees or "decision stumps" with a depth of 1 or 2).

---

## Part 2: The Gradient Boosting Algorithm

Gradient Boosting formulates boosting as a numerical optimization problem where we minimize the loss function by adding weak learners using gradient descent.

Let $L(y, \hat{y})$ be our differentiable loss function. The algorithm proceeds as follows:

### 2.1 Initialization
Initialize the model with a constant value (typically the mean for regression):

$$ F_0(\mathbf{x}) = \arg\min_{\gamma} \sum_{i=1}^m L(y^{(i)}, \gamma) $$

### 2.2 The Boosting Loop
For $t = 1$ to $T$ (where $T$ is the number of trees):
1. **Compute Pseudo-Residuals:** Calculate the negative gradient of the loss function with respect to the current model's predictions. These act as the training targets for the next tree:
   $$ r_{it} = -\left[ \frac{\partial L(y^{(i)}, F(\mathbf{x}^{(i)}))}{\partial F(\mathbf{x}^{(i)})} \right]_{F(\mathbf{x}) = F_{t-1}(\mathbf{x})} $$
   *(Note: For MSE Loss $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$, the pseudo-residual is exactly the prediction error $y^{(i)} - F_{t-1}(\mathbf{x}^{(i)})$).*

2. **Fit Weak Learner:** Train a decision tree $h_t(\mathbf{x})$ to predict the pseudo-residuals $r_{it}$.

3. **Update Model:** Add the new tree to our prediction with a learning rate $\eta \in (0, 1]$ (also called **shrinkage**):
   $$ F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \eta h_t(\mathbf{x}) $$

---

## Part 3: XGBoost (Extreme Gradient Boosting)

XGBoost is an optimized, highly scalable implementation of gradient boosting that dominates tabular data competitions. It introduces several critical improvements over vanilla gradient boosting:

### 3.1 Taylor Expansion for Loss Optimization
Rather than just using the first derivative (gradients), XGBoost uses a second-order Taylor expansion of the loss function to optimize splits. This allows it to work with any custom, twice-differentiable loss function:

$$ L(y, F_{t-1}(\mathbf{x}) + h_t(\mathbf{x})) \approx L(y, F_{t-1}(\mathbf{x})) + g_i h_t(\mathbf{x}) + \frac{1}{2} h_i h_t(\mathbf{x})^2 $$

where $g_i$ is the first derivative (gradient) and $h_i$ is the second derivative (Hessian) of the loss.

### 3.2 Regularization
XGBoost adds a regularization term to the objective function that penalizes tree complexity (number of leaves $T$ and leaf weights $w$):

$$ \Omega(h) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 $$

This built-in $L_1$ and $L_2$ regularization on the tree weights directly combats overfitting.

---

## Part 4: Practice Exercises

### Exercise 1: Toy Gradient Boosting from Scratch
Let's build a manual boosting regression loop for a 1D dataset:
1. Initialize the prediction $F_0$ as the mean of $y$.
2. Loop 5 times:
   - Compute the residuals: $r_t = y - F_{t-1}$.
   - Fit a shallow decision tree (e.g., `DecisionTreeRegressor(max_depth=1)`) to $r_t$.
   - Update prediction: $F_t = F_{t-1} + 0.1 \times h_t(\mathbf{x})$.
3. **Code it:** Implement this loop using `scikit-learn`'s `DecisionTreeRegressor` as the base learner. Plot the training data and show how the composite model boundary $F_t$ gets progressively closer to fitting the non-linear target at each step.

### Exercise 2: Tuning XGBoost Hyperparameters
1. Load a tabular dataset and split it into training and validation sets.
2. Train an `xgboost.XGBClassifier`.
3. **Code it:** Implement a grid search over:
   - `max_depth` $\in \{3, 5, 7\}$
   - `learning_rate` $\in \{0.01, 0.1, 0.2\}$
   - `n_estimators` $\in \{50, 100, 200\}$
   Plot the validation accuracy curve as a function of `n_estimators` to identify early stopping criteria.

---

## Self-Test Questions

1. **How does Gradient Boosting differ from AdaBoost?** *(AdaBoost identifies errors by increasing the weights of misclassified samples in subsequent datasets. Gradient Boosting identifies errors by calculating the gradients (residuals) of the loss function and training the next model to predict those gradients directly.)*
2. **What is the purpose of the shrinkage/learning rate parameter ($\eta$)?** *(Shrinkage reduces the influence of each individual tree, leaving room for future trees to improve the model. This acts as a regularizer, making the training process slower but preventing the model from quickly overfitting to early training targets.)*
3. **Why does XGBoost use the Hessian (second derivative) of the loss function?** *(Using the Hessian provides curvature information of the loss space, allowing the split-finding algorithm to take larger steps in flat directions and smaller steps in steep directions, leading to faster and more stable optimization.)*
4. **When should you use Random Forests vs. Gradient Boosting?** *(Random Forests are easier to tune because they are robust to hyperparameters and do not overfit as you add more trees. Gradient Boosting models (like XGBoost) usually achieve higher accuracy but are highly sensitive to hyperparameters and can easily overfit if n_estimators is too high.)*
