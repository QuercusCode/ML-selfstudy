# Week 14: Logistic Regression & Classification Metrics

> **Goal:** Transition from regression to classification. You will learn the mechanics of Logistic Regression, derive the Binary Cross-Entropy loss function, understand decision boundaries, and master evaluation metrics (Precision, Recall, F1-Score, ROC-AUC) to diagnose classifier performance, especially on imbalanced datasets.

---

## Part 1: The Logistic Regression Model

In classification tasks, we want to predict a discrete class label. In the binary case, $y \in \{0, 1\}$. 

A linear regression model $\mathbf{\theta}^T \mathbf{x}$ output values in $(-\infty, \infty)$, which is not suitable for predicting probabilities. To map our linear output to a probability $p \in [0, 1]$, we pass it through the **logistic function** (also called the **sigmoid function**):

$$ g(z) = \frac{1}{1 + e^{-z}} $$

### 1.1 The Sigmoid Function
The sigmoid function maps any real-valued number $z$ to a range between 0 and 1.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Our hypothesis function for logistic regression is:

$$ h_\mathbf{\theta}(\mathbf{x}) = g(\mathbf{\theta}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{\theta}^T \mathbf{x}}} $$

We interpret $h_\mathbf{\theta}(\mathbf{x})$ as the probability that $y = 1$ given $\mathbf{x}$, parameterized by $\mathbf{\theta}$:

$$ P(y = 1 \mid \mathbf{x}; \mathbf{\theta}) = h_\mathbf{\theta}(\mathbf{x}) $$
$$ P(y = 0 \mid \mathbf{x}; \mathbf{\theta}) = 1 - h_\mathbf{\theta}(\mathbf{x}) $$

---

## Part 2: Loss Function — Binary Cross-Entropy

We cannot use Mean Squared Error (MSE) for Logistic Regression because the resulting cost function $J(\mathbf{\theta})$ would be non-convex (wavy), meaning gradient descent could get stuck in local minima. Instead, we use **Binary Cross-Entropy (BCE)** loss, derived from Maximum Likelihood Estimation (MLE).

### 2.1 Derivation of the Loss
For a single training example $(\mathbf{x}, y)$, we can write the probability distribution of the target variable in a single equation:

$$ P(y \mid \mathbf{x}; \mathbf{\theta}) = \left( h_\mathbf{\theta}(\mathbf{x}) \right)^y \left( 1 - h_\mathbf{\theta}(\mathbf{x}) \right)^{1-y} $$

If we assume our $m$ training examples are independent, the likelihood of parameters $\mathbf{\theta}$ is:

$$ L(\mathbf{\theta}) = \prod_{i=1}^m \left( h_\mathbf{\theta}(\mathbf{x}^{(i)}) \right)^{y^{(i)}} \left( 1 - h_\mathbf{\theta}(\mathbf{x}^{(i)}) \right)^{1-y^{(i)}} $$

To make optimization easier, we take the natural logarithm to get the log-likelihood:

$$ \log L(\mathbf{\theta}) = \sum_{i=1}^m \left[ y^{(i)} \log h_\mathbf{\theta}(\mathbf{x}^{(i)}) + (1 - y^{(i)}) \log(1 - h_\mathbf{\theta}(\mathbf{x}^{(i)})) \right] $$

In machine learning, we prefer to *minimize* cost, so we define our cost function $J(\mathbf{\theta})$ as the negative log-likelihood normalized by the dataset size $m$:

$$ J(\mathbf{\theta}) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log h_\mathbf{\theta}(\mathbf{x}^{(i)}) + (1 - y^{(i)}) \log(1 - h_\mathbf{\theta}(\mathbf{x}^{(i)})) \right] $$

### 2.2 Gradient of BCE Loss
Using the chain rule, we can show that the gradient of $J(\mathbf{\theta})$ with respect to the weights is:

$$ \nabla_\mathbf{\theta} J(\mathbf{\theta}) = \frac{1}{m} X^T \left( h_\mathbf{\theta}(X) - \mathbf{y} \right) $$

Notice that this gradient is mathematically identical in form to the gradient for linear regression! The only difference is that $h_\mathbf{\theta}(\mathbf{x})$ now contains the sigmoid function.

---

## Part 3: Classification Metrics for Imbalanced Data

If you have a dataset where 99% of the samples belong to Class 0 (e.g., rare disease diagnostics), a dummy classifier that always predicts Class 0 will achieve **99% accuracy**. Accuracy is a misleading metric for imbalanced data.

### 3.1 The Confusion Matrix
A confusion matrix categorizes predictions into four quadrants:

| | Predicted Positive ($1$) | Predicted Negative ($0$) |
|---|---|---|
| **Actual Positive ($1$)** | True Positive (TP) | False Negative (FN) |
| **Actual Negative ($0$)** | False Positive (FP) | True Negative (TN) |

### 3.2 Precision, Recall, and F1-Score
- **Precision:** Of all predicted positives, what fraction was actually positive?
  $$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$
- **Recall (Sensitivity):** Of all actual positives, what fraction did we capture?
  $$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$
- **F1-Score:** The harmonic mean of precision and recall. It balances both metrics:
  $$ \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

### 3.3 ROC and AUC
- **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate ($\text{TPR} = \text{Recall}$) against the False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$) at various threshold settings.
- **Area Under Curve (AUC)** measures the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one. Perfect classifiers have $\text{AUC} = 1.0$.

---

## Part 4: Practice Exercises

### Exercise 1: Logistic Regression from Scratch
1. Generate a 2D toy dataset with two classes that are linearly separable.
2. Implement the sigmoid function, BCE loss, and Batch Gradient Descent from scratch.
3. **Code it:** Train the model and plot the decision boundary line ($h_\mathbf{\theta}(\mathbf{x}) = 0.5$, which simplifies to $\mathbf{\theta}^T \mathbf{x} = 0$).

### Exercise 2: Metric Calculations by Hand
Given the following outcomes from a diagnostic test on 100 patient samples:
- Actual Sick ($1$): 10 patients
- Actual Healthy ($0$): 90 patients
- Your model predicts Sick ($1$) for 8 actual sick patients and 12 actual healthy patients.
1. Construct the confusion matrix.
2. Calculate Accuracy, Precision, Recall, and F1-score by hand.
3. **Code it:** Write a small script using `sklearn.metrics` to verify your manual calculations.

---

## Self-Test Questions

1. **Why is the decision boundary of logistic regression linear?** *(Because we classify a point as $1$ if $h_\mathbf{\theta}(\mathbf{x}) \geq 0.5$. Since $g(z) \geq 0.5$ when $z \geq 0$, our boundary is $\mathbf{\theta}^T \mathbf{x} = 0$. Since this is a linear equation in the feature space, the boundary is a line/hyperplane.)*
2. **If a cancer detector requires extremely high recall, what does that mean for precision?** *(There is a tradeoff. To avoid missing any actual cancer (high recall / low FN), we must lower our threshold to flag potential cases, which increases false positives (FP), thereby decreasing precision.)*
3. **Why do we use the harmonic mean instead of the arithmetic mean for F1-Score?** *(Arithmetic mean is susceptible to extreme values. A model with 100% precision and 0% recall would get an arithmetic mean of 50%, but its F1-Score is 0%, reflecting its uselessness.)*
4. **What does an AUC of 0.5 represent?** *(An AUC of 0.5 means the model performs no better than random guessing.)*
