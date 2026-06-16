# Week 13: Linear Regression, Properly

> **Goal:** Re-derive and re-implement linear regression as a full machine learning workflow. You will understand both the analytical solution (Normal Equation) and the iterative solution (Gradient Descent), see why feature scaling is mathematically necessary, and learn how to diagnose gradient descent convergence.

---

## Part 1: The Analytical Solution — The Normal Equation

Linear regression aims to model the relationship between a vector of inputs $\mathbf{x} \in \mathbb{R}^d$ and a scalar target $y \in \mathbb{R}$ using a linear combination:

$$ \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_d x_d $$

To simplify notation, we append a dummy feature $x_0 = 1$ to our input vector, allowing us to write the model in vectorized form:

$$ \hat{y} = \mathbf{\theta}^T \mathbf{x} $$

Given a dataset of $m$ examples represented as a design matrix $X \in \mathbb{R}^{m \times (d+1)}$ and a target vector $\mathbf{y} \in \mathbb{R}^m$, we want to find the parameter vector $\mathbf{\theta} \in \mathbb{R}^{d+1}$ that minimizes the Mean Squared Error (MSE) cost function:

$$ J(\mathbf{\theta}) = \frac{1}{2m} \sum_{i=1}^m \left( h_\mathbf{\theta}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 = \frac{1}{2m} \|X\mathbf{\theta} - \mathbf{y}\|_2^2 $$

### 1.1 Derivation of the Normal Equation
To find the global minimum of $J(\mathbf{\theta})$, we compute its gradient with respect to $\mathbf{\theta}$ and set it to zero.

Let's expand the matrix expression for the sum of squared errors (omitting the constant $\frac{1}{2m}$ for convenience during derivation):

$$ E(\mathbf{\theta}) = (X\mathbf{\theta} - \mathbf{y})^T (X\mathbf{\theta} - \mathbf{y}) $$
$$ E(\mathbf{\theta}) = (\mathbf{\theta}^T X^T - \mathbf{y}^T) (X\mathbf{\theta} - \mathbf{y}) $$
$$ E(\mathbf{\theta}) = \mathbf{\theta}^T X^T X \mathbf{\theta} - \mathbf{\theta}^T X^T \mathbf{y} - \mathbf{y}^T X \mathbf{\theta} + \mathbf{y}^T \mathbf{y} $$

Since $\mathbf{\theta}^T X^T \mathbf{y}$ is a scalar, it equals its transpose $\mathbf{y}^T X \mathbf{\theta}$. We can combine them:

$$ E(\mathbf{\theta}) = \mathbf{\theta}^T X^T X \mathbf{\theta} - 2\mathbf{\theta}^T X^T \mathbf{y} + \mathbf{y}^T \mathbf{y} $$

Now, we compute the gradient with respect to $\mathbf{\theta}$ using matrix calculus rules:
1. $\nabla_\mathbf{\theta} (\mathbf{u}^T \mathbf{\theta}) = \mathbf{u}$
2. $\nabla_\mathbf{\theta} (\mathbf{\theta}^T A \mathbf{\theta}) = 2 A \mathbf{\theta}$ (for symmetric $A$)

Applying these rules:

$$ \nabla_\mathbf{\theta} E(\mathbf{\theta}) = 2 X^T X \mathbf{\theta} - 2 X^T \mathbf{y} $$

Setting the gradient to the zero vector:

$$ 2 X^T X \mathbf{\theta} - 2 X^T \mathbf{y} = \mathbf{0} $$
$$ X^T X \mathbf{\theta} = X^T \mathbf{y} $$

Assuming $X^T X$ is invertible (non-singular), we multiply both sides by $(X^T X)^{-1}$:

$$ \mathbf{\theta} = (X^T X)^{-1} X^T \mathbf{y} $$

This is the **Normal Equation**. It provides the exact analytical solution to the ordinary least squares problem in one step.

```python
import numpy as np

# Generate random toy data
np.random.seed(42)
X_raw = 2 * np.random.rand(100, 1)
y = 4 + 3 * X_raw + np.random.randn(100, 1)

# Add bias column (x0 = 1)
X_b = np.c_[np.ones((100, 1)), X_raw]

# Compute normal equation: theta = inv(X^T * X) * X^T * y
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("Normal Equation theta:", theta_best.ravel())
# Expected: close to [4, 3]
```

---

## Part 2: The Iterative Solution — Gradient Descent

While the Normal Equation is convenient, calculating the inverse of $X^T X$ has a computational complexity of $\mathcal{O}(d^3)$, where $d$ is the number of features. If your dataset has $100,000$ features, inverting the matrix is extremely slow.

**Gradient Descent** is an optimization algorithm that iteratively minimizes a cost function by moving in the direction of steepest descent (opposite to the gradient).

### 2.1 The Update Rule
The cost function is:
$$ J(\mathbf{\theta}) = \frac{1}{2m} (X\mathbf{\theta} - \mathbf{y})^T (X\mathbf{\theta} - \mathbf{y}) $$

The gradient is:
$$ \nabla_\mathbf{\theta} J(\mathbf{\theta}) = \frac{1}{m} X^T (X\mathbf{\theta} - \mathbf{y}) $$

At each step, we update the parameters:

$$ \mathbf{\theta} \leftarrow \mathbf{\theta} - \alpha \nabla_\mathbf{\theta} J(\mathbf{\theta}) $$

where $\alpha > 0$ is the **learning rate**.

```python
# Batch Gradient Descent implementation
learning_rate = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T @ (X_b @ theta - y)  # Note: using 2/m to match common convention
    theta = theta - learning_rate * gradients

print("Gradient Descent theta:", theta.ravel())
```

---

## Part 3: The Importance of Feature Scaling

When features have wildly different ranges (e.g., house size in sq ft $[500, 5000]$ vs. number of bedrooms $[1, 5]$), the cost function contours will look like highly elongated ellipses.

### 3.1 Why Elongated Contours Harm Gradient Descent
- If contours are elongated, gradient descent will oscillate back and forth in the direction of the steep slope (the small-scale feature) while making very slow progress along the gentle slope (the large-scale feature).
- To prevent divergence, you have to use a very small learning rate, which makes training extremely slow.
- Scaling features makes the cost contours circular, allowing gradient descent to path directly towards the minimum using a much larger learning rate.

### 3.2 Standardization (Z-score normalization)
Subtract the mean $\mu$ and divide by the standard deviation $\sigma$:

$$ x' = \frac{x - \mu}{\sigma} $$

```python
# Feature scaling example
X_scaled = (X_raw - np.mean(X_raw)) / np.std(X_raw)
```

---

## Part 4: Practice Exercises

### Exercise 1: Normal Equation vs Gradient Descent from Scratch
1. Generate synthetic data with 3 features and 500 samples.
2. Implement linear regression using the Normal Equation.
3. Implement Batch Gradient Descent from scratch.
4. **Code it:** Measure the accuracy of both models. Do they converge to the same parameters? Experiment with setting the learning rate too high (e.g., $\alpha = 2.0$) or too low (e.g., $\alpha = 0.0001$).

### Exercise 2: Gradient Descent Path Visualization
1. Write a function that records the path of $\mathbf{\theta}$ during training.
2. Create a grid of $\theta_0$ and $\theta_1$ values and compute the cost $J(\theta)$ for each point on the grid.
3. Plot the contours of the cost function.
4. Superimpose the parameter trajectories for three learning rates: $\alpha = 0.01$, $\alpha = 0.1$, and $\alpha = 0.5$.

---

## Self-Test Questions

1. **What is the computational complexity of the Normal Equation vs. Gradient Descent?** *(The Normal Equation requires computing $(X^T X)^{-1}$, which is $\mathcal{O}(d^3)$. Batch Gradient Descent is $\mathcal{O}(k \cdot m \cdot d)$ where $k$ is the number of iterations, making it much more scalable for large feature dimensions.)*
2. **What happens if $X^T X$ is not invertible?** *(If $X^T X$ is singular (not invertible), it is usually because features are linearly dependent (e.g., redundant features like Celsius and Fahrenheit) or $m < d$. We can resolve this using pseudoinverse `np.linalg.pinv` or applying regularization.)*
3. **Why do we need feature scaling for Gradient Descent but not for the Normal Equation?** *(Normal Equation is an analytical solution that finds the exact minimum in a single algebraic projection, which is scale-invariant. Gradient Descent steps are perpendicular to the cost contours; asymmetric scaling distorts gradients and forces inefficient path steps.)*
4. **How do learning curves help diagnose overshooting?** *(If the learning rate is too high, the loss $J(\mathbf{\theta})$ will increase at each iteration, or fluctuate wildly and diverge, rather than decreasing smoothly over time.)*
