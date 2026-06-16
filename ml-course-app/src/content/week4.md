# Week 5: Matrix Calculus & Optimization

> **Goal:** Move from scalar gradients to matrix/vector calculus, then implement the foundational optimizers you will use forever (SGD, Momentum, Adam).

---

## Part 1: Matrix Calculus Essentials

Last week, we looked at gradients of vectors. Now, we expand to matrices.
When you take the derivative of a scalar $y$ with respect to a matrix $X$, the derivative $\frac{\partial y}{\partial X}$ has the *exact same shape* as $X$. Every element $(i, j)$ in the derivative matrix is $\frac{\partial y}{\partial X_{ij}}$.

### 1.1 The Six Key Matrix Calculus Rules
You don't need to derive these from scratch every time. Memorize (or bookmark) these identities for a scalar $y$, vectors $\mathbf{x}, \mathbf{a}$, and symmetric matrix $A$:

1. $\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$
2. $\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T \mathbf{a}) = \mathbf{a}$
3. $\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}$
4. $\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = 2A\mathbf{x}$ *(Only if $A$ is symmetric!)*
5. $\frac{\partial}{\partial X} (\text{Tr}(X A)) = A^T$
6. $\frac{\partial}{\partial X} \log |X| = (X^{-1})^T$

### 1.2 Example: Least Squares Gradient
Let's re-derive the Normal Equations from Week 2 using matrix calculus!
Our loss is $L = \|\mathbf{b} - A\mathbf{x}\|^2 = (\mathbf{b} - A\mathbf{x})^T (\mathbf{b} - A\mathbf{x})$.

Expand it:
$L = \mathbf{b}^T\mathbf{b} - \mathbf{b}^T A\mathbf{x} - (A\mathbf{x})^T \mathbf{b} + (A\mathbf{x})^T(A\mathbf{x})$
$L = \mathbf{b}^T\mathbf{b} - 2\mathbf{b}^T A\mathbf{x} + \mathbf{x}^T A^T A \mathbf{x}$

Now take the derivative with respect to $\mathbf{x}$ using our rules:
$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{0} - 2A^T\mathbf{b} + 2A^T A \mathbf{x}$

To find the minimum, set the gradient to zero:
$2A^T A \mathbf{x} = 2A^T\mathbf{b}$
$A^T A \mathbf{x} = A^T\mathbf{b}$  🎉 *(The Normal Equations!)*

---

## Part 2: Optimization Algorithms

Once we have the gradient $\mathbf{g} = \nabla L(\mathbf{w})$, how do we update our weights $\mathbf{w}$?

### 2.1 Vanilla Gradient Descent (GD)
The simplest approach: step in the opposite direction of the gradient.
$$ \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{g}_t $$
where $\eta$ is the learning rate.
- **Problem:** If the loss landscape is a narrow ravine, GD bounces back and forth across the walls, making very slow progress towards the minimum.

### 2.2 Momentum
Imagine a ball rolling down a hill. It gains momentum.
Instead of using *only* the current gradient, we maintain a running velocity $\mathbf{v}$.

$$ \mathbf{v}_{t+1} = \beta \mathbf{v}_t + (1 - \beta) \mathbf{g}_t $$
$$ \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1} $$
*(Note: standard PyTorch implementation scales the gradient slightly differently, but the concept is identical).*
- $\beta$ is usually 0.9. 
- **Why it works:** Gradients pointing across the ravine cancel out over time, while gradients pointing down the ravine accumulate, speeding up convergence.

### 2.3 RMSProp
What if some weights need a large learning rate and others need a small one? RMSProp divides the learning rate by an exponentially decaying average of squared gradients.

$$ \mathbf{s}_{t+1} = \beta \mathbf{s}_t + (1 - \beta) \mathbf{g}_t^2 $$
$$ \mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_{t+1}} + \epsilon} \mathbf{g}_t $$
- $\mathbf{g}^2$ is element-wise squaring.
- **Why it works:** If a weight has huge gradients, $\mathbf{s}$ gets large, scaling down the learning rate. If a weight has tiny gradients, $\mathbf{s}$ gets small, scaling up the learning rate.

### 2.4 Adam (Adaptive Moment Estimation)
Adam is literally just Momentum + RMSProp. It keeps track of a moving average of the gradient (1st moment) AND a moving average of the squared gradient (2nd moment).

$$ \mathbf{m}_{t+1} = \beta_1 \mathbf{m}_t + (1 - \beta_1) \mathbf{g}_t $$
$$ \mathbf{v}_{t+1} = \beta_2 \mathbf{v}_t + (1 - \beta_2) \mathbf{g}_t^2 $$

Because $\mathbf{m}$ and $\mathbf{v}$ are initialized to 0, they are biased toward 0 in the early steps. Adam includes a **bias correction**:
$$ \hat{\mathbf{m}} = \frac{\mathbf{m}_{t+1}}{1 - \beta_1^t}, \quad \hat{\mathbf{v}} = \frac{\mathbf{v}_{t+1}}{1 - \beta_2^t} $$

$$ \mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}} + \epsilon} \hat{\mathbf{m}} $$

---

## Part 3: Practice Exercises

### Exercise 1: Optimizers from Scratch
You will implement Vanilla GD, Momentum, and Adam to minimize a simple quadratic bowl $f(x, y) = x^2 + 5y^2$. The gradient is $[2x, 10y]$. Notice how the $y$ direction is 5 times steeper!

```python
import numpy as np

# Gradient of f(x, y) = x^2 + 5y^2
def grad(w):
    return np.array([2*w[0], 10*w[1]])

# Initial weights
w_init = np.array([-5.0, -5.0])
epochs = 50
eta = 0.1

# 1. Vanilla GD
w = w_init.copy()
for i in range(epochs):
    w = w - eta * grad(w)
print(f"Vanilla GD final weights: {w}")

# 2. Momentum
w = w_init.copy()
v = np.zeros(2)
beta = 0.9
for i in range(epochs):
    v = beta * v + (1 - beta) * grad(w)
    w = w - eta * v
print(f"Momentum final weights: {w}")

# 3. Code Adam yourself! 
# Initialize m and v to zeros. Use beta1=0.9, beta2=0.999, epsilon=1e-8.
# Remember to apply bias correction!
```

### Exercise 2: Hessian Matrices
The Hessian matrix $H$ is the matrix of second derivatives: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$.
1. Compute the Hessian of $f(x, y) = x^2 + 5y^2$. 
2. What does the fact that the diagonal elements are different tell you about the shape of the loss landscape?

---

## Self-Test Questions
1. **What is the difference between Momentum and RMSProp?** *(Momentum smooths the gradients to build up speed in consistent directions. RMSProp normalizes the learning rate based on the variance of the gradients, slowing down rapidly changing parameters and speeding up sluggish ones.)*
2. **Why does Adam need bias correction?** *(Because the moving averages $m$ and $v$ are initialized to zero, they will artificially remain close to zero for the first few iterations unless corrected.)*
3. **What is the derivative of $\mathbf{x}^T A \mathbf{x}$ with respect to $\mathbf{x}$ if $A$ is symmetric?** *($2A\mathbf{x}$)*
