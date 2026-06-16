# Week 4: Calculus, Gradients, and the Chain Rule

> **Goal:** Backpropagation is just the multivariate chain rule applied systematically. If you own the chain rule and the Jacobian, backprop stops being magic and becomes pure mechanics.

---

## Part 1: Derivatives and Gradients

In 1D calculus, the derivative $f'(x)$ tells you the rate of change of $f$ with respect to $x$. 
In Machine Learning, we have loss functions $L(\mathbf{w})$ that depend on thousands or millions of parameters $\mathbf{w}$. We need to know how the loss changes when we tweak *any* of those parameters.

### 1.1 Partial Derivatives
A partial derivative $\frac{\partial f}{\partial x_1}$ is just the derivative of $f$ with respect to $x_1$, treating all other variables as constants.

Example: $f(x, y) = x^2 y + \sin(x)$
- $\frac{\partial f}{\partial x} = 2xy + \cos(x)$
- $\frac{\partial f}{\partial y} = x^2$

### 1.2 The Gradient
The **gradient** of a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of all its partial derivatives:

$$ \nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} $$

**Geometric meaning:** The gradient vector points in the direction of **steepest ascent**. Its magnitude tells you how steep that slope is.
To minimize a loss function, we want to go down, so we move in the direction of the **negative gradient**: $-\nabla f(\mathbf{x})$.

---

## Part 2: The Jacobian Matrix

What if our function outputs a vector instead of a scalar?
Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$. The gradient is no longer a single vector; it's a matrix of all possible partial derivatives. This is the **Jacobian matrix** $J$:

$$ J_{ij} = \frac{\partial f_i}{\partial x_j} $$

$$ J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} $$

The Jacobian is the ultimate generalization of the derivative. If $f$ is a linear transformation $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$, then its Jacobian is simply $A$. For non-linear functions, the Jacobian is the *best linear approximation* of the function at a specific point.

---

## Part 3: The Multivariate Chain Rule

In 1D, the chain rule is $\frac{d}{dx} f(g(x)) = f'(g(x)) g'(x)$.
In ML, we compose functions: $\text{Loss} = L(\mathbf{z})$ where $\mathbf{z} = \mathbf{f}(\mathbf{w})$.

To find how the loss changes with respect to the weights $\mathbf{w}$, we multiply Jacobians:

$$ \frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial \mathbf{z}} \frac{\partial \mathbf{z}}{\partial \mathbf{w}} $$

Notice the shapes! 
- $L$ is a scalar (shape $1 \times 1$).
- $\mathbf{z}$ is a vector of size $m$.
- $\mathbf{w}$ is a vector of size $n$.
- $\frac{\partial L}{\partial \mathbf{z}}$ is a $1 \times m$ row vector (the gradient).
- $\frac{\partial \mathbf{z}}{\partial \mathbf{w}}$ is an $m \times n$ Jacobian matrix.
- Their product is a $1 \times n$ row vector, exactly the gradient we want!

### 3.1 Backpropagation is Just Left-to-Right Matrix Multiplication
If we have a deep network: $L(f_3(f_2(f_1(\mathbf{x}))))$, the chain rule gives:

$$ \frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial f_3} \frac{\partial f_3}{\partial f_2} \frac{\partial f_2}{\partial f_1} \frac{\partial f_1}{\partial \mathbf{x}} $$

**Backpropagation** evaluates this product from left to right (starting at the loss and moving backwards). Why? Because $\frac{\partial L}{\partial f_3}$ is a $1 \times m$ vector. Multiplying a vector by a matrix is an $O(n^2)$ operation. If we evaluated right-to-left, we'd be multiplying massive Jacobian matrices together, which is $O(n^3)$! Backprop is just computationally efficient chain rule.

---

## Part 4: Practice Exercises

### Exercise 1: Gradient of MSE
Let $\mathbf{y}$ be the true labels and $\mathbf{\hat{y}} = \mathbf{w}^T \mathbf{x}$ be the predictions.
The Mean Squared Error is $L = \frac{1}{2} \|\mathbf{y} - \mathbf{w}^T \mathbf{x}\|^2$.
1. Derive $\frac{\partial L}{\partial \mathbf{w}}$ by hand using the chain rule.
2. **Code it:** Verify your analytic gradient using finite differences.

```python
import numpy as np

np.random.seed(42)
x = np.random.randn(3)
y = 2.5
w = np.random.randn(3)

def loss(w):
    return 0.5 * (y - np.dot(w, x))**2

# 1. Analytic gradient (from your hand derivation)
grad_analytic = -(y - np.dot(w, x)) * x

# 2. Numerical gradient (finite differences)
epsilon = 1e-5
grad_numerical = np.zeros(3)
for i in range(3):
    w_plus = w.copy()
    w_plus[i] += epsilon
    grad_numerical[i] = (loss(w_plus) - loss(w)) / epsilon

print("Analytic:", grad_analytic)
print("Numerical:", grad_numerical)
print("Difference:", np.abs(grad_analytic - grad_numerical))
```

### Exercise 2: The Softmax Jacobian
Let $\mathbf{z} \in \mathbb{R}^k$ be a vector of logits. The softmax function is $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}$.
1. Derive the Jacobian matrix $\frac{\partial \sigma}{\partial \mathbf{z}}$. Note that the derivative $\frac{\partial \sigma_i}{\partial z_j}$ depends on whether $i=j$ or $i \neq j$.
2. Write down the two cases ($i=j$ and $i \neq j$).

---

## Self-Test Questions
1. **What is the shape of the Jacobian of a function that takes a vector of size 10 and outputs a vector of size 5?** *(A $5 \times 10$ matrix.)*
2. **Why does gradient descent step in the direction of the NEGATIVE gradient?** *(Because the gradient vector points in the direction of steepest ascent. We want to minimize the loss, so we go the opposite way.)*
3. **Why do we evaluate the chain rule from left to right (from loss to input) during backpropagation?** *(To ensure we are always doing vector-matrix multiplications rather than matrix-matrix multiplications, saving massive amounts of computation.)*
