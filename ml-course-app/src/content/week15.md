# Week 16: Support Vector Machines & Kernels

> **Goal:** Master Support Vector Machines (SVMs). You will understand the mathematical definition of maximum-margin classification, the distinction between hard and soft margins, the mechanism of the "kernel trick," and how to tune critical hyperparameters like $C$ and $\gamma$.

---

## Part 1: Linear SVM and the Maximum Margin

For a binary classification task where data is linearly separable, many decision boundaries can separate the two classes.

```
Class 0  x      / (boundary A)
        x x    /
              /    o o
             /    o     Class 1
            / (boundary B)
```

**Support Vector Machines** choose the boundary that maximizes the **margin**—the distance between the decision boundary (hyperplane) and the closest data points of either class (called **support vectors**).

### 1.1 Hard Margin Formulation
Let the decision boundary be defined by the hyperplane:

$$ \mathbf{w}^T \mathbf{x} + b = 0 $$

For all training points $(\mathbf{x}^{(i)}, y^{(i)})$ where $y^{(i)} \in \{-1, 1\}$, we want:

$$ \begin{cases} 
\mathbf{w}^T \mathbf{x}^{(i)} + b \geq 1 & \text{if } y^{(i)} = 1 \\ 
\mathbf{w}^T \mathbf{x}^{(i)} + b \leq -1 & \text{if } y^{(i)} = -1 
\end{cases} $$

Which can be combined into a single constraint:

$$ y^{(i)} (\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 \quad \forall i $$

The margin width is mathematically shown to be $\frac{2}{\|\mathbf{w}\|_2}$. Maximizing the margin is equivalent to minimizing the inverse:

$$ \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 \quad \text{subject to} \quad y^{(i)} (\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 $$

This is a quadratic programming optimization problem with linear constraints.

---

## Part 2: Soft Margin and Hyperparameter $C$

If data is noisy or not perfectly separable, a "hard margin" is impossible. We introduce **slack variables** $\xi_i \geq 0$ that allow individual points to violate the margin:

$$ y^{(i)} (\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 - \xi_i $$

We then add a penalty for non-zero slack variables to our objective function:

$$ \min_{\mathbf{w}, b, \mathbf{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^m \xi_i $$

### 2.1 The Hyperparameter $C$
The hyperparameter $C$ controls the tradeoff between margin size and classification violations:
- **Large $C$:** High penalty for violations. The model focuses on classifying training points correctly, resulting in a **narrower margin** (higher variance, risk of overfitting).
- **Small $C$:** Low penalty for violations. The model accepts misclassifications to achieve a **wider margin** (higher bias, more robust).

---

## Part 3: The Kernel Trick

Many real-world datasets are not linearly separable in their raw feature space. However, if we project the features into a higher-dimensional space, they often become separable.

### 3.1 Mapping to Higher Dimensions
Imagine 1D data where points near the origin are Class 0 and points far away are Class 1. They are not linearly separable in 1D.
If we map $x \rightarrow (x, x^2)$ (2D), the data forms a parabola, which *is* linearly separable by a 2D line.

### 3.2 The Kernel Function
The dual optimization problem of the SVM depends only on the inner products between feature vectors: $\langle \phi(\mathbf{x}^{(i)}), \phi(\mathbf{x}^{(j)}) \rangle$.

Computing $\phi(\mathbf{x})$ directly in very high dimensions is computationally expensive or impossible. A **Kernel** is a function $K(\mathbf{x}, \mathbf{z})$ that computes the inner product in the high-dimensional space *without* explicitly mapping the vectors:

$$ K(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle $$

### 3.3 Common Kernels
1. **Linear:** $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$
2. **Polynomial:** $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$
3. **Radial Basis Function (RBF / Gaussian):** 
   $$ K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2) $$
   where $\gamma > 0$ defines how far the influence of a single training example reaches (large $\gamma$ means local influence, leading to high variance).

---

## Part 4: Practice Exercises

### Exercise 1: Visualizing SVM Decision Boundaries
1. Create a 2D toy dataset with two concentric circles (using `sklearn.datasets.make_circles`).
2. Train SVMs using:
   - A Linear Kernel.
   - An RBF Kernel.
3. **Code it:** Plot the decision boundaries and highlight the support vectors for both models. Observe how the RBF kernel easily handles the non-linear separation.

### Exercise 2: Grid Sweep for $C$ and $\gamma$
1. Generate synthetic non-linear classification data.
2. Train RBF SVMs across a grid: $C \in \{0.1, 1, 10, 100\}$ and $\gamma \in \{0.01, 0.1, 1, 10\}$.
3. **Code it:** Plot the decision boundaries for all 16 combinations in a grid of subplots. Write a short paragraph explaining which configurations overfit (too complex) vs. underfit (too simple).

---

## Self-Test Questions

1. **Why are support vectors the only points that determine the decision boundary?** *(Because the coefficients $\alpha_i$ in the dual representation are zero for all points that lie strictly outside the margin. Removing non-support vector points from the training set has no effect on the final boundary.)*
2. **Explain the physical meaning of a large vs. small $\gamma$ in the RBF kernel.** *(A large $\gamma$ means the bell-shaped Gaussian curve is very narrow. Only points very close to a support vector are influenced by it, creating complex, tight decision boundaries (overfitting). A small $\gamma$ means a wide Gaussian curve, leading to smooth, general boundaries.)*
3. **If your training set is very large ($m > 100,000$), why is a non-linear SVM slow?** *(Computing the kernel matrix requires calculating pairwise distances between all examples, which scales as $\mathcal{O}(m^2)$ in space and up to $\mathcal{O}(m^3)$ in time during optimization.)*
4. **How does an SVM compare to Logistic Regression on high-dimensional biological embeddings?** *(For high-dimensional inputs with small sample sizes ($d \gg m$), linear SVMs and Logistic Regression perform similarly. However, SVMs are often preferred because maximizing the margin helps prevent overfitting in high dimensions.)*
