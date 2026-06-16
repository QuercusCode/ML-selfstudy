# Week 2: Linear Systems, Rank, and The Four Subspaces

> **Goal:** Understand solvability (when does `Ax = b` have 0/1/∞ solutions?) and projection — the geometric heart of least squares regression.

---

## Part 1: Solving $A\mathbf{x} = \mathbf{b}$

Last week, we saw matrices as transformations. This week, we use them to solve equations.

A system of linear equations is written as $A\mathbf{x} = \mathbf{b}$:

$$ \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix} $$

### 1.1 The Column Space Perspective
Instead of looking at rows (equations of planes), look at the columns. 

$$ x_1 \begin{bmatrix} a_{11} \\ a_{21} \\ a_{31} \end{bmatrix} + x_2 \begin{bmatrix} a_{12} \\ a_{22} \\ a_{32} \end{bmatrix} + x_3 \begin{bmatrix} a_{13} \\ a_{23} \\ a_{33} \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix} $$

The question "$A\mathbf{x} = \mathbf{b}$" is exactly asking: **"Can I find a linear combination of the columns of $A$ that produces the vector $\mathbf{b}$?"**

The set of *all possible linear combinations* of the columns of $A$ is called the **Column Space**, denoted $C(A)$. 
Therefore, $A\mathbf{x} = \mathbf{b}$ has a solution **if and only if** $\mathbf{b}$ is in the column space of $A$.

### 1.2 The Null Space Perspective
What if $\mathbf{b} = \mathbf{0}$? 
The equation $A\mathbf{x} = \mathbf{0}$ always has the trivial solution $\mathbf{x} = \mathbf{0}$. But are there others?

The set of all vectors $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$ is called the **Null Space** of $A$, denoted $N(A)$.
Geometrically, the null space consists of all vectors that are "squashed" to the origin by the transformation $A$.

If $N(A)$ contains only the zero vector, then the columns of $A$ are linearly independent. If $N(A)$ contains non-zero vectors, the columns are linearly dependent, and you have redundant information.

---

## Part 2: Rank and The Fundamental Theorem of Linear Algebra

### 2.1 Matrix Rank
The **rank** of a matrix $A$ is the dimension of its column space. It is the number of linearly independent columns.
- If $A$ is $m \times n$, then $\text{rank}(A) \leq \min(m, n)$.
- If $\text{rank}(A) = n$, the matrix has **full column rank**.
- If $\text{rank}(A) = m$, the matrix has **full row rank**.

### 2.2 The Rank-Nullity Theorem
For any $m \times n$ matrix $A$:
$$ \text{Rank}(A) + \text{Dimension}(N(A)) = n $$

This means every column of a matrix must contribute either to the dimensions of the output space (Rank) or it must get collapsed to zero (Nullity). Energy is conserved.

### 2.3 The Four Fundamental Subspaces
Every $m \times n$ matrix $A$ has four fundamental subspaces. Understanding how they interact is the crown jewel of linear algebra.

| Subspace | Symbol | Lives in | Dimension |
| :--- | :---: | :---: | :---: |
| **Column Space** | $C(A)$ | $\mathbb{R}^m$ | $r$ |
| **Null Space** | $N(A)$ | $\mathbb{R}^n$ | $n - r$ |
| **Row Space** | $C(A^T)$ | $\mathbb{R}^n$ | $r$ |
| **Left Null Space** | $N(A^T)$ | $\mathbb{R}^m$ | $m - r$ |

**Crucial Geometric Fact:** 
- The Null Space is **orthogonal** to the Row Space. Every vector in $N(A)$ is perpendicular to every row of $A$.
- The Left Null Space is **orthogonal** to the Column Space.

```python
import numpy as np

# A 3x4 matrix
A = np.array([
    [1, 2, 3, 1],
    [1, 1, 2, 1],
    [1, 2, 3, 1]  # Row 3 is identical to Row 1
])

# Rank computation
rank = np.linalg.matrix_rank(A)
n_cols = A.shape[1]
nullity = n_cols - rank

print(f"Matrix shape: {A.shape}")
print(f"Rank r = {rank}")
print(f"Nullity (n - r) = {nullity}")
# Since n=4 and rank=2, the null space is 2-dimensional.
```

---

## Part 3: Projections and Least Squares

In machine learning, we rarely have $A\mathbf{x} = \mathbf{b}$ perfectly solvable. Usually, we have more equations (data points) than variables (features). This means $A$ is tall and skinny, and $\mathbf{b}$ is almost certainly *not* in the column space $C(A)$.

So, no exact solution $\mathbf{x}$ exists. What is the next best thing?
We want to find an $\mathbf{\hat{x}}$ such that $A\mathbf{\hat{x}}$ is as close as possible to $\mathbf{b}$.

### 3.1 Geometric Projection
The closest point to $\mathbf{b}$ in the column space $C(A)$ is its **orthogonal projection**, denoted $\mathbf{p}$.
Since $\mathbf{p}$ is in the column space, we know there must be some vector $\mathbf{\hat{x}}$ such that $A\mathbf{\hat{x}} = \mathbf{p}$.

Because the projection is orthogonal, the "error" vector $\mathbf{e} = \mathbf{b} - A\mathbf{\hat{x}}$ must be perpendicular to the entire column space of $A$.
This means $\mathbf{e}$ is in the Left Null Space $N(A^T)$, so:

$$ A^T (\mathbf{b} - A\mathbf{\hat{x}}) = \mathbf{0} $$

Expand this equation to get the **Normal Equations**:

$$ A^T A \mathbf{\hat{x}} = A^T \mathbf{b} $$

Solving for $\mathbf{\hat{x}}$ gives the absolute foundation of Linear Regression:

$$ \mathbf{\hat{x}} = (A^T A)^{-1} A^T \mathbf{b} $$

The matrix $(A^T A)^{-1} A^T$ is called the **pseudo-inverse** of $A$.
The projection matrix that projects any vector onto $C(A)$ is:

$$ P = A (A^T A)^{-1} A^T $$

```python
import numpy as np
import matplotlib.pyplot as plt

# A tall, skinny matrix (3 equations, 2 unknowns)
A = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])
# A vector b NOT in the column space
b = np.array([1, 2, 2])

# Calculate least squares solution: x_hat = (A^T A)^-1 A^T b
AtA = A.T @ A
Atb = A.T @ b
x_hat = np.linalg.solve(AtA, Atb)

print(f"Best fit weights (x_hat): {x_hat}")

# Calculate the projection p
p = A @ x_hat
print(f"Projection (p): {p}")

# The error e must be orthogonal to A's columns
e = b - p
print(f"Error vector (e): {e}")
print(f"A^T @ e (should be ~0): {A.T @ e}")
```

---

## Part 4: Practice Exercises

Complete these exercises directly using the Python blocks.

### Exercise 1: Finding the Null Space
Let $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}$.
1. What is the rank of $A$? 
2. Without coding, what is the dimension of the null space?
3. Find a non-zero vector $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$.
4. **Code it:** Create $A$ in NumPy, compute its rank using `np.linalg.matrix_rank`, and verify your vector $\mathbf{x}$.

### Exercise 2: Orthogonal Projections
You want to project the vector $\mathbf{b} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$ onto the line spanned by $\mathbf{a} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$.
1. In this case, $A$ is just the column vector $\mathbf{a}$. Write out $A^T A$. What is it?
2. Use the projection formula $P = A(A^T A)^{-1} A^T$ to find the projection matrix $P$.
3. Compute the projected vector $\mathbf{p} = P\mathbf{b}$.
4. **Code it:** Write a short script to compute this automatically.

### Exercise 3: Linear Regression from Scratch
You have 4 data points $(x, y)$: `(1, 2), (2, 3), (3, 5), (4, 4)`. You want to fit a line $y = mx + c$.
1. Set up the matrix $A$ (hint: one column is $x$ values, the other is all 1s for the intercept $c$).
2. Set up the vector $\mathbf{b}$ (the $y$ values).
3. **Code it:** Use the Normal Equations $(A^T A)^{-1} A^T \mathbf{b}$ to find $m$ and $c$, then plot the points and the best-fit line.

---

## Self-Test Questions

1. **If a $5 \times 3$ matrix has rank 3, does $A\mathbf{x} = \mathbf{b}$ always have a solution?** *(No. The column space is 3-dimensional, but it lives in $\mathbb{R}^5$. Most vectors $\mathbf{b}$ will not be in the column space.)*
2. **For that same matrix, if $A\mathbf{x} = \mathbf{b}$ DOES have a solution, is it unique?** *(Yes. Since $n=3$ and rank=3, the nullity is 0. There are no free variables to create infinitely many solutions.)*
3. **Why do we use $A^T A$ in least squares regression?** *(Because the error vector $\mathbf{e}$ is orthogonal to the column space of $A$. Orthogonality means $A^T \mathbf{e} = \mathbf{0}$. Expanding $\mathbf{e} = \mathbf{b} - A\mathbf{\hat{x}}$ leads directly to $A^T A \mathbf{\hat{x}} = A^T \mathbf{b}$.)*
4. **What is the dimension of the row space of a matrix with rank $r$?** *(Always exactly $r$, same as the column space!)*
