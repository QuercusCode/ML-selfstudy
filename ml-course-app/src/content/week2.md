# Week 3: Eigenvalues, Eigenvectors, SVD, and PCA

> **Goal:** This is arguably the most important week of linear algebra for Machine Learning. Singular Value Decomposition (SVD) and Principal Component Analysis (PCA) underlie dimensionality reduction, embeddings, recommendation systems, and extensive biological data analysis.

---

## Part 1: Eigenvectors and Eigenvalues

When a matrix $A$ transforms space, most vectors get knocked off their original span. They change direction.
However, for a square matrix, there are special vectors that **stay on their original line**. They are only stretched or squished (scaled) by the transformation. 

These special vectors are **eigenvectors**, and the factor by which they are scaled is their **eigenvalue**.

Mathematically, a non-zero vector $\mathbf{v}$ is an eigenvector of a square matrix $A$ with eigenvalue $\lambda$ if:

$$ A\mathbf{v} = \lambda\mathbf{v} $$

### 1.1 Finding Eigenvalues Geometrically
Think about these transformations:
- **Scaling by 2:** *Every* vector is an eigenvector with eigenvalue $\lambda = 2$.
- **180° Rotation:** *Every* vector is an eigenvector with eigenvalue $\lambda = -1$.
- **Projection onto the x-axis:** 
  - Vectors on the x-axis stay there and don't change size ($\lambda = 1$).
  - Vectors on the y-axis get collapsed to zero ($\lambda = 0$).
- **90° Rotation:** *No* real vector stays on its span. This matrix has no *real* eigenvalues (they are complex numbers).

### 1.2 Finding Eigenvalues Algebraically
To solve $A\mathbf{v} = \lambda\mathbf{v}$, we can rewrite it:

$$ A\mathbf{v} - \lambda I \mathbf{v} = \mathbf{0} $$
$$ (A - \lambda I)\mathbf{v} = \mathbf{0} $$

For this equation to have a non-zero solution $\mathbf{v}$, the matrix $(A - \lambda I)$ must squish space into a lower dimension. This means its determinant must be zero:

$$ \det(A - \lambda I) = 0 $$

This is the **characteristic equation**. Solving it gives you the eigenvalues $\lambda$. Once you have $\lambda$, plug it back into $(A - \lambda I)\mathbf{v} = \mathbf{0}$ to find the corresponding eigenvector $\mathbf{v}$.

```python
import numpy as np

# A symmetric 2x2 matrix
A = np.array([[3, 1],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# Verify A * v = lambda * v for the first eigenvector
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]

print("\nA * v1:", A @ v1)
print("lambda1 * v1:", lambda1 * v1)
# They match!
```

---

## Part 2: Eigendecomposition (Diagonalization)

If a $n \times n$ matrix $A$ has $n$ linearly independent eigenvectors, we can put them into the columns of a matrix $V$. We put the corresponding eigenvalues into a diagonal matrix $\Lambda$.

This allows us to write:
$$ AV = V\Lambda $$

By multiplying both sides by $V^{-1}$ on the right, we get the **Eigendecomposition** of $A$:

$$ A = V \Lambda V^{-1} $$

This is a beautiful factorization. It says that the transformation $A$ can be broken down into three steps:
1. $V^{-1}$: Change the coordinate system to align with the eigenvectors.
2. $\Lambda$: Stretch/squish along these new axes (a simple diagonal scaling).
3. $V$: Rotate back to the original coordinate system.

### 2.1 The Magic of Symmetric Matrices
In Machine Learning (specifically covariance matrices), we deal with **symmetric matrices** ($A = A^T$).
Symmetric matrices have incredible properties:
1. They always have real eigenvalues.
2. Their eigenvectors are always **orthogonal** to each other.

Because the eigenvectors are orthogonal, the matrix $V$ is an orthogonal matrix (we usually call it $Q$). The inverse of an orthogonal matrix is simply its transpose ($Q^{-1} = Q^T$). So for symmetric matrices:

$$ A = Q \Lambda Q^T $$

---

## Part 3: Singular Value Decomposition (SVD)

Eigendecomposition is great, but it only works for square, diagonalizable matrices. What if you have a rectangular $m \times n$ dataset (e.g. $m$ users, $n$ movies)? 
**Enter the SVD.**

The SVD says that **any** matrix $A$, square or not, can be factored into three matrices:

$$ A = U \Sigma V^T $$

1. **$V^T$**: An $n \times n$ orthogonal matrix. Its rows are the **right singular vectors**. It performs a rotation in the input space.
2. **$\Sigma$**: An $m \times n$ diagonal matrix containing the **singular values** ($\sigma_i \geq 0$). It performs a scaling.
3. **$U$**: An $m \times m$ orthogonal matrix. Its columns are the **left singular vectors**. It performs a rotation in the output space.

### 3.1 The Geometry of SVD
SVD tells us a profound geometric truth: **Any linear transformation, no matter how complex, is just a rotation, followed by a scaling, followed by another rotation.**

### 3.2 Matrix Approximation (Truncated SVD)
The singular values in $\Sigma$ are sorted in descending order: $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.
The larger the singular value, the more "important" that geometric direction is to the matrix.

If we keep only the top $k$ singular values (and set the rest to zero), we get the **best rank-$k$ approximation** of $A$. This is the basis of image compression, noise reduction, and recommendation systems.

```python
import numpy as np

# A random 4x3 matrix
A = np.random.randn(4, 3)

# Compute SVD
U, S, Vt = np.linalg.svd(A)

print("U shape:", U.shape)
print("Singular Values (S):", S)  # 3 values, already sorted!
print("V^T shape:", Vt.shape)

# Reconstruct A
# Since S is a 1D array, we need to create the diagonal matrix Sigma
Sigma = np.zeros((4, 3))
Sigma[:3, :3] = np.diag(S)

A_reconstructed = U @ Sigma @ Vt
print("\nReconstruction error:", np.max(np.abs(A - A_reconstructed)))
# Should be ~1e-15
```

---

## Part 4: Principal Component Analysis (PCA)

PCA is the single most widely used algorithm for dimensionality reduction. It takes a high-dimensional dataset and projects it down to a lower dimension while preserving as much variance (information) as possible.

### 4.1 The Setup
Imagine a dataset $X$ as an $n \times d$ matrix ($n$ samples, $d$ features).
1. **Center the data**: Subtract the mean of each column from $X$.
2. Compute the **Covariance Matrix**: $C = \frac{1}{n-1} X^T X$.
   - $C$ is a $d \times d$ symmetric matrix. It measures how features vary with themselves (variance) and with each other (covariance).

### 4.2 The Eigendecomposition Perspective
Since $C$ is symmetric, we can eigendecompose it: $C = V \Lambda V^T$.
- The eigenvectors (columns of $V$) are the **Principal Components**. They are the new orthogonal axes of our data.
- The eigenvalues (diagonal of $\Lambda$) tell us the **variance** explained by each principal component.

To project our data $X$ down to $k$ dimensions, we multiply $X$ by the first $k$ eigenvectors:
$$ X_{\text{reduced}} = X V_k $$

### 4.3 The SVD Perspective
Computing the covariance matrix $X^T X$ can cause massive numerical instability if $X$ is large. We can skip it entirely using SVD!

Let $X = U \Sigma V^T$. Then:
$$ X^T X = (V \Sigma U^T)(U \Sigma V^T) = V \Sigma^2 V^T $$

Comparing this to $C = V \Lambda V^T$, we see:
- The right singular vectors $V$ of $X$ are exactly the principal components!
- The singular values $\sigma_i$ are the square roots of the eigenvalues: $\lambda_i = \sigma_i^2 / (n-1)$.

**Conclusion:** To do PCA, you don't compute the covariance matrix. You just mean-center $X$ and take its SVD.

```python
import numpy as np

# Create some correlated 2D data (100 points)
np.random.seed(42)
x1 = np.random.randn(100)
x2 = 2 * x1 + np.random.randn(100) * 0.5
X = np.column_stack((x1, x2))

# 1. Center the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 2. SVD
U, S, Vt = np.linalg.svd(X_centered)

# The principal components are the rows of Vt (or columns of V)
PC1 = Vt[0, :]
PC2 = Vt[1, :]

print(f"First Principal Component (Direction of max variance): {PC1}")
print(f"Second Principal Component: {PC2}")

# 3. Project data onto the first principal component (1D reduction)
# Keep only the top component in V (which is the first row of Vt transposed)
V_1 = Vt[0:1, :].T
X_reduced = X_centered @ V_1

print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")
```

---

## Part 5: Practice Exercises

### Exercise 1: Finding Eigenvalues by Hand
Let $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$.
1. Write the characteristic polynomial $\det(A - \lambda I) = 0$.
2. Solve for the two eigenvalues $\lambda_1, \lambda_2$.
3. Plug each eigenvalue back into $(A - \lambda I)\mathbf{v} = \mathbf{0}$ to find the eigenvectors.
4. **Code it:** Verify your answers using `np.linalg.eig(A)`.

### Exercise 2: SVD Image Compression
1. Load a grayscale image (or generate a random 100x100 matrix to represent one).
2. Compute its SVD using `np.linalg.svd`.
3. Reconstruct the image using only the top 10 singular values. 
4. Calculate the percentage of the original data you needed to store. (Hint: A $100 \times 100$ matrix has 10,000 numbers. If $k=10$, you store $100 \times 10$ for $U$, $10$ for $S$, and $10 \times 100$ for $V^T$).

### Exercise 3: PCA Variance
Using the code from Part 4.3:
1. The singular values $S$ are related to the variance. Compute the total variance: $\text{Total} = \sum S^2$.
2. What percentage of the total variance is captured by the first principal component alone?
3. In a real ML pipeline, why do we usually keep enough components to explain 95% of the variance, rather than keeping all of them?

---

## Self-Test Questions

1. **Can a matrix have an eigenvalue of 0? What does that mean?** *(Yes. If $\lambda = 0$, then $A\mathbf{v} = 0\mathbf{v} = \mathbf{0}$. This means $\mathbf{v}$ is in the null space of $A$. So a matrix has a zero eigenvalue if and only if its null space is non-trivial, which means it is singular/non-invertible.)*
2. **What is the geometric difference between eigendecomposition ($A = V\Lambda V^{-1}$) and SVD ($A = U\Sigma V^T$)?** *(Eigendecomposition uses the same basis $V$ for input and output, stretching space along those skewed axes. SVD uses two different orthogonal bases ($V$ and $U$), meaning it rotates the input, scales it, then rotates it to a different output space.)*
3. **Why do we center data before running PCA?** *(If we don't, the first principal component will just point from the origin to the center of the data cloud, capturing the mean rather than the direction of maximum variance.)*
4. **If you have a 1000x1000 image, how many numbers do you need to store if you keep the top 50 singular values?** *(50,000 for $U_k$ + 50 for $\Sigma_k$ + 50,000 for $V_k^T$ = 100,050 numbers. This is a 10x compression ratio compared to 1,000,000 numbers!)*
