# Week 1: Vectors, Matrices, and Geometric Intuition

> **Goal:** See matrices as *linear transformations of space*, not grids of numbers. This single mental shift makes everything downstream easier.

---

## Part 1: Vectors — More Than Just Lists of Numbers

### 1.1 What Is a Vector?

At its core, a vector is an **ordered list of numbers**. In Machine Learning, a data point is almost always a vector. An image? A vector of pixel intensities. A word? A vector of learned embedding values. A protein? A vector of amino acid features.

But mathematically, vectors have a much richer interpretation. A vector $\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ isn't just the pair $(3, 4)$ — it's an **arrow** from the origin to the point $(3, 4)$ in 2D space. It has:

- A **magnitude** (length): $\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = 5$
- A **direction**: the angle it makes with the x-axis

This dual nature — *algebraic object* vs *geometric arrow* — is a theme that runs through all of linear algebra.

### 1.2 Vector Operations

**Addition** — Vectors add tip-to-tail. If $\mathbf{a} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$, then:

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} 1 + 3 \\ 2 + 1 \end{bmatrix} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}$$

Geometrically, you place $\mathbf{b}$'s tail at $\mathbf{a}$'s tip. The result is the diagonal of the parallelogram formed by the two vectors.

**Scalar multiplication** — Multiplying by a scalar $c$ stretches (or shrinks, or flips) the vector:

$$c\mathbf{v} = c \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} cv_1 \\ cv_2 \end{bmatrix}$$

- $c > 1$: stretches
- $0 < c < 1$: shrinks
- $c < 0$: reverses direction

**Linear combinations** — Any vector $\mathbf{w}$ that can be written as $\mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n$ is a *linear combination* of the vectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$. This is arguably the most important concept in linear algebra: **everything is built from linear combinations**.

### 1.3 The Span and Linear Independence

The **span** of a set of vectors is the set of *all possible linear combinations* of those vectors. If two vectors in $\mathbb{R}^2$ point in different directions, their span is the entire 2D plane. If they point in the same direction, their span is just a line — they are **linearly dependent**.

A set of vectors is **linearly independent** if no vector in the set can be written as a linear combination of the others. This matters because:

> In ML, if your features are linearly dependent, you have *redundant information*. This leads to numerical instability in regression, wasted computation, and the need for regularization.

### 1.4 Norms — Measuring Vector Size

The **norm** of a vector measures its "size." The most common norms:

**L2 norm (Euclidean):**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

This is the straight-line distance from the origin to the point. It's the default notion of "length."

**L1 norm (Manhattan):**
$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$$

The "taxicab distance" — you can only walk along grid lines. L1 is crucial in ML because it promotes **sparsity** (pushing some values to exactly zero). This is why Lasso regression uses L1 regularization.

**L∞ norm (Max norm):**
$$\|\mathbf{x}\|_\infty = \max_i |x_i|$$

Simply the largest absolute component. Used in adversarial ML attacks.

```python
import numpy as np

x = np.array([3, -4, 0, 5])

l2 = np.linalg.norm(x)           # Euclidean: sqrt(9+16+0+25) = sqrt(50)
l1 = np.linalg.norm(x, ord=1)    # Manhattan: 3+4+0+5 = 12
linf = np.linalg.norm(x, ord=np.inf)  # Max: 5

print(f"L2: {l2:.4f}, L1: {l1:.1f}, L∞: {linf:.1f}")
# L2: 7.0711, L1: 12.0, L∞: 5.0
```

---

## Part 2: The Dot Product — The Engine of Machine Learning

### 2.1 Algebraic Definition

Given two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

### 2.2 Geometric Definition

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos\theta$$

where $\theta$ is the angle between the vectors.

This is deeply important. The dot product simultaneously tells you about:
- **Similarity** of direction (via $\cos\theta$)
- **Magnitude** of both vectors

| Dot product | $\cos\theta$ | Meaning |
|---|---|---|
| $\mathbf{a} \cdot \mathbf{b} > 0$ | $\cos\theta > 0$ | Vectors point roughly the same way |
| $\mathbf{a} \cdot \mathbf{b} = 0$ | $\cos\theta = 0$ | Vectors are **orthogonal** (perpendicular) |
| $\mathbf{a} \cdot \mathbf{b} < 0$ | $\cos\theta < 0$ | Vectors point in opposite directions |

### 2.3 Why the Dot Product Is Everywhere in ML

Almost every neural network layer computes:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

This is the dot product of the weights $\mathbf{w}$ with the input $\mathbf{x}$, plus a bias. The network is literally asking: *"How much does this input align with the pattern encoded in my weights?"*

When you hear about:
- **Cosine similarity** in NLP → it's the normalized dot product
- **Attention** in Transformers → it's dot products between query and key vectors
- **Convolution** in CNNs → it's a dot product between a filter and a patch of the image

```python
import numpy as np

a = np.array([1, 0])
b = np.array([0, 1])
c = np.array([1, 1])

print(f"a · b = {np.dot(a, b)}")  # 0 → orthogonal (right angle)
print(f"a · c = {np.dot(a, c)}")  # 1 → 45 degrees

# Compute the angle between a and c
cos_theta = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
theta = np.degrees(np.arccos(cos_theta))
print(f"Angle between a and c: {theta:.1f}°")  # 45.0°
```

### 2.4 Inner Products and Orthogonality

The dot product is a specific case of an **inner product** — a function $\langle \cdot, \cdot \rangle$ that takes two vectors and returns a scalar, satisfying linearity, symmetry, and positive-definiteness.

Two vectors are **orthogonal** if $\langle \mathbf{a}, \mathbf{b} \rangle = 0$.

A set of vectors where *every pair is orthogonal* and each vector has unit length is called an **orthonormal basis**. These are ideal for computation because projections become trivially simple — this is why algorithms like QR decomposition and PCA produce orthonormal outputs.

---

## Part 3: Matrices as Linear Transformations

This is the **key mental shift** of the entire week.

### 3.1 What Is a Matrix, Really?

A matrix $A \in \mathbb{R}^{m \times n}$ is a rectangular grid of numbers with $m$ rows and $n$ columns. But the grid is not the point. A matrix is a **function** that maps vectors from one space to another:

$$A: \mathbb{R}^n \to \mathbb{R}^m$$

When you multiply $A\mathbf{x}$, you are *transforming* the vector $\mathbf{x}$.

### 3.2 Reading a Matrix Geometrically

Here's the critical insight from 3Blue1Brown: **The columns of a matrix tell you where the basis vectors land after the transformation.**

Consider the 2D identity basis vectors $\hat{\mathbf{i}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\hat{\mathbf{j}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

A matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ tells us:

- $\hat{\mathbf{i}}$ lands at $\begin{bmatrix} a \\ c \end{bmatrix}$ (first column)
- $\hat{\mathbf{j}}$ lands at $\begin{bmatrix} b \\ d \end{bmatrix}$ (second column)

Every other vector, being a linear combination of $\hat{\mathbf{i}}$ and $\hat{\mathbf{j}}$, goes wherever the linear combination says.

### 3.3 Classic Transformations

| Matrix | Name | Effect |
|---|---|---|
| $\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ | Scaling | Doubles everything |
| $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | Rotation | Rotates by $\theta$ |
| $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ | Shear | Tilts the x-axis |
| $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ | Reflection | Flips across x-axis |
| $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ | Projection | Collapses to x-axis |

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a unit square
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# Rotation by 45 degrees
theta = np.radians(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

rotated = R @ square

# Shear
S = np.array([[1, 0.5],
              [0, 1]])
sheared = S @ square

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(*square, 'b-', linewidth=2, label='Original')
axes[0].set_title('Original'); axes[0].set_xlim(-1, 2); axes[0].set_ylim(-1, 2)
axes[0].set_aspect('equal'); axes[0].grid(True)

axes[1].plot(*square, 'b--', alpha=0.3)
axes[1].plot(*rotated, 'r-', linewidth=2, label='Rotated 45°')
axes[1].set_title('Rotation (45°)'); axes[1].set_xlim(-1, 2); axes[1].set_ylim(-1, 2)
axes[1].set_aspect('equal'); axes[1].grid(True)

axes[2].plot(*square, 'b--', alpha=0.3)
axes[2].plot(*sheared, 'g-', linewidth=2, label='Sheared')
axes[2].set_title('Shear'); axes[2].set_xlim(-1, 2); axes[2].set_ylim(-1, 2)
axes[2].set_aspect('equal'); axes[2].grid(True)

plt.tight_layout()
plt.savefig('transformations.png', dpi=150)
plt.show()
```

### 3.4 Matrix Multiplication = Composition of Transformations

If $A$ rotates space and $B$ shears space, then $BA$ means: **first apply $A$, then apply $B$**.

This is why matrix multiplication is **not commutative** ($AB \neq BA$ in general). Rotating then shearing gives a different result than shearing then rotating. Just like in real life: putting on socks then shoes ≠ putting on shoes then socks.

**The mechanics of matrix multiplication:**

$$[AB]_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$

Each entry is a **dot product** of row $i$ of $A$ with column $j$ of $B$.

**Dimensions:** If $A$ is $m \times n$ and $B$ is $n \times p$, then $AB$ is $m \times p$. The inner dimensions must match.

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# These are NOT the same!
AB = A @ B
BA = B @ A

print(f"AB =\n{AB}")
# [[19 22]
#  [43 50]]

print(f"BA =\n{BA}")
# [[23 34]
#  [31 46]]

print(f"AB == BA? {np.array_equal(AB, BA)}")  # False
```

### 3.5 Implementing Matrix Multiplication from Scratch

Understanding the element-by-element mechanics is essential:

```python
def matmul(A, B):
    """Matrix multiplication using explicit loops.
    
    A: (m, n) matrix
    B: (n, p) matrix
    Returns: (m, p) matrix
    """
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"Inner dimensions don't match: {n} vs {n2}"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            # Dot product of row i of A with column j of B
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Verify against NumPy
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

ours = matmul(A, B)
numpy_result = A @ B

print(f"Max error: {np.max(np.abs(ours - numpy_result)):.2e}")
# Max error: ~1e-15 (numerical precision)
```

---

## Part 4: Systems of Linear Equations

### 4.1 The Problem

A system of linear equations is a set of equations like:

$$2x + 3y = 7$$
$$x - y = 1$$

In matrix form, this is $A\mathbf{x} = \mathbf{b}$, where:

$$\begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 7 \\ 1 \end{bmatrix}$$

**Geometric interpretation:** Each equation defines a line (or plane, or hyperplane). The solution is where they intersect.

### 4.2 When Does a Solution Exist?

Three possible outcomes:
1. **One unique solution** — the lines/planes intersect at exactly one point
2. **No solution** — the lines are parallel (inconsistent)
3. **Infinitely many solutions** — the lines overlap (underdetermined)

The **determinant** of $A$ tells you which case you're in for square systems:
- $\det(A) \neq 0$ → unique solution exists (invertible matrix)
- $\det(A) = 0$ → either no solution or infinitely many (singular matrix)

```python
import numpy as np

# System with a unique solution
A = np.array([[2, 3],
              [1, -1]])
b = np.array([7, 1])

print(f"det(A) = {np.linalg.det(A):.1f}")  # -5.0 ≠ 0 → unique solution

x = np.linalg.solve(A, b)
print(f"Solution: x = {x[0]:.1f}, y = {x[1]:.1f}")
# x = 2.0, y = 1.0

# Verify: A @ x should equal b
print(f"Verification: A @ x = {A @ x}")  # [7, 1] ✓
```

---

## Part 5: Vector Spaces — The Arena of Linear Algebra

### 5.1 What Is a Vector Space?

A **vector space** $V$ is a set of objects (vectors) where you can do two things:
1. **Add** any two vectors and stay in $V$
2. **Scalar multiply** any vector and stay in $V$

$\mathbb{R}^n$ (the set of all $n$-dimensional vectors with real entries) is the canonical example. But vector spaces can be abstract: the set of all polynomials of degree $\leq 3$, the set of all $2 \times 2$ matrices, etc.

### 5.2 Basis and Dimension

A **basis** of a vector space is a minimal set of vectors that spans the entire space. Every vector in the space can be uniquely written as a linear combination of the basis vectors.

The **dimension** of a vector space is the number of vectors in any basis.

For $\mathbb{R}^3$: the standard basis is $\{\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{e}}_3\}$, and the dimension is 3.

> **Why this matters for ML:** When we say a dataset "lives in a 768-dimensional space" (like BERT embeddings), we mean each data point is a vector in $\mathbb{R}^{768}$ with 768 basis directions. PCA finds that most of the information lives in a much lower-dimensional subspace.

---

## Part 6: The Dot Product in Two Ways — A Deep Verification

Let's compute a dot product both algebraically and geometrically and verify they agree.

**Problem:** Compute $\mathbf{a} \cdot \mathbf{b}$ where $\mathbf{a} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

**Algebraic:**
$$\mathbf{a} \cdot \mathbf{b} = 3(1) + 4(0) = 3$$

**Geometric:**
$$\|\mathbf{a}\| = \sqrt{9 + 16} = 5, \quad \|\mathbf{b}\| = 1$$
$$\theta = \arctan(4/3) \approx 53.13°$$
$$\mathbf{a} \cdot \mathbf{b} = 5 \times 1 \times \cos(53.13°) = 5 \times 0.6 = 3 \quad \checkmark$$

```python
import numpy as np

a = np.array([3, 4])
b = np.array([1, 0])

# Algebraic
algebraic = np.dot(a, b)
print(f"Algebraic: {algebraic}")  # 3

# Geometric
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
cos_theta = np.dot(a, b) / (norm_a * norm_b)
theta_deg = np.degrees(np.arccos(cos_theta))
geometric = norm_a * norm_b * np.cos(np.radians(theta_deg))
print(f"Geometric: {geometric:.4f}")  # 3.0000
print(f"Angle: {theta_deg:.2f}°")    # 53.13°
```

---

## Part 7: Matrix Multiplication = Composition — Proof by Example

**Claim:** Matrix multiplication represents the composition of two linear maps.

Let $A$ be a rotation by 90° and $B$ be a horizontal scaling by 2:

$$A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$

Track where the basis vectors land when we apply $B$ first, then $A$:

**Step 1: Apply $B$ to $\hat{\mathbf{i}}$:**
$$B\hat{\mathbf{i}} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$$

**Step 2: Apply $A$ to the result:**
$$A\begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \end{bmatrix}$$

**Now compute $AB$ and check:**
$$AB = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 2 & 0 \end{bmatrix}$$

First column of $AB$ = $\begin{bmatrix} 0 \\ 2 \end{bmatrix}$ ✓ — exactly where $\hat{\mathbf{i}}$ landed after both transformations!

---

## Part 8: 3×3 Matrix Product — Element by Element

Expanding a $3 \times 3$ product is tedious but deeply instructive. Let's do it:

$$C = AB = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \\ 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 0 \\ 3 & 0 & 1 \end{bmatrix}$$

$C_{11} = 1(1) + 2(0) + 0(3) = 1$

$C_{12} = 1(0) + 2(1) + 0(0) = 2$

$C_{13} = 1(2) + 2(0) + 0(1) = 2$

$C_{21} = 0(1) + 1(0) + 3(3) = 9$

$C_{22} = 0(0) + 1(1) + 3(0) = 1$

$C_{23} = 0(2) + 1(0) + 3(1) = 3$

$C_{31} = 1(1) + 0(0) + 1(3) = 4$

$C_{32} = 1(0) + 0(1) + 1(0) = 0$

$C_{33} = 1(2) + 0(0) + 1(1) = 3$

$$C = \begin{bmatrix} 1 & 2 & 2 \\ 9 & 1 & 3 \\ 4 & 0 & 3 \end{bmatrix}$$

```python
import numpy as np

A = np.array([[1, 2, 0], [0, 1, 3], [1, 0, 1]])
B = np.array([[1, 0, 2], [0, 1, 0], [3, 0, 1]])

C = A @ B
print(C)
# [[ 1  2  2]
#  [ 9  1  3]
#  [ 4  0  3]]
```

---

## Part 9: Practice Exercises

To truly master linear algebra, you must practice. Complete these exercises before moving on to Week 2.

### Exercise 1: Geometric Transformations
You are given a triangle with vertices at $A(0,0)$, $B(2,0)$, and $C(1,2)$. 
1. Write these vertices as a $2 \times 3$ matrix $V$.
2. Apply a scaling matrix $S = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ to $V$. What are the new vertices?
3. Apply a 90° counterclockwise rotation matrix $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ to $V$. 
4. **Code it:** Write a short Python script using NumPy and Matplotlib to plot the original triangle and both transformations.

### Exercise 2: The Non-Commutativity of Matrices
Let $A = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$ and $B = \begin{bmatrix} 1 & 0 \\ 3 & 1 \end{bmatrix}$.
1. Compute $AB$ by hand.
2. Compute $BA$ by hand.
3. Compare the results. What does this tell you geometrically about applying a horizontal shear versus a vertical shear?

### Exercise 3: Inner Products and Orthogonality
Consider the vectors $\mathbf{u} = \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}$, $\mathbf{v} = \begin{bmatrix} 2 \\ -1 \\ 0 \end{bmatrix}$, and $\mathbf{w} = \begin{bmatrix} 1 \\ 2 \\ 5 \end{bmatrix}$.
1. Compute $\mathbf{u} \cdot \mathbf{v}$. Are they orthogonal?
2. Compute $\mathbf{u} \cdot \mathbf{w}$. Are they orthogonal?
3. Compute $\mathbf{v} \cdot \mathbf{w}$. Are they orthogonal?
4. **Code it:** Verify your answers using `np.dot()`.

### Exercise 4: Solving a Linear System
Translate the following word problem into a matrix equation $A\mathbf{x} = \mathbf{b}$ and solve it:
*A machine learning model uses two features: $x_1$ (hours studied) and $x_2$ (practice tests taken). For student A, 3 hours studied and 1 practice test yields a score of 85. For student B, 2 hours studied and 2 practice tests yields a score of 90. Find the exact linear weights the model has learned.*

1. Write the system of equations.
2. Formulate $A$ and $\mathbf{b}$.
3. **Code it:** Use `np.linalg.solve(A, b)` to find the weights.

---

## Self-Test Questions

Before moving on to Week 2, make sure you can answer these **without looking at your notes**:

1. **What does a dot product of zero mean geometrically?** *(The vectors are orthogonal — they point in completely independent directions.)*

2. **Why is $AB \neq BA$ in general?** *(Because matrix multiplication represents composition of transformations, and the order of transformations matters — rotating then shearing gives a different result than shearing then rotating.)*

3. **What do the columns of a matrix represent?** *(Where the basis vectors land after the transformation. The first column is the image of $\hat{\mathbf{i}}$, the second is the image of $\hat{\mathbf{j}}$, etc.)*

4. **If $A$ is $3 \times 2$ and $\mathbf{x}$ is $2 \times 1$, what is the dimension of $A\mathbf{x}$?** *($3 \times 1$ — the transformation maps from $\mathbb{R}^2$ to $\mathbb{R}^3$.)*

5. **Write a Python expression for the dot product of `[1, 0]` and `[0, 1]` and predict the result before running it.** *(`np.dot([1,0], [0,1])` → result is `0`, because these are orthogonal basis vectors.)*

6. **What is the difference between L1 and L2 norms, and when would you prefer each?** *(L2 is Euclidean distance and penalizes large values smoothly. L1 promotes sparsity — it drives some components to exactly zero, useful when you want feature selection.)*

---

## What's Next

In **Week 2**, we will go deeper into the structure of linear systems: *When does $Ax = b$ have 0, 1, or infinitely many solutions?* We will learn about **rank**, the **four fundamental subspaces** of a matrix, and **projections** — the geometric foundation of least-squares regression, which is the starting point for all of machine learning.
