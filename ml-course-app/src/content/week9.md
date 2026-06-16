# Week 10: NumPy Mastery & Vectorization

> **Goal:** Vectorization fluency. This single skill most determines how fast you will move once training loops appear. If you write a `for` loop in NumPy, you are doing it wrong.

---

## Part 1: Why Vectorize?

Python is a beautifully readable language, but it is incredibly slow for numerical loops.
When you write a `for` loop in Python:
1. Python checks the type of the variable every iteration.
2. Python manages memory dynamically for every operation.
3. Python cannot leverage CPU-level SIMD (Single Instruction, Multiple Data) instructions automatically.

**NumPy** fixes this by pushing the loop down to highly optimized C or Fortran code. You describe the operation on the *entire array* at once, and NumPy does the looping instantly in C.

### 1.1 The Rule of Vectorization
If you see a `for` loop iterating over the elements of an array, replace it with a NumPy built-in function or a vectorized operator.

```python
import numpy as np
import time

# A massive array of 10 million numbers
arr = np.random.rand(10000000)

# BAD: A Python for loop
start = time.time()
total = 0
for x in arr:
    total += x
end = time.time()
print(f"Python loop took {end - start:.4f} seconds")

# GOOD: NumPy Vectorization
start = time.time()
total_np = np.sum(arr)
end = time.time()
print(f"NumPy vectorization took {end - start:.4f} seconds")
# The NumPy version will be roughly 100x to 500x faster!
```

---

## Part 2: Broadcasting

Broadcasting is NumPy's superpower. It allows you to perform operations on arrays of different shapes without having to explicitly write loops or duplicate data in memory.

### 2.1 The Rules of Broadcasting
When operating on two arrays, NumPy compares their shapes element-wise, starting from the trailing dimensions (the right-most side) and working left. Two dimensions are compatible if:
1. They are equal.
2. One of them is 1.

If a dimension is 1, NumPy will conceptually "stretch" or "broadcast" that dimension to match the other array's dimension.

### 2.2 Example: Mean Centering a Matrix
Suppose you have a dataset of 1000 samples and 5 features (shape `1000 x 5`). You want to subtract the mean of each feature from every sample.

```python
import numpy as np

X = np.random.rand(1000, 5)

# Calculate the mean of each feature (collapse axis 0)
# The shape of X_mean is (5,)
X_mean = np.mean(X, axis=0) 

# Subtract the mean! 
# X has shape (1000, 5)
# X_mean has shape (5,)
# NumPy implicitly treats X_mean as (1, 5) and broadcasts it to (1000, 5)
X_centered = X - X_mean 
```

---

## Part 3: Fancy Indexing and Boolean Masks

You don't need loops to filter data. You can index arrays using other arrays or boolean masks.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Boolean Masking
# Find all elements greater than 25
mask = arr > 25  # Returns [False, False, True, True, True]
filtered = arr[mask] # Returns [30, 40, 50]

# Fancy Indexing
# Grab elements at specific indices: 0, 1, and 4
indices = [0, 1, 4]
grabbed = arr[indices] # Returns [10, 20, 50]
```

---

## Part 4: Einstein Summation (`einsum`)

`np.einsum` is the final boss of NumPy. It allows you to express almost any multi-dimensional tensor operation (dot products, transposes, batched matrix multiplications, traces) using a concise string syntax.

### 4.1 The Rules of Einsum
The string defines the indices of the input arrays and the output array.
- **Repeated indices** between input arrays mean *multiply those elements*.
- **Omitted indices** in the output mean *sum over those indices*.

### 4.2 Common Einsum Patterns

1. **Matrix Transpose** $A^T$:
   `np.einsum('ij->ji', A)`
2. **Diagonal / Trace** $\sum_i A_{ii}$:
   `np.einsum('ii->', A)`
3. **Dot Product** $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$:
   `np.einsum('i,i->', a, b)`
4. **Matrix-Vector Multiplication** $A\mathbf{x} = \sum_j A_{ij} x_j$:
   `np.einsum('ij,j->i', A, x)`
5. **Matrix Multiplication** $AB = \sum_k A_{ik} B_{kj}$:
   `np.einsum('ik,kj->ij', A, B)`
6. **Batched Matrix Multiplication** (vital for Deep Learning!):
   `np.einsum('bij,bjk->bik', A, B)` (where `b` is the batch size).

---

## Part 5: Practice Exercises

### Exercise 1: Pairwise Distances (No Loops!)
You have $N$ points in 3D space, represented as an $N \times 3$ array `P`. You want to find the Euclidean distance between *every pair* of points, resulting in an $N \times N$ matrix.
Doing this with two `for` loops is incredibly slow. Let's vectorize it using broadcasting.

Recall that $(a - b)^2 = a^2 - 2ab + b^2$.
We can do this for vectors: $\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 - 2\mathbf{x} \cdot \mathbf{y} + \|\mathbf{y}\|^2$.

```python
import numpy as np

N = 100
P = np.random.rand(N, 3)

# 1. Compute the squared norm of each point: ||x||^2
# shape (N, 1)
sq_norms = np.sum(P**2, axis=1, keepdims=True)

# 2. Compute the cross term: -2 * x * y
# shape (N, N)
cross_term = -2 * (P @ P.T) 

# 3. Use broadcasting to add them all together!
# sq_norms is (N, 1), sq_norms.T is (1, N)
# NumPy broadcasts both to (N, N)
dist_squared = sq_norms + cross_term + sq_norms.T

# Ensure no negative numbers due to floating point inaccuracies, then square root
distances = np.sqrt(np.maximum(dist_squared, 0))
print(f"Distance matrix shape: {distances.shape}")
```

### Exercise 2: Einsum Linear Regression
Re-implement the Normal Equations $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ from Week 2, but use `np.einsum` for all matrix multiplications instead of `@`.

1. Compute $X^T X$ using `einsum('ji,jk->ik', X, X)`. (Wait, if $X$ is $N \times D$, $X^T$ is $D \times N$. Using `einsum('ni,nj->ij', X, X)` is even cleaner!)
2. Compute $X^T \mathbf{y}$ using `einsum('ni,n->i', X, y)`.
3. Solve for $\mathbf{w}$.

---

## Self-Test Questions
1. **If array A is shape `(4, 1, 3)` and array B is shape `(1, 5, 3)`, what is the shape of `A + B`?** *(Broadcasting stretches the 1s. The result is `(4, 5, 3)`.)*
2. **Why is `np.einsum('ij,j->i', A, x)` faster than writing a nested Python loop?** *(Because `einsum` is evaluated in highly optimized C code that skips the Python interpreter overhead entirely.)*
3. **What is a boolean mask and how does it avoid loops?** *(It is an array of True/False values the exact same shape as your data. You pass the mask directly into the data array as an index, and NumPy's C backend instantly returns only the True elements, skipping Python iteration.)*
