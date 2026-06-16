# Week 19: Unsupervised Learning: Clustering, GMMs, & Dimensionality Reduction

> **Goal:** Explore structure in data without explicit labels. You will learn the mechanics of K-Means clustering, understand how Gaussian Mixture Models (GMMs) provide soft classification, and study non-linear dimensionality reduction techniques like t-SNE and UMAP, learning how to use them safely.

---

## Part 1: K-Means Clustering

K-Means is a centroid-based clustering algorithm that partitions a dataset $X = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(m)}\}$ into $K$ distinct, non-overlapping clusters.

### 1.1 The Objective Function (Inertia)
K-Means aims to minimize the within-cluster sum-of-squares (inertia):

$$ J(C_1, \dots, C_K, \mathbf{\mu}_1, \dots, \mathbf{\mu}_K) = \sum_{k=1}^K \sum_{\mathbf{x}^{(i)} \in C_k} \|\mathbf{x}^{(i)} - \mathbf{\mu}_k\|_2^2 $$

where $\mathbf{\mu}_k$ is the mean (centroid) of cluster $C_k$.

### 1.2 The Lloyd's Algorithm
The optimization is performed iteratively:
1. **Initialize:** Choose $K$ centroids $\{\mathbf{\mu}_1, \dots, \mathbf{\mu}_K\}$ randomly.
2. **Assign step:** Assign each sample $\mathbf{x}^{(i)}$ to the closest centroid:
   $$ C_k = \left\{ \mathbf{x}^{(i)} : k = \arg\min_j \|\mathbf{x}^{(i)} - \mathbf{\mu}_j\|_2^2 \right\} $$
3. **Update step:** Recalculate centroids as the mean of all samples assigned to that cluster:
   $$ \mathbf{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}^{(i)} \in C_k} \mathbf{x}^{(i)} $$
4. **Repeat** steps 2 and 3 until centroids stop changing.

### 1.3 Finding the Optimal $K$
- **Elbow Method:** Plot the inertia $J$ vs. $K$. As $K$ increases, inertia always decreases. The "elbow" is the point where the rate of decrease drops sharply.
- **Silhouette Coefficient:** Measures how similar a point is to its own cluster compared to other clusters. Values range from $[-1, 1]$, where higher is better.

---

## Part 2: Gaussian Mixture Models (GMMs) & Expectation-Maximization

K-Means assumes clusters are spherical and assigns each point to exactly one cluster (hard clustering). **Gaussian Mixture Models** assume that data is generated from a mixture of several Gaussian distributions, allowing for elliptical clusters and **soft assignment** (probabilities).

### 2.1 The Probability Density
The probability density of a sample $\mathbf{x}$ in a GMM with $K$ components is:

$$ p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \mathbf{\mu}_k, \Sigma_k) $$

where $\pi_k$ is the mixture weight of component $k$ ($\sum \pi_k = 1$), and $\mathcal{N}$ is a multivariate normal distribution with mean $\mathbf{\mu}_k$ and covariance matrix $\Sigma_k$.

### 2.2 The Expectation-Maximization (EM) Algorithm
Since we don't know which Gaussian generated which data point, we treat cluster memberships as latent variables. We optimize the parameters $(\pi, \mathbf{\mu}, \Sigma)$ using EM:
1. **Expectation Step (E-step):** Estimate the probability (responsibility) $\gamma_{ik}$ that component $k$ generated point $\mathbf{x}^{(i)}$:
   $$ \gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}^{(i)} \mid \mathbf{\mu}_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}^{(i)} \mid \mathbf{\mu}_j, \Sigma_j)} $$
2. **Maximization Step (M-step):** Update the parameters using the responsibilities as weights:
   $$ \mathbf{\mu}_k^{\text{new}} = \frac{\sum_{i=1}^m \gamma_{ik} \mathbf{x}^{(i)}}{\sum_{i=1}^m \gamma_{ik}} $$
   $$ \Sigma_k^{\text{new}} = \frac{\sum_{i=1}^m \gamma_{ik} (\mathbf{x}^{(i)} - \mathbf{\mu}_k^{\text{new}})(\mathbf{x}^{(i)} - \mathbf{\mu}_k^{\text{new}})^T}{\sum_{i=1}^m \gamma_{ik}} $$
   $$ \pi_k^{\text{new}} = \frac{1}{m} \sum_{i=1}^m \gamma_{ik} $$

---

## Part 3: Dimensionality Reduction for Visualization

While PCA finds linear directions of maximum variance, non-linear techniques are required to project complex manifolds (e.g., biological clusters) into 2D.

### 3.1 t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Maps pairwise distances in high-dimensional space to probabilities representing similarities.
- Minimizes the Kullback-Leibler (KL) divergence between high-dimensional probabilities and low-dimensional student-t probabilities.
- **Limitation:** Preserves **local structure** (similar things stay close) but destroys **global structure** (distances between distant clusters are meaningless).

### 3.2 UMAP (Uniform Manifold Approximation and Projection)
- Grounded in Riemannian geometry and algebraic topology.
- Preserves both **local** and **global structure** better than t-SNE.
- Much faster computationally than t-SNE, allowing it to scale to millions of samples.

> [!WARNING]
> **Do not use t-SNE or UMAP coordinates as features for downstream classifiers.** They are stochastic, non-invertible, and distort distance scales. Instead, train downstream classifiers on raw features or linear PCA projections.

---

## Part 4: Practice Exercises

### Exercise 1: K-Means from Scratch
1. Generate a synthetic dataset with 3 clear 2D clusters (using `sklearn.datasets.make_blobs`).
2. Implement Lloyd's K-Means algorithm from scratch using NumPy.
3. **Code it:** Run your function, plot the cluster assignments, and compare the final centroids to those computed by `sklearn.cluster.KMeans`.

### Exercise 2: Clustering Protein Embeddings
1. Generate a simulated high-dimensional feature set (e.g., shape $1000 \times 128$) containing 3 hidden classes.
2. Run PCA to reduce it to 50 dimensions, then apply UMAP to project it to 2D.
3. **Code it:** Plot the 2D UMAP projection twice: once colored by a K-Means clustering assignment ($K=3$), and once colored by a GMM soft-probability threshold.

---

## Self-Test Questions

1. **Why is K-Means highly sensitive to centroid initialization?** *(Because the objective function (inertia) is non-convex and has many local minima. A bad random initialization can cause centroids to get stuck in local configurations. We mitigate this using `K-means++` initialization.)*
2. **What is the structural difference between K-Means and GMMs?** *(K-Means assumes spherical clusters and performs hard clustering. GMMs support covariance matrices that capture ellipsoidal clusters of varying sizes and orientations, and assign points probabilistic (soft) memberships.)*
3. **Why is it misleading to compare distances between different clusters on a t-SNE plot?** *(t-SNE uses dynamic perplexity to adjust local density scales. It expands dense clusters and compresses sparse ones, meaning global distances between distant clusters are arbitrary.)*
4. **Why is the Student-t distribution used in t-SNE instead of a Gaussian in the low-dimensional space?** *(To solve the "crowding problem." In 2D, there is much less volume available to accommodate moderate distances than in high-dimensional space. The heavy tails of the Student-t distribution allow points to be placed further apart without inflating the error.)*
