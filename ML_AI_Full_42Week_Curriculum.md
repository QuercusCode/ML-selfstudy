# ML/AI Learning Path — Foundations Block (Weeks 1–12)

**Math (Weeks 1–8) + Python (Weeks 9–12)** · ~10 h/week · ~3 months

This is the detailed expansion of the foundations block. Every week lists: the goal, exact chapters/sections/lectures from each resource, what to derive *by hand*, what to *code*, the deliverable, and a self-test. Treat the schedule as a target, not a contract — repeat a week if it hasn't clicked, skip ahead if it has.

---

## How to use this document

**Spine textbook.** *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) — free at **mml-book.com**. Everything math orbits this. The other resources are intuition-builders (3Blue1Brown, StatQuest) or specialists (Strang, Stat 110, Bishop) around it.

**Weekly rhythm (~10 h):**
- ~4 h primary reading (textbook + lectures)
- ~1.5 h intuition videos
- ~4 h coding / hand-derivation
- ~0.5 h notes + spaced review

**Two non-negotiable habits:**
1. **Derive before you code.** For every formula, write the derivation on paper *once* before implementing it. The motor memory of expanding a chain rule beats re-watching a video three times.
2. **Daily beats binge.** 1–1.5 h/day, especially for math — it consolidates with sleep.

**Self-test rule.** At each weekly checkpoint, if you can't reconstruct the core result from a blank page, you haven't finished the week yet.

---

# PART I — MATHEMATICS (Weeks 1–8)

---

## Week 1 — Vectors, matrices, geometric intuition

**Goal:** See matrices as *linear transformations of space*, not grids of numbers. This single mental shift makes everything downstream easier.

**Read**
- MML **§2.1–2.4** (Systems of Linear Equations, Matrices, Solving Systems, Vector Spaces)
- MML **§3.1–3.4** (Norms, Inner Products, Lengths/Distances, Angles & Orthogonality)

**Watch**
- 3Blue1Brown *Essence of Linear Algebra* — **chapters 1–8** (Vectors → Nonsquare matrices). ~1.5 h total.
- Strang MIT 18.06 — **Lectures 1–3** (Geometry of linear equations, Elimination, Multiplication & inverse matrices)

**Derive by hand**
- Expand a 3×3 matrix–matrix product element by element.
- Show that matrix multiplication = composition of two linear maps (track where the basis vectors land).
- Compute a dot product two ways: algebraically and as `|a||b|cosθ`.

**Code**
- Notebook that applies a 2×2 matrix to the unit square / a grid of points and plots before-and-after (rotation, shear, scaling, reflection).
- Implement `matmul(A, B)` with explicit loops, verify against `np.dot`.

**Deliverable:** "Linear transformations visualized" notebook — at least 4 transformations with plots and a one-line geometric description of each.

**Self-test:** Without notes, explain why `AB ≠ BA` in geometric terms, and what the columns of a matrix represent.

---

## Week 2 — Linear systems, rank, the four subspaces, projections

**Goal:** Understand solvability (when does `Ax = b` have 0/1/∞ solutions?) and projection — the geometric heart of least squares.

**Read**
- MML **§2.5–2.8** (Linear Independence, Basis & Rank, Linear Mappings, Affine Spaces)
- MML **§3.8** (Orthogonal Projections) — read closely, this returns in regression

**Watch**
- 3B1B *Essence of LA* — **ch. 7** (Inverse, column space, null space), **ch. 9** (Dot products & duality)
- Strang 18.06 — **Lectures 4–10** (LU; Transposes/Permutations/Spaces; Column space & nullspace; Solving Ax=0; Solving Ax=b; Independence/Basis/Dimension; The four fundamental subspaces)

**Derive by hand**
- Solve a 4×4 system by Gaussian elimination; identify pivot vs free variables.
- Project a vector onto a line, then onto a plane; derive the projection matrix `P = A(AᵀA)⁻¹Aᵀ`.

**Code**
- Gaussian elimination with partial pivoting, from scratch.
- Compute rank (`np.linalg.matrix_rank`), null space (via SVD), and verify `rank + nullity = n`.
- Projection function; project random points onto a chosen subspace and plot residuals.

**Deliverable:** Elimination + projection notebook; include one worked example where `Ax = b` has no exact solution and you find the least-squares projection instead.

**Self-test:** Name the four fundamental subspaces and their dimensions for a given `A`. Explain why least squares is "projection onto the column space."

---

## Week 3 — Eigenvalues, eigenvectors, SVD, PCA

**Goal:** The most important week of linear algebra for ML. SVD and PCA underlie dimensionality reduction, embeddings, and a lot of bio data analysis.

**Read**
- MML **Chapter 4** in full: §4.1 Determinant & Trace, §4.2 Eigenvalues & Eigenvectors, §4.3 Cholesky, §4.4 Eigendecomposition & Diagonalization, §4.5 **SVD**, §4.6 Matrix Approximation, §4.7 Matrix Phylogeny
- MML **§10.1–10.4** (PCA: problem setting, max-variance & projection perspectives)

**Watch**
- 3B1B *Essence of LA* — **ch. 14** (Eigenvectors & eigenvalues), **ch. 15** (Abstract vector spaces)
- Strang 18.06 — **Lectures 21–22** (Eigenvalues/eigenvectors; Diagonalization & powers of A), **Lecture 29** (SVD)
- StatQuest — *PCA Step-by-Step* + *SVD* videos

**Derive by hand**
- Find eigenvalues/eigenvectors of a 2×2 matrix from `det(A − λI) = 0`.
- Show how `A = UΣVᵀ` relates to the eigendecomposition of `AᵀA` and `AAᵀ`.
- Derive PCA as either max-variance projection *or* minimum reconstruction error (do one fully).

**Code**
- Power iteration to find the dominant eigenvector; compare with `np.linalg.eig`.
- **SVD image compression:** load a grayscale image, truncate to top-k singular values, plot reconstruction vs k.
- **PCA from scratch** (center → covariance → eig, *and* via SVD); apply to a real dataset; compare to `sklearn.decomposition.PCA`. Use one of your own datasets (e.g., ESM embeddings or BGC feature vectors) for the bio version.

**Deliverable:** PCA-from-scratch notebook that matches sklearn to numerical tolerance, plus the SVD compression demo.

**Self-test:** Explain what singular values *mean*, and why PCA = SVD of the centered data matrix. Reconstruct the PCA algorithm from a blank page.

---

## Week 4 — Calculus, gradients, Jacobians, the chain rule

**Goal:** Backprop is the multivariate chain rule applied systematically. Own the chain rule and backprop stops being magic.

**Read**
- MML **Chapter 5** in full: §5.1 Univariate Differentiation, §5.2 Partial Differentiation & Gradients, §5.3 Gradients of Vector-Valued Functions, §5.4 Gradients of Matrices, §5.5 Useful Identities, **§5.6 Backpropagation & Automatic Differentiation**, §5.7 Higher-Order Derivatives, §5.8 Linearization & Taylor

**Watch**
- 3B1B *Essence of Calculus* — **all chapters** (~3 h); prioritize derivatives, chain rule, implicit differentiation
- Khan Academy *Multivariable Calculus* — Gradient, Partial derivatives, Jacobian units
- 3B1B *Neural Networks* — **ch. 3 & 4** (What is backprop / Backprop calculus) as a bridge

**Derive by hand**
- Gradients of: **MSE loss**, **cross-entropy loss**, and **softmax** (the softmax Jacobian is the key exercise — do it fully).
- Chain rule through a 2-layer network: `x → Wx+b → ReLU → Wx+b → loss`. Get `∂L/∂W` for both layers.

**Code**
- Finite-difference **gradient checker**: numerically estimate gradients and compare to your analytic ones.
- Implement softmax + cross-entropy and verify the gradient numerically.

**Deliverable:** "Gradients by hand, verified by code" notebook — at least MSE, cross-entropy, softmax, each with hand derivation (photo/markdown) + numerical check passing.

**Self-test:** Derive the softmax Jacobian and the full gradient of a 2-layer MLP from scratch.

---

## Week 5 — Matrix calculus + optimization

**Goal:** Move from scalar gradients to matrix/vector calculus, then implement the optimizers you'll use forever.

**Read**
- **"The Matrix Calculus You Need For Deep Learning"** (Parr & Howard) — free at **explained.ai/matrix-calculus** — read in full (~30 pp)
- MML **Chapter 7**: §7.1 Optimization Using Gradient Descent, §7.2 Constrained Optimization & Lagrange Multipliers, §7.3 Convex Optimization

**Watch (optional depth)**
- Stephen Boyd, Stanford **EE364a** — Lectures 1–4 (Intro, Convex sets, Convex functions, Convex problems). Optional unless optimization grabs you.

**Derive by hand**
- The gradient and Hessian of a quadratic `½xᵀAx − bᵀx`.
- One Lagrange-multiplier constrained-optimization problem end to end.
- The momentum and Adam update equations — write out each term.

**Code**
- From scratch, implement and compare: **vanilla GD, SGD, Momentum, Nesterov, RMSProp, Adam**.
- Visualize their trajectories on (a) a quadratic bowl and (b) the Rosenbrock function. Plot loss-vs-iteration for each.

**Deliverable:** "Optimizers from scratch" notebook with trajectory plots — you should *see* why momentum helps in ravines and why Adam adapts step sizes.

**Self-test:** Write the Adam update from memory and explain what each moving average does.

---

## Week 6 — Probability, distributions, Bayes

**Goal:** Fluency with probability as the language of ML — uncertainty, likelihood, and Bayesian updating.

**Read**
- MML **Chapter 6**: §6.1 Probability Space, §6.2 Discrete & Continuous Probabilities, §6.3 Sum/Product Rules & **Bayes' Theorem**, §6.4 Summary Statistics & Independence, §6.5 **Gaussian Distribution**, §6.6 Conjugacy & Exponential Family, §6.7 Change of Variables

**Watch**
- Harvard **Stat 110** (Joe Blitzstein) — **Lectures 1–10** (Probability & counting, Story proofs, Conditional probability, Bayes, Independence, Monty Hall, Random variables, Distributions, Expectation)
- StatQuest — Probability vs Likelihood, common distributions

**Derive by hand**
- Bayes' theorem applied to a **diagnostic test** (base rate, sensitivity, specificity → posterior). This is the canonical intuition-builder.
- Mean and variance of Bernoulli, Binomial, and Gaussian from their definitions.

**Code**
- Monte Carlo estimation (estimate π; verify the law of large numbers).
- Bayesian update notebook: prior → likelihood → posterior, with plots, on the diagnostic-test or a coin-bias example.
- Sample from and plot the key distributions; overlay theoretical PDFs.

**Deliverable:** "Bayes in practice" notebook with the diagnostic-test problem solved analytically *and* by simulation, agreeing.

**Self-test:** Explain the difference between probability and likelihood, and update a posterior by hand for a small example.

---

## Week 7 — Statistical inference: MLE, MAP, exponential family

**Goal:** Connect probability to *fitting models*. MLE is what most loss functions secretly are.

**Read**
- Bishop **PRML** — **Ch. 1** §1.1–1.6 (Curve fitting, Probability theory, Model selection, Curse of dimensionality, Decision theory, Information theory) and **Ch. 2** §2.1–2.5 (Binary, Multinomial, Gaussian, Exponential family, Nonparametric) — read the key sections, skip heavy proofs
- MML **Chapter 8** §8.1–8.4 (Empirical Risk Minimization, Parameter Estimation **MLE/MAP**, Probabilistic Modeling)

**Watch**
- Stat 110 — **Lectures 11–20** (continuous RVs, Normal, Exponential, joint distributions, conditional expectation)
- StatQuest — Maximum Likelihood, MLE for the Normal distribution

**Derive by hand**
- **MLE for a Gaussian** (mean and variance) — the foundational derivation.
- **MLE for Bernoulli** `p`.
- Show that minimizing MSE = MLE under Gaussian noise; cross-entropy = MLE under a categorical model. (This is the "aha" that links Week 4 to Week 6.)
- One **MAP** estimate with a prior; show how the prior becomes a regularizer.

**Code**
- Implement the Gaussian/Bernoulli MLEs; verify against `numpy`/`scipy` estimates.
- Show numerically that L2 regularization ≈ Gaussian prior (MAP).

**Deliverable:** "From likelihood to loss" notebook tying MLE → MSE/cross-entropy and MAP → regularization, with derivations and numerical checks.

**Self-test:** Derive MLE for a Gaussian from scratch and explain why "minimize cross-entropy" = "maximize likelihood."

---

## Week 8 — Information theory + numerical stability

**Goal:** Entropy/KL/cross-entropy (the basis of classification losses) and the numerical tricks that keep training from exploding.

**Read**
- MacKay, *Information Theory, Inference, and Learning Algorithms* (free PDF) — **Ch. 2** (Probability, Entropy, Inference) and **Ch. 4** (The Source Coding Theorem) for entropy intuition
- Goodfellow et al., *Deep Learning* (free at deeplearningbook.org) — **§3.13** (Information Theory) and **Ch. 4 Numerical Computation**: §4.1 Overflow/Underflow, §4.2 Poor Conditioning, §4.3 Gradient-Based Optimization

**Watch**
- StatQuest — Entropy, Cross-Entropy, KL Divergence

**Derive by hand**
- Entropy, cross-entropy, and KL divergence definitions; show `cross-entropy = entropy + KL`.
- Why naive `softmax` overflows, and how subtracting the max fixes it (the log-sum-exp trick).

**Code**
- Implement **stable softmax**, **log-sum-exp**, and **numerically stable cross-entropy**; demonstrate the unstable version overflowing on large logits.
- Compute entropy/KL/cross-entropy between empirical distributions.

**Deliverable:** **ML loss-function cheat sheet** — MSE, MAE, cross-entropy, KL, hinge — each with its formula, gradient, the probabilistic story (which MLE it corresponds to), and when to use it. This becomes a permanent reference.

**Self-test:** Define KL divergence, explain why it's not symmetric, and write a numerically stable softmax from memory.

---

### ▣ Math block capstone (end of Week 8)

Spend an extra session building one artifact that exercises the whole block:

> **Implement linear/logistic regression from scratch in pure NumPy**, trained by gradient descent, with: your own gradient derivation, a gradient checker (Wk 4), an optimizer you wrote (Wk 5), the MLE-justified loss (Wk 7), and a stable softmax for the multiclass case (Wk 8). Evaluate with PCA-based visualization (Wk 3).

If you can build that without external ML libraries, the math block is done.

---

# PART II — PYTHON (Weeks 9–12)

You use Python daily, but research Python is often "scripts that run." ML needs *idiomatic, vectorized, engineered* Python. These four weeks close that gap.

---

## Week 9 — Idiomatic Python

**Goal:** Write Python the way the ML ecosystem writes it — data model, generators, decorators, type hints.

**Read**
- Ramalho, *Fluent Python* (2nd ed.):
  - **Ch. 1** The Python Data Model (dunder methods)
  - **Ch. 2** An Array of Sequences
  - **Ch. 3** Dictionaries and Sets
  - **Ch. 5** Data Class Builders (`dataclass`, `NamedTuple`)
  - **Ch. 7** Functions as First-Class Objects
  - **Ch. 8** Type Hints in Functions
  - **Ch. 9** Decorators and Closures
  - **Ch. 17** Iterators, Generators, and Classic Coroutines
  - **Ch. 18** `with`, `match`, and `else` Blocks (context managers)

**Code**
- Refactor one of your existing analysis scripts to use: a generator pipeline (instead of building big lists), a `dataclass` for structured records, a custom context manager, and full type hints + docstrings.
- Write one custom decorator (e.g., a `@timed` profiler) and one custom iterator class.

**Deliverable:** A before/after refactor of a real script, with a short note on what each idiom bought you (memory, readability, safety).

**Self-test:** Explain generators vs lists (laziness, memory), and write a context manager two ways (`__enter__/__exit__` and `@contextmanager`).

---

## Week 10 — NumPy mastery (vectorization, broadcasting, einsum)

**Goal:** Vectorization fluency. This single skill most determines how fast you'll move once training loops appear.

**Read**
- VanderPlas, *Python Data Science Handbook* — **Ch. 2** in full (data types, ufuncs, aggregations, **broadcasting**, masks/boolean logic, fancy indexing, sorting, structured arrays). Free online.
- *SciPy Lecture Notes* **§1.3 NumPy** (scipy-lectures.org)
- An `einsum` primer (ajcr's "einsum is all you need" blog or Rocktäschel's note)

**Code**
- **`rougier/numpy-100`** (GitHub) — do at least the first ~60 exercises.
- Reimplement **linear regression** (normal equations *and* gradient descent) using only NumPy, expressing the matrix ops with `einsum`.
- Take a loop-heavy bit of your own bio code and vectorize it; benchmark the speedup with `%timeit`.

**Deliverable:** "Vectorization wins" notebook: the einsum linear-regression + one real before/after vectorization with timing.

**Self-test:** Predict the broadcast result and shape of a non-trivial operation before running it; explain what `np.einsum('ij,jk->ik', A, B)` does.

---

## Week 11 — Pandas mastery + data wrangling

**Goal:** Turn messy real-world tables into clean, analysis-ready data — fluently.

**Read**
- McKinney, *Python for Data Analysis* (3rd ed., free at **wesmckinney.com/book**):
  - **Ch. 5** Getting Started with pandas
  - **Ch. 6** Data Loading, Storage, and File Formats
  - **Ch. 7** Data Cleaning and Preparation
  - **Ch. 8** Data Wrangling: Join, Combine, Reshape
  - **Ch. 9** Plotting and Visualization
  - **Ch. 10** Data Aggregation and Group Operations
  - **Ch. 11** Time Series (skim unless relevant)

**Code**
- Full pipeline on a **messy real bio CSV**: load → diagnose issues → clean (types, missing values, duplicates) → reshape (melt/pivot) → group/aggregate → plot.
- Practice `groupby`, `merge`, `pivot_table`, and `apply` vs vectorized alternatives.

**Deliverable:** End-to-end wrangling notebook on one of your own datasets, going from raw file to a publication-style summary figure.

**Self-test:** Explain `groupby`'s split-apply-combine, and when `merge` (join keys) vs `concat` (stacking) is right.

---

## Week 12 — Software engineering for ML

**Goal:** Stop living in single notebooks. Environments, packaging, testing, experiment tracking — the scaffolding that makes Block C (deep learning) sane.

**Read / follow**
- **Hypermodern Python** tutorial (Claudio Jolowicz) — the canonical modern-tooling walkthrough
- Slatkin, *Effective Python* (2nd ed.) — selected items on functions, comprehensions, classes, and robustness
- **calmcode.io** short courses — `pytest`, `Hydra`, `wandb`, `rich`, `logging`

**Set up properly (and keep)**
- Environments: `uv` or your existing `conda`
- Editor: VS Code + Python, Pylance, **Ruff**, Jupyter
- **Ruff** for lint + format (replaces black/flake8/isort)
- **pytest** for tests
- **pre-commit** hooks
- **Weights & Biases (wandb)** for experiment tracking — you'll lean on this from Block C onward
- Git: feature branches, and open PRs to yourself to practice the workflow

**Code**
- Convert a Jupyter notebook into a **proper installable package**: `pyproject.toml`, `src/` layout, type hints, a handful of `pytest` tests, a `README`, pre-commit config, and a logged wandb run.

**Deliverable:** A small but real Python package on your machine (and ideally GitHub) — installable, tested, linted, with one tracked experiment.

**Self-test:** From an empty folder, scaffold a new project with env + ruff + pytest + pyproject and run a passing test, without copying from notes.

---

### ▣ Foundations block exit criteria

You're ready for Block B (Classical ML) when you can, from a blank page:
1. Explain backprop as the chain rule and derive a 2-layer MLP's gradients.
2. Reconstruct PCA via SVD and implement it in NumPy.
3. State which MLE each common loss corresponds to.
4. Write a stable softmax + cross-entropy from memory.
5. Vectorize a loop with broadcasting/einsum and reason about shapes.
6. Scaffold a tested, linted Python package from scratch.

---

## Master resource list (all free unless noted)

**Math**
- *Mathematics for Machine Learning* — mml-book.com (spine)
- 3Blue1Brown — *Essence of Linear Algebra*, *Essence of Calculus*, *Neural Networks* (YouTube)
- Gilbert Strang — MIT 18.06 (OCW / YouTube)
- Harvard Stat 110, Joe Blitzstein (YouTube + projects.iq.harvard.edu/stat110)
- StatQuest with Josh Starmer (YouTube)
- *The Matrix Calculus You Need For Deep Learning* — explained.ai/matrix-calculus
- Bishop, *Pattern Recognition and Machine Learning* (PRML) — free PDF
- MacKay, *Information Theory, Inference, and Learning Algorithms* — free PDF
- Goodfellow et al., *Deep Learning* — deeplearningbook.org
- *(optional)* Boyd & Vandenberghe, *Convex Optimization* + EE364a

**Python**
- Ramalho, *Fluent Python* 2nd ed. (book, paid)
- VanderPlas, *Python Data Science Handbook* — free online
- McKinney, *Python for Data Analysis* 3rd ed. — wesmckinney.com/book
- `rougier/numpy-100` (GitHub)
- SciPy Lecture Notes — scipy-lectures.org
- Slatkin, *Effective Python* 2nd ed. (book, paid)
- Hypermodern Python (Claudio Jolowicz, blog series)
- calmcode.io

---

*Pacing reminder: Weeks 3, 5, and 8 are the steepest math hills (SVD, matrix calculus, information theory). Week 9 (Fluent Python) feels slow but has the highest long-term ROI. Re-do Week 10 if NumPy doesn't click — everything after depends on it.*

---
---

# PROGRAM CONTINUES — Blocks B, C, D (Weeks 13–42)

> **Note on numbering:** Expanding the foundations block to 12 weeks shifted the full program from ~36 to ~42 weeks (~10.5 months at ~10 h/week). The blocks below pick up where foundations ends. Same rules apply throughout: derive/understand before you code, build alongside every lecture (target ~1:2 watching:coding), and repeat any week that hasn't clicked.

**Framework commitment from here on:** **PyTorch.** Andrew Ng's Deep Learning courses use a mix and some Keras; that's fine for concepts, but re-implement everything in PyTorch. Don't dilute energy across TensorFlow/JAX this year.

---
---

# PART III — CLASSICAL MACHINE LEARNING (Weeks 13–20)

Don't skip this for deep learning. Trees, regression, and SVMs remain the *right* tool for most tabular and small-data problems — which describes a lot of biology. This block also drills the discipline (validation, leakage, evaluation) that deep learning makes easy to get wrong.

**Block resources**
- **Andrew Ng — Machine Learning Specialization** (Coursera, 2022, Python): Course 1 *Supervised Learning: Regression & Classification*, Course 2 *Advanced Learning Algorithms*, Course 3 *Unsupervised Learning, Recommenders, RL*
- **Géron — *Hands-On Machine Learning*** (3rd ed.), Part I, chapters 1–9 (the best practical ML book)
- **ISLP — *An Introduction to Statistical Learning with Python*** — free at statlearning.com (the theory companion)
- **StatQuest** for per-algorithm intuition
- **Kaggle Learn** micro-courses + competitions as the practice gym

---

## Week 13 — Linear regression, properly

**Goal:** Re-derive and re-implement what you saw in Week 7, now as a full ML workflow with feature scaling and diagnostics.

**Read / watch**
- Ng ML Spec — **Course 1, Weeks 1–2** (model, cost function, gradient descent, multiple features, feature scaling, polynomial regression)
- Géron — **Ch. 4** through "Polynomial Regression" (Linear Regression, Normal Equation, Gradient Descent variants)
- ISLP — **Ch. 3** (Simple & Multiple Linear Regression, model fit, potential problems)

**Build**
- Linear regression from scratch in NumPy via **both** the normal equation and gradient descent; confirm they agree.
- Plot the cost surface and the GD path on it; show the effect of learning rate and feature scaling.

**Deliverable:** "Linear regression, two ways" notebook with cost-surface visualization and a feature-scaling ablation.

**Self-test:** When does the normal equation beat GD and vice versa? What breaks without feature scaling?

---

## Week 14 — Logistic regression & classification metrics

**Goal:** Classification, decision boundaries, and the metrics that matter (most real problems are imbalanced — accuracy lies).

**Read / watch**
- Ng ML Spec — **Course 1, Week 3** (logistic regression, decision boundary, cost, overfitting, regularization)
- Géron — **Ch. 3** (Classification: MNIST, confusion matrix, precision/recall, ROC, multiclass) + **Ch. 4** logistic/softmax regression sections
- ISLP — **Ch. 4** (Logistic Regression, LDA/QDA, Naive Bayes, comparison)
- StatQuest — ROC and AUC, Confusion Matrix

**Build**
- Logistic regression from scratch (sigmoid + cross-entropy + GD); plot decision boundaries on a 2D toy set.
- Compute precision, recall, F1, ROC-AUC by hand from a confusion matrix, then verify with sklearn.

**Deliverable:** Classifier notebook on an imbalanced dataset reporting the full metric suite, with a short note on why accuracy alone would mislead here.

**Self-test:** Explain precision vs recall with a real consequence (e.g., a peptide-toxicity screen), and read an ROC curve aloud.

---

## Week 15 — Regularization, bias–variance, cross-validation

**Goal:** The single most important *discipline* week. Understand overfitting structurally and validate honestly.

**Read / watch**
- Ng ML Spec — **Course 2, Week 3** (bias/variance, ML diagnostics, error analysis, learning curves)
- Géron — **Ch. 4** regularized models (Ridge, Lasso, Elastic Net, Early Stopping)
- ISLP — **Ch. 5** (Cross-Validation, Bootstrap) + **Ch. 6** (Ridge, Lasso, dimension reduction, selection)
- StatQuest — Bias/Variance, Ridge, Lasso, Cross-Validation

**Build**
- Implement k-fold cross-validation from scratch; compare to `sklearn.model_selection`.
- Ridge vs Lasso on the same data; plot coefficient paths vs the regularization strength; show Lasso's sparsity.
- Reproduce a bias–variance tradeoff curve (train vs validation error vs model complexity).

**Deliverable:** "Validation discipline" notebook — a reusable CV pipeline template plus the bias–variance and coefficient-path plots. **Keep this template; you'll reuse it all year.**

**Self-test:** Define data leakage and name three ways it sneaks in. Explain why Lasso zeros coefficients but Ridge doesn't.

---

## Week 16 — Support vector machines & kernels

**Goal:** Margins and the kernel trick — still excellent for small, high-dimensional datasets (common in biology).

**Read / watch**
- Géron — **Ch. 5** (Linear SVM classification, soft margin, nonlinear SVM, kernels, SVM regression)
- ISLP — **Ch. 9** (Maximal Margin Classifier, SVC, SVM, kernels)
- StatQuest — SVM main ideas, Polynomial & RBF kernels

**Understand**
- Why maximizing the margin is a convex optimization (ties back to Week 5/7).
- The kernel trick: computing inner products in a high-dimensional space without ever going there.

**Build**
- Train SVMs with linear, polynomial, and RBF kernels; visualize decision boundaries and the effect of `C` and `gamma`.
- Apply to a high-dimensional bio feature set (e.g., ESM embeddings) and compare to logistic regression.

**Deliverable:** Kernel-comparison notebook with boundary plots and a hyperparameter (`C`, `gamma`) sweep.

**Self-test:** Explain the kernel trick in one paragraph and what `C` trades off.

---

## Week 17 — Decision trees, bagging & random forests

**Goal:** Nonlinear, interpretable models and the variance-reduction idea behind ensembles.

**Read / watch**
- Ng ML Spec — **Course 2, Week 4** (decision trees, tree ensembles)
- Géron — **Ch. 6** (Decision Trees) + **Ch. 7** through Random Forests (Voting, Bagging/Pasting, Random Forests, feature importance)
- ISLP — **Ch. 8** (Trees, Bagging, Random Forests)
- StatQuest — Decision Trees, Random Forests

**Build**
- A decision tree from scratch on a small dataset (recursive splitting by Gini/entropy) — even a shallow one builds deep intuition.
- Random forest via sklearn; extract and interpret feature importances on a real dataset.

**Deliverable:** From-scratch tree + a random-forest analysis with a feature-importance plot you can explain biologically.

**Self-test:** Why does a single deep tree overfit, and exactly how does bagging fix it?

---

## Week 18 — Gradient boosting & XGBoost (+ first serious Kaggle)

**Goal:** The workhorse of tabular ML. For many bio tabular tasks, a tuned gradient-boosted model is the model to beat.

**Read / watch**
- Géron — **Ch. 7** boosting sections (AdaBoost, Gradient Boosting, stacking)
- StatQuest — Gradient Boost (Parts 1–4), XGBoost (Parts 1–4)
- XGBoost docs — "Introduction to Boosted Trees"

**Build**
- Gradient boosting by hand on a tiny dataset (fit residuals stage by stage) to *see* the mechanism.
- XGBoost / LightGBM on a Kaggle tabular competition (**Titanic** or **House Prices** to learn the pipeline); submit and read the leaderboard.

**Deliverable:** A submitted Kaggle entry + a notebook documenting your validation strategy, feature engineering, and hyperparameter tuning.

**Self-test:** Explain how boosting differs from bagging (sequential residual-fitting vs parallel averaging) and what regularizes XGBoost.

---

## Week 19 — Unsupervised learning: clustering, GMMs, dimensionality reduction

**Goal:** Find structure without labels — directly useful for exploring embeddings and high-dimensional bio data.

**Read / watch**
- Ng ML Spec — **Course 3, Weeks 1–2** (k-means, anomaly detection / Gaussian models, recommenders)
- Géron — **Ch. 8** (PCA, kernel PCA, LLE) + **Ch. 9** (k-means, DBSCAN, Gaussian Mixture Models)
- ISLP — **Ch. 12** (PCA, k-means, hierarchical clustering)
- StatQuest — k-means, Hierarchical clustering, t-SNE, UMAP

**Build**
- k-means from scratch (assignment/update loop); compare to sklearn.
- Cluster a real embedding set (e.g., ESM-2 protein embeddings or BGC features); visualize with **PCA → UMAP/t-SNE**; sanity-check clusters against known labels.
- Fit a GMM; contrast soft (GMM) vs hard (k-means) assignments.

**Deliverable:** "Structure in embeddings" notebook: clustering + 2D projection of a biological dataset with an interpretation of what the clusters mean.

**Self-test:** When is GMM preferable to k-means? Why is t-SNE/UMAP great for visualization but unsafe to feed downstream as features?

---

## Week 20 — Pipelines, leakage, feature engineering + **Block project**

**Goal:** Tie the block together into one clean, leak-free, end-to-end project.

**Read / watch**
- Géron — **Ch. 2** (End-to-End ML Project) — deep re-read, this is the template
- scikit-learn docs — `Pipeline`, `ColumnTransformer`, `cross_val_score`, `GridSearchCV`
- Kaggle Learn — *Intermediate ML*, *Feature Engineering*

**Build (project week)**
- A full sklearn `Pipeline`: preprocessing → feature engineering → model → tuned via cross-validated grid/random search, with **zero leakage** (all transforms fit inside CV folds).

**Deliverable — Block B capstone:** A complete, documented project on a **biological dataset of your choice** (peptide property prediction, BGC classification, condensate-propensity from sequence features…), going raw data → clean pipeline → tuned model → honest held-out evaluation → written interpretation. This is portfolio-worthy.

**Self-test:** Hand someone your notebook — could they reproduce your result and trust there's no leakage? If not, it's not done.

> **▣ Block B exit criteria:** You can choose the right classical model for a tabular problem, build a leak-free cross-validated pipeline, evaluate with the correct metric, and explain *why* the model works.

---
---

# PART IV — DEEP LEARNING FOUNDATIONS (Weeks 21–30)

Now you *build* neural networks — first from scratch (so nothing is magic), then fluently in PyTorch. The combination of Karpathy (from-scratch intuition) and Ng (structured breadth) is deliberate; do both.

**Block resources**
- **Andrej Karpathy — *Neural Networks: Zero to Hero*** (YouTube) — the essential from-scratch series
- **Andrew Ng — Deep Learning Specialization** (Coursera) — Courses 1–5
- **PyTorch official tutorials** — *Learn the Basics* + *Deep Learning with PyTorch: A 60 Minute Blitz*
- **Dive into Deep Learning** — d2l.ai (free, PyTorch code) as a reference
- **Stanford CS231n** notes (cs231n.github.io) for CNN depth
- **Weights & Biases (wandb)** — start tracking every run from this block on

---

## Week 21 — Build autograd from scratch + PyTorch basics

**Goal:** Understand backprop by *implementing* it, then meet PyTorch's tensor/autograd model.

**Watch / read**
- Karpathy ZTH — **Video 1: "The spelled-out intro to neural networks and backpropagation: building micrograd"** (follow along, don't just watch)
- PyTorch — *Learn the Basics* (Tensors, Autograd, Build the Neural Network) + the *60 Minute Blitz*

**Build**
- Type out **micrograd** yourself: a scalar-valued autograd engine + a tiny MLP that learns. Re-derive each `_backward` from the chain rule (Week 4 pays off here).
- Re-create the same tiny MLP in PyTorch using `autograd`; confirm gradients match your engine.

**Deliverable:** Your own micrograd implementation + a PyTorch port, both training a toy classifier to the same result.

**Self-test:** Explain what a computational graph is and how reverse-mode autodiff walks it. Why is `.backward()` "just" your `_backward` automated?

---

## Week 22 — makemore: bigram → MLP language model

**Goal:** Your first real (small) models, and the count-based vs neural framing.

**Watch / read**
- Karpathy ZTH — **Video 2 (bigram makemore)** and **Video 3 (MLP makemore)**
- Ng DL Spec — **Course 1, Weeks 1–2** (intro to deep learning; logistic regression as a neural net; Python/NumPy vectorization)

**Build**
- Bigram character model two ways: explicit counts, then a single-layer neural net trained by GD — show they converge to the same thing.
- The MLP character model (Bengio-style); generate samples from your own dataset (names, or short peptide sequences for fun).

**Deliverable:** A character-level generator trained on a dataset you chose, with sampled outputs.

**Self-test:** Explain why the neural bigram model and the count model agree, and what the embedding layer is doing.

---

## Week 23 — Deep networks + the PyTorch training loop

**Goal:** Generalize to deep nets and own the canonical train/eval loop you'll write hundreds of times.

**Watch / read**
- Ng DL Spec — **Course 1, Weeks 3–4** (shallow then deep neural networks; forward/backprop for L layers)
- d2l.ai — chapters on MLPs and the training loop (reference)

**Build**
- An MLP for **MNIST** in PyTorch with a proper loop: `DataLoader`, forward, loss, `backward`, `optimizer.step`, train/val split, accuracy tracking.
- Log the run to **wandb** (loss/accuracy curves).

**Deliverable:** MNIST MLP hitting solid accuracy, with wandb-logged curves and a clean, reusable training-loop function.

**Self-test:** Write the six lines of the core training step from memory and explain `optimizer.zero_grad()`.

---

## Week 24 — Activations, initialization, BatchNorm

**Goal:** Why deep nets fail to train (vanishing/exploding activations & gradients) and the fixes.

**Watch / read**
- Karpathy ZTH — **Video 4: "Activations & Gradients, BatchNorm"** (the diagnostics here are gold)
- Ng DL Spec — **Course 2, Week 1** (regularization, dropout, initialization, gradient checking)
- CS231n notes — *Neural Networks Part 2* (init, batchnorm)

**Build**
- Reproduce Karpathy's activation/gradient histograms; show a net failing with bad init, then fixed with proper init + BatchNorm.
- Add dropout and L2; run a small regularization ablation on your MNIST net.

**Deliverable:** "Why my net wouldn't train" notebook — activation/gradient diagnostics before/after fixes.

**Self-test:** Explain vanishing gradients and how (a) initialization and (b) BatchNorm each help.

---

## Week 25 — Optimization, tuning + backprop ninja

**Goal:** Modern optimizers in practice, and one last from-scratch backprop drill to cement mastery.

**Watch / read**
- Ng DL Spec — **Course 2, Weeks 2–3** (mini-batch GD, momentum, RMSProp, Adam, learning-rate decay; hyperparameter tuning, batchnorm, frameworks)
- Karpathy ZTH — **Video 5: "Becoming a Backprop Ninja"** (manually backprop a whole MLP, no autograd)

**Build**
- Manually backprop every operation of an MLP (no `autograd`); verify against PyTorch gradients to the last decimal.
- Compare optimizers and LR schedules on the same model; plot convergence.

**Deliverable:** Backprop-ninja notebook passing the gradient check + an optimizer/LR-schedule comparison.

**Self-test:** Do it from a blank page — manually compute the gradient through one linear+activation+loss stack.

---

## Week 26 — Structuring ML projects + experiment tracking

**Goal:** Think like a practitioner — diagnose, prioritize, and track experiments systematically.

**Watch / read**
- Ng DL Spec — **Course 3** (ML strategy: train/dev/test splits, distribution mismatch, error analysis, when to change strategy)
- calmcode.io / wandb docs — experiment tracking, sweeps

**Build**
- Set up a **wandb sweep** to tune your MNIST/CIFAR model; do an error analysis on the worst-misclassified examples.

**Deliverable:** A documented hyperparameter sweep + a one-page error-analysis writeup that proposes the next experiment.

**Self-test:** Given a model that does well on train but poorly on dev, list the candidate causes in priority order.

---

## Week 27 — Convolutional networks, part 1

**Goal:** The core vision architecture — convolutions, pooling, and why they suit images.

**Watch / read**
- Ng DL Spec — **Course 4, Weeks 1–2** (CNN foundations; classic architectures: LeNet, AlexNet, VGG, ResNet, Inception)
- CS231n notes — *Convolutional Networks* (the canonical written reference)

**Build**
- A CNN for **CIFAR-10** in PyTorch; aim to beat ~80% test accuracy. Track with wandb.

**Deliverable:** CIFAR-10 CNN with logged curves and a short architecture rationale.

**Self-test:** Compute the output shape and parameter count of a conv layer by hand. Why do CNNs need far fewer parameters than an equivalent MLP?

---

## Week 28 — CNNs, part 2: architectures & transfer learning

**Goal:** Use pretrained backbones — the realistic way most vision (and many bio-imaging) tasks get solved.

**Watch / read**
- Ng DL Spec — **Course 4, Weeks 3–4** (detection, face recognition, neural style transfer — for breadth)
- CS231n notes — *Transfer Learning*
- PyTorch — *Transfer Learning* tutorial

**Build**
- Fine-tune a pretrained ResNet on a small custom dataset (a microscopy/cell-image set if you have one, else a standard small set); compare training from scratch vs fine-tuning.

**Deliverable:** Transfer-learning notebook quantifying the data-efficiency gain from a pretrained backbone.

**Self-test:** When do you freeze the backbone vs fine-tune end to end? Why does transfer learning need so much less data?

---

## Week 29 — Sequence models: RNN, LSTM, GRU

**Goal:** Recurrent architectures and why they struggle with long contexts — the motivation for attention next block.

**Watch / read**
- Ng DL Spec — **Course 5, Weeks 1–2** (RNNs, GRU, LSTM; word embeddings)
- d2l.ai — RNN/LSTM/GRU chapters (reference)
- Karpathy — *The Unreasonable Effectiveness of RNNs* (blog)

**Build**
- A char-level RNN/LSTM in PyTorch trained on **protein sequences** (or text); generate samples and inspect what it learned.

**Deliverable:** Char-level sequence model on biological sequences, with generated output and a note on its failure modes (long-range dependencies).

**Self-test:** Explain the vanishing-gradient problem in RNNs and how LSTM gates mitigate it.

---

## Week 30 — WaveNet / hierarchical models + **Block project**

**Goal:** A deeper sequence architecture, then consolidate the block by reproducing a real result.

**Watch / read**
- Karpathy ZTH — **Video 6: "Building a WaveNet"** (hierarchical, dilated structure)

**Build (project week)**
- Build the WaveNet-style model; train on your sequence dataset.
- **Replicate one figure** from a small, accessible deep-learning paper end to end (pick something with public data and a clear metric).

**Deliverable — Block C capstone:** A reproduced paper figure + your WaveNet model, both in a clean repo with a README and wandb runs.

**Self-test:** You can take an architecture description from a paper and implement it in PyTorch without a tutorial.

> **▣ Block C exit criteria:** You can build, train, and *debug* a neural net from scratch in PyTorch, reason about why training fails, and reproduce a published result.

---
---

# PART V — SPECIALIZATION: TRANSFORMERS + BIO ML (Weeks 31–42)

The frontier block. You'll build a GPT from scratch, learn the Hugging Face ecosystem, then turn toward protein/structure models — the part that makes you distinctively valuable rather than a generic ML engineer. From here, courses give way to papers + code.

**Block resources**
- **Karpathy** — *Let's build GPT* (ZTH video 7), *Let's build the GPT Tokenizer*, *Let's reproduce GPT-2 (nanoGPT)*
- **Stanford CS224n** — *NLP with Deep Learning* (lectures on YouTube, notes online)
- **Hugging Face LLM/NLP Course** — free, hands-on with `transformers`, `datasets`, `tokenizers`, PEFT
- **Jay Alammar** — *The Illustrated Transformer*; **Harvard NLP** — *The Annotated Transformer*
- **Lilian Weng (Lil'Log)** — attention, transformer family, diffusion deep-dives
- **Bio:** ESM-2 (Lin et al., *Science* 2023) + `facebookresearch/esm`; AlphaFold2 (Jumper et al., *Nature* 2021); ProteinMPNN (Dauparas et al., *Science* 2022); RFdiffusion (Watson et al., *Nature* 2023)
- **Hugging Face Diffusion course** for generative modeling

---

## Week 31 — Attention from first principles + build GPT

**Goal:** *The* architecture of modern AI. Build a working transformer so attention is concrete, not a diagram.

**Watch / read**
- Karpathy ZTH — **Video 7: "Let's build GPT: from scratch, in code, spelled out"** (build it line by line)
- Jay Alammar — *The Illustrated Transformer* (read first for the mental model)
- Harvard NLP — *The Annotated Transformer* (the paper as runnable code)

**Build**
- A decoder-only transformer (nanoGPT-scale) from scratch in PyTorch: token + positional embeddings, multi-head self-attention, feed-forward, residuals + layernorm. Train it on a small corpus.

**Deliverable:** Your own working GPT generating coherent samples, with an annotated diagram mapping each code block to the architecture.

**Self-test:** Derive scaled dot-product attention and explain Q/K/V, multi-head, and why the `√d` scaling. Why is the causal mask there?

---

## Week 32 — Tokenizers

**Goal:** The unglamorous but crucial input layer — how text/sequence becomes tokens.

**Watch / read**
- Karpathy — *Let's build the GPT Tokenizer*
- HF — *The Tokenizers library* (course Ch. 6)

**Build**
- A **byte-pair encoding (BPE)** tokenizer from scratch; train it on a corpus; compare vocab/sequence-length tradeoffs. (Try tokenizing protein sequences vs natural language and observe the difference.)

**Deliverable:** Working BPE tokenizer + a short analysis of vocabulary size vs sequence length.

**Self-test:** Walk through BPE merge-by-merge on a short string. Why do tokenization choices affect model quality and cost?

---

## Week 33 — CS224n foundations: word vectors

**Goal:** The conceptual roots of representation learning in NLP.

**Watch / read**
- CS224n — lectures on **Word Vectors / word2vec & GloVe** and **Neural Classifiers** (first ~2–3 lectures); matching lecture notes
- Lil'Log — *Learning Word Embedding* (optional)

**Build**
- Implement **word2vec (skip-gram with negative sampling)** on a small corpus; inspect nearest neighbors and analogies in the embedding space.

**Deliverable:** word2vec implementation + an embedding-space exploration (nearest neighbors, simple analogies).

**Self-test:** Explain the skip-gram objective and what negative sampling approximates.

---

## Week 34 — CS224n: dependency parsing, RNNs, attention in context

**Goal:** Fill in the NLP backbone between embeddings and transformers.

**Watch / read**
- CS224n — lectures on **Dependency Parsing**, **RNNs/Language Models**, **LSTMs & vanishing gradients**, **Machine Translation, Seq2Seq, and Attention**; matching notes

**Build**
- A small seq2seq model with attention (translation or a toy mapping); visualize the attention weights as a heatmap.

**Deliverable:** Seq2seq-with-attention notebook including an attention-weight visualization.

**Self-test:** Explain why attention was introduced for translation (the seq2seq bottleneck) and what the attention weights represent.

---

## Week 35 — CS224n: transformers, pretraining, large models

**Goal:** Connect your from-scratch GPT to the research framing of pretraining and scaling.

**Watch / read**
- CS224n — lectures on **Self-Attention & Transformers**, **Pretraining**, and **large language models / prompting & RLHF**; matching notes
- Lil'Log — *The Transformer Family* (variants overview)

**Build**
- Annotate a full transformer diagram (encoder vs decoder vs encoder-decoder); write a one-pager comparing BERT (encoder), GPT (decoder), and T5 (encoder-decoder) objectives.

**Deliverable:** A reference one-pager on transformer variants and pretraining objectives that you actually understand.

**Self-test:** Contrast masked LM vs causal LM vs seq2seq pretraining and which model family uses which.

---

## Week 36 — The Hugging Face ecosystem

**Goal:** Become productive with the tools the whole field uses — `transformers`, `datasets`, `tokenizers`, `Trainer`.

**Watch / read**
- HF Course — **Chapters 1–4** (transformer models; using transformers; fine-tuning with `Trainer`; sharing models/tokenizers)

**Build**
- Load a pretrained **BERT/DistilBERT**, fine-tune it on a text (or sequence) classification task with the `Trainer` API; push the result to the Hub.

**Deliverable:** A fine-tuned classifier on the HF Hub with a model card describing data, metrics, and limitations.

**Self-test:** Explain the difference between a pipeline, a model, and a tokenizer in HF, and what `AutoModel` vs `AutoModelForSequenceClassification` give you.

---

## Week 37 — Fine-tuning at depth: PEFT / LoRA

**Goal:** Adapt large models cheaply — the practical reality of working with LLMs on limited hardware.

**Watch / read**
- HF Course — **Chapters 5–7** (the `datasets` library; the `tokenizers` library; main NLP tasks)
- HF **PEFT** docs + the **LoRA** paper (Hu et al., 2021) and **QLoRA** (Dettmers et al., 2023)

**Build**
- **LoRA-fine-tune** a small open LLM on a domain dataset; compare parameter count and memory vs full fine-tuning; evaluate the adapted model.

**Deliverable:** A LoRA fine-tune with a before/after evaluation and a note on the compute/memory savings.

**Self-test:** Explain what LoRA freezes and what it trains, and why that's so much cheaper. What does QLoRA add?

---

## Week 38 — Reproduce GPT-2 (nanoGPT)

**Goal:** Scale your from-scratch transformer toward a real, known model — the capstone of the "build it yourself" thread.

**Watch / read**
- Karpathy — *Let's reproduce GPT-2 (124M)* + the **nanoGPT** repo (read the code closely)

**Build**
- Train a small GPT (nanoGPT-scale) on a dataset within your compute budget; benchmark against a known baseline; profile training (tokens/sec, memory).

**Deliverable:** A trained small-GPT repo with training curves, samples, and notes on the engineering (data loading, mixed precision, checkpointing).

**Self-test:** You can read nanoGPT and explain every component, and you know which knobs matter most for training stability and speed.

---

## Week 39 — Protein language models (ESM-2)

**Goal:** Bring transformers into your domain — protein LMs you can actually use in your research.

**Watch / read**
- **ESM-2 / ESMFold** paper (Lin et al., *Science* 2023) + `facebookresearch/esm` README and example notebooks
- Mohammed AlQuraishi's blog posts on protein LMs (for framing)

**Build**
- Run **ESM-2** inference: extract per-residue and per-protein **embeddings**; try a downstream probe (e.g., predict a property, or contact/structure features) on your own sequences.

**Deliverable:** An ESM-2 embedding pipeline + a small downstream task (probe classifier/regressor) on sequences relevant to your work (peptides, condensate-formers).

**Self-test:** Explain what a protein language model learns from sequence alone and why its embeddings are useful features for downstream tasks.

---

## Week 40 — Diffusion model fundamentals

**Goal:** The generative paradigm behind modern structure/design tools (and image generation).

**Watch / read**
- Lil'Log — *What are Diffusion Models?* (the canonical written deep-dive)
- HF **Diffusion Course** — Units 1–2 (the diffusion process; training a diffusion model)

**Build**
- A **toy diffusion model** on simple data (2D shapes or small images): implement the forward noising and the learned reverse denoising; sample from it.

**Deliverable:** A minimal working diffusion model with a sampling animation/notebook.

**Self-test:** Explain the forward and reverse processes and what the network actually predicts (noise vs `x₀`).

---

## Week 41 — Structure & design models: AlphaFold2, ProteinMPNN, RFdiffusion

**Goal:** Understand — at the level of reading the code — the models you already use, so they stop being black boxes.

**Watch / read**
- **AlphaFold2** (Jumper et al., *Nature* 2021) + AlQuraishi's unpacking posts
- **ProteinMPNN** (Dauparas et al., *Science* 2022) — paper + Baker-lab repo
- **RFdiffusion** (Watson et al., *Nature* 2023) — paper + repo (you already use this — now read it)

**Build**
- An **annotated code walkthrough** of ProteinMPNN or RFdiffusion: map the paper's method to the repo's modules; run an example end to end.

**Deliverable:** A written + code-annotated walkthrough connecting one model's paper to its implementation, with a worked example.

**Self-test:** Explain, in your own words, how RFdiffusion generates backbones and where ProteinMPNN fits in the design pipeline.

---

## Week 42 — **Capstone**

**Goal:** Produce one original artifact that uses what you've built — ideally something useful to your actual research.

**Build (capstone)**
- An original mini-project. Options scaled to your interests/compute:
  - A property predictor for **PeptArm** (e.g., condensate/LLPS propensity or solubility) using ESM-2 embeddings + a trained head.
  - A fine-tuned sequence model for a specific bio task.
  - A reproduction-and-extension of a recent bio-ML paper.

**Deliverable — Program capstone:** A clean, documented repo (README, tests, tracked experiments, results, honest limitations) that you could show a PI or put in a portfolio. Write a short post explaining what you built and what you learned.

**Self-test:** You can scope, build, evaluate, and document an ML project end to end, and explain every modeling decision.

> **▣ Block D exit criteria:** You can read a recent ML or bio-ML paper, implement its core idea in PyTorch, apply it to a domain problem, and communicate the result.

---
---

# AFTER WEEK 42 — CONTINUOUS MODE

No more weekly schedule. Replace it with three durable habits:

- **One paper/week**, read deeply. Sources: Lil'Log, The Annotated Transformer, Papers with Code (paperswithcode.com), arxiv-sanity, and the bio-ML community (Mohammed AlQuraishi, Sergey Ovchinnikov, Pranam Chatterjee; the OpenBioML Discord).
- **One project/quarter** — reproduce a paper, contribute to an open-source repo (Karpathy's, lucidrains', HF `transformers`), or build a tool you'll actually use.
- **One Kaggle or domain challenge/year** to keep pipeline and validation skills sharp.

The bio/protein-ML track (Weeks 39–41) never really ends — it's where you'll keep going deepest, and where your existing lab experience compounds fastest.

---

# FULL PROGRAM TIMELINE AT A GLANCE

| Block | Weeks | Theme | Capstone |
|-------|-------|-------|----------|
| I — Foundations | 1–12 | Math (1–8) + Python (9–12) | Regression from scratch in NumPy |
| II — Classical ML | 13–20 | Regression → trees → ensembles → unsupervised | Leak-free bio tabular project |
| III — Deep Learning | 21–30 | Autograd → PyTorch → CNNs → RNNs | Reproduce a DL paper figure |
| IV — Specialization | 31–42 | Transformers → HF → protein/structure models | Original bio-ML project |
| Continuous | 43+ | Papers, projects, contributions | — |

**~42 weeks ≈ 10.5 months at ~10 h/week.** Slower is fine — depth beats speed. The two hardest cliffs are **Weeks 3/5/8** (SVD, matrix calculus, information theory) and **Weeks 21–24** (autograd → PyTorch → why-won't-it-train). Plan buffer there.

---

# EXPANDED MASTER RESOURCE LIST (Blocks II–IV)

**Classical ML**
- Andrew Ng, *Machine Learning Specialization* (Coursera, 2022)
- Géron, *Hands-On Machine Learning* 3rd ed. (book)
- *An Introduction to Statistical Learning with Python* (ISLP) — statlearning.com (free)
- StatQuest (YouTube); Kaggle Learn + Competitions

**Deep Learning**
- Karpathy, *Neural Networks: Zero to Hero* (YouTube)
- Andrew Ng, *Deep Learning Specialization* (Coursera)
- PyTorch tutorials — pytorch.org/tutorials
- *Dive into Deep Learning* — d2l.ai (free)
- Stanford CS231n — cs231n.github.io
- Weights & Biases — wandb.ai

**Transformers & NLP**
- Karpathy — *Let's build GPT*, *GPT Tokenizer*, *Reproduce GPT-2* + nanoGPT repo
- Stanford CS224n — web.stanford.edu/class/cs224n
- Hugging Face Course — huggingface.co/learn
- Jay Alammar — *The Illustrated Transformer*
- Harvard NLP — *The Annotated Transformer*
- Lilian Weng — lilianweng.github.io

**Bio / Protein ML**
- ESM — github.com/facebookresearch/esm (+ Lin et al., *Science* 2023)
- AlphaFold2 — Jumper et al., *Nature* 2021
- ProteinMPNN — Dauparas et al., *Science* 2022
- RFdiffusion — Watson et al., *Nature* 2023
- Hugging Face Diffusion Course — huggingface.co/learn/diffusion-course
- Papers with Code — paperswithcode.com; OpenBioML Discord

---

*End of curriculum. Build something every single week — the people who finish this path are the ones who wrote code alongside every lecture, not the ones who watched the most videos.*
