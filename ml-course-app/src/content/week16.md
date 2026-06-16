# Week 17: Decision Trees, Bagging, & Random Forests

> **Goal:** Explore non-linear, tree-based models. You will understand how decision trees make recursive partitions using Gini impurity and entropy, learn the mathematical reason why bagging reduces model variance, and implement Random Forests to leverage tree ensembles and extract feature importances.

---

## Part 1: Decision Trees and Splitting Criteria

A decision tree splits the feature space into hyper-rectangles by making sequential, axis-aligned cuts. 

```
          [ Feature X1 < 3.5 ]
               /        \
             Yes         No
             /            \
    [ Feature X2 < 1.0 ]   [ Class 1 ]
       /         \
   [ Class 0 ]  [ Class 1 ]
```

To decide where to make a split, we want to find the feature and threshold that result in the "purest" child nodes.

### 1.1 Entropy and Information Gain
**Entropy** measures the impurity or disorder in a set of samples $S$:

$$ H(S) = -\sum_{c=1}^C p_c \log_2 p_c $$

where $p_c$ is the proportion of samples in $S$ belonging to class $c$. If all samples belong to one class, $H(S) = 0$. If classes are evenly split, $H(S) = 1.0$.

**Information Gain** measures the reduction in entropy after splitting $S$ into subsets $S_{\text{left}}$ and $S_{\text{right}}$ based on a feature split $T$:

$$ \text{IG}(S, T) = H(S) - \left( \frac{|S_{\text{left}}|}{|S|} H(S_{\text{left}}) + \frac{|S_{\text{right}}|}{|S|} H(S_{\text{right}}) \right) $$

We choose the split that maximizes Information Gain.

### 1.2 Gini Impurity
An alternative metric used by the CART algorithm is **Gini Impurity**:

$$ I_G(S) = 1 - \sum_{c=1}^C p_c^2 $$

Gini measures the probability of misclassifying a randomly chosen element from the set if it were randomly labeled according to the distribution of labels in the subset. Gini is computationally faster to calculate than entropy because it does not require logarithmic operations.

---

## Part 2: Overfitting and Regulating Trees

A decision tree left unconstrained can split until every leaf node contains exactly one sample. Such a tree has **zero training error** but **high variance** (overfitting).

To control overfitting, we apply regularization parameters:
- **Maximum Depth:** Limit how deep the tree can grow.
- **Minimum Samples Split:** Minimum number of samples a node must have before it can be split.
- **Minimum Samples Leaf:** Minimum number of samples a leaf node must contain.
- **Pruning:** Post-processing to remove splits that provide little predictive value.

---

## Part 3: Bagging and Random Forests

Single decision trees have high variance. We can reduce variance by combining multiple trees into an ensemble.

### 3.1 Bagging (Bootstrap Aggregating)
Suppose we have $B$ independent datasets. We could train $B$ separate models $h_1(\mathbf{x}), \dots, h_B(\mathbf{x})$ and average their predictions:

$$ h_{\text{avg}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B h_b(\mathbf{x}) $$

If each model has variance $\sigma^2$ and they are completely independent, the variance of the average model is:

$$ \text{Var}(h_{\text{avg}}) = \frac{\sigma^2}{B} $$

Averaging independent models **reduces variance by a factor of $B$** while leaving bias unchanged!

Since we don't have $B$ independent datasets in reality, we use **bootstrapping**: we sample $N$ points from our single dataset *with replacement* to create $B$ pseudo-datasets.

### 3.2 Random Forests
If we train standard decision trees on bootstrapped datasets, the trees will be highly correlated because they will all select the same strong features for their top splits. Averaging correlated models does not reduce variance as effectively.

**Random Forests** solve this by adding feature randomness: at each node split, only a **random subset of $k$ features** (usually $k = \sqrt{d}$) is considered. This decorrelates the trees, maximizing the variance reduction from bagging.

```python
from sklearn.ensemble import RandomForestClassifier

# Create and fit a random forest
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
```

---

## Part 4: Practice Exercises

### Exercise 1: Gini Impurity Split from Scratch
1. Given a node with 10 samples of Class A and 10 samples of Class B.
2. Suppose a split divides this into:
   - Left child: 8 Class A, 2 Class B.
   - Right child: 2 Class A, 8 Class B.
3. Calculate the Gini impurity before the split, the Gini impurity of each child, and the weighted Gini decrease.
4. **Code it:** Write a Python function `gini_impurity(y)` that takes an array of labels and returns the Gini impurity.

### Exercise 2: Feature Importance Extraction
1. Load a multi-feature dataset (e.g., standard wine or breast cancer dataset from `sklearn.datasets`).
2. Train a `RandomForestClassifier`.
3. **Code it:** Extract the feature importances (`rf.feature_importances_`) and plot them as a sorted bar chart. Explain how random forest calculates feature importance (Mean Decrease in Impurity).

---

## Self-Test Questions

1. **Why does bagging not reduce bias?** *(Because bagging is simply an averaging process: $E[\frac{1}{B} \sum h_b] = E[h_b]$. If the base models are biased (e.g., shallow trees that underfit), the ensemble will share the exact same bias.)*
2. **What is the difference between Bagging and Random Forests?** *(Bagging averages trees trained on bootstrapped samples. Random Forests go a step further by restricting each split to a random subset of features, decorrelating the trees.)*
3. **What is an "Out-of-Bag" (OOB) error?** *(For each bootstrapped tree, about 37% of the training examples are left out. We can use these left-out samples to evaluate the tree's performance. The average OOB error across all trees serves as an honest validation score without needing a separate split.)*
4. **Why are decision trees invariant to feature scaling?** *(Because a split is of the form $x_j < t$. Scaling a feature by a constant simply scales the threshold $t$ by the same constant, leaving the optimal split point and impurity calculations unchanged.)*
