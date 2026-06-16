# Week 33: CS224n Foundations: Word Vectors

> **Goal:** Explore the historical foundations of representation learning in NLP. You will understand why one-hot word encodings fail to capture similarity, derive the Word2Vec Skip-Gram objective, study how Negative Sampling approximates the partition function, and run vector analogy queries.

---

## Part 1: Representing Word Meaning

To process words, we must convert them to numbers.

### 1.1 One-Hot Encodings
In a vocabulary of size $V$, we can represent each word as a vector of size $V$ with a single $1$ and the rest $0$.
- **Limitation:** All one-hot vectors are orthogonal:
  $$ \mathbf{w}_{\text{hotel}}^T \mathbf{w}_{\text{motel}} = 0 $$
  One-hot representations contain **no information about word similarity**. The model cannot generalize from "hotel" to "motel."

### 1.2 Dense Word Vectors (Word Embeddings)
We instead represent each word as a dense vector in a low-dimensional continuous space (e.g., $d = 300$). The similarity between words is measured by the cosine similarity of their vectors:

$$ \text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} $$

---

## Part 2: The Word2Vec Skip-Gram Model

Developed by Mikolov et al. (2013), Word2Vec learns word representations by predicting surrounding context words using a central word.

### 2.1 The Skip-Gram Objective
Given a sequence of words $w_1, w_2, \dots, w_T$, we define a sliding context window of size $m$ around a center word $w_t$. We want to maximize the probability of context words given the center word:

$$ L(\mathbf{\theta}) = \prod_{t=1}^T \prod_{-m \leq j \leq m, j \neq 0} P(w_{t+j} \mid w_t; \mathbf{\theta}) $$

For each word $w$, we learn two vectors: $\mathbf{v}_w$ when it acts as a center word, and $\mathbf{u}_w$ when it acts as a context word. The probability of context word $O$ given center word $C$ is computed using the softmax function:

$$ P(O \mid C) = \frac{e^{\mathbf{u}_O^T \mathbf{v}_C}}{\sum_{w=1}^V e^{\mathbf{u}_w^T \mathbf{v}_C}} $$

---

## Part 3: Negative Sampling

The denominator of the softmax equation requires summing over all words in the vocabulary $V$ (which can be $100,000+$ words). This makes computing the exact gradient extremely slow.

**Negative Sampling** solves this by approximating the softmax. For each positive pair (center word $C$, actual context word $O$), we sample $K$ random "negative" words from the vocabulary (noise words that do not appear in the context).

We train the model to classify whether a word pair is positive or negative using binary logistic regression:

$$ J_{\text{neg\_sample}}(\mathbf{\theta}) = -\log \sigma\left(\mathbf{u}_O^T \mathbf{v}_C\right) - \sum_{k=1}^K \log \sigma\left(-\mathbf{u}_{w_k}^T \mathbf{v}_C\right) $$

where $w_k$ is the $k$-th sampled negative word and $\sigma(z) = \frac{1}{1 + e^{-z}}$. This replaces the expensive summation over $V$ with a small summation over $K$ (typically $K \in [5, 20]$), accelerating training.

---

## Part 4: Vector Space Geometry (Analogies)

Once trained, word vectors exhibit remarkable geometric properties. Linear translations capture semantic relationships:

$$ \mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}} $$

```python
# Conceptual analogy lookup using cosine similarity
def find_analogy(a, b, c, word_vectors):
    # Finds word d such that a is to b as c is to d (d = c + b - a)
    target_vector = word_vectors[c] + word_vectors[b] - word_vectors[a]
    
    best_word = None
    max_sim = -1.0
    for word, vector in word_vectors.items():
        if word in [a, b, c]:
            continue
        sim = dot(vector, target_vector) / (norm(vector) * norm(target_vector))
        if sim > max_sim:
            max_sim = sim
            best_word = word
    return best_word
```

---

## Practice Exercises

### Exercise 1: Negative Sampling Loss
1. Given a center word vector $\mathbf{v}_C$ (size 5), a positive context word vector $\mathbf{u}_O$ (size 5), and three negative word vectors $\mathbf{u}_{N1}, \mathbf{u}_{N2}, \mathbf{u}_{N3}$.
2. **Code it:** Write a Python function that computes the negative sampling loss using the equation in Part 3.

### Exercise 2: Word2Vec Analogy Exploration
1. Load a pre-trained Word2Vec or GloVe model (using libraries like `gensim` or loading raw vector files).
2. **Code it:** Run three analogy queries (e.g., country-capital analogies, verb tenses) and print the closest word along with its similarity score.

---

## Self-Test Questions

1. **Why does Word2Vec learn two separate vectors ($\mathbf{u}_w$ and $\mathbf{v}_w$) for each word?** *(It simplifies optimization. If we used the same vector for both center and context representations, the dot product $\mathbf{v}_w^T \mathbf{v}_w$ would force words to be highly similar to themselves, introducing unwanted diagonal constraints during gradient descent.)*
2. **What does the Skip-Gram model predict vs. the Continuous Bag of Words (CBOW) model?** *(Skip-Gram takes a single center word and predicts its surrounding context words. CBOW takes a window of surrounding context words and predicts the single target center word.)*
3. **How are negative words sampled in Word2Vec?** *(They are sampled from a modified unigram distribution where word frequencies $U(w)$ are raised to the power of $3/4$: $P(w) = \frac{U(w)^{3/4}}{\sum_k U(w_k)^{3/4}}$. This increases the probability of sampling rare words and decreases the probability of sampling extremely common words like "the" and "of".)*
4. **Why do we use cosine similarity instead of Euclidean distance to measure word vector similarity?** *(Because Euclidean distance is highly sensitive to word frequency. High-frequency words will have larger gradients and drift further from the origin. Cosine similarity normalizes vector lengths, measuring only the direction/angle of vectors, which captures semantic meaning independent of word frequency.)*
