# Week 22: makemore: Bigram → MLP Language Model

> **Goal:** Build your first autoregressive character-level language models. You will understand count-based bigram models, implement an equivalent single-layer neural network trained with gradient descent, and construct a Bengio-style Multi-Layer Perceptron (MLP) language model with character embeddings.

---

## Part 1: The Bigram Model (Two Ways)

A bigram language model predicts the next character in a sequence given only the current character. It assumes the Markov property: $P(X_t \mid X_{t-1}, \dots, X_1) = P(X_t \mid X_{t-1})$.

### 1.1 The Count-Based Approach
We count the frequencies of all character pairs in our training corpus and store them in a 2D matrix $N$, where row indices represent the current character and column indices represent the next character. We normalize each row to sum to 1 to obtain a transition probability matrix $P$:

$$ P(c_{\text{next}} \mid c_{\text{curr}}) = \frac{N(c_{\text{curr}}, c_{\text{next}})}{\sum_{c} N(c_{\text{curr}}, c)} $$

To sample from this model, we index into row $c_{\text{curr}}$ and use `np.random.choice` on the probability distribution.

```python
import numpy as np

# A toy corpus
words = ["hello", "world"]

# Create vocabulary mapping
chars = sorted(list(set(''.join(words) + '.'))) # '.' represents start/end token
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
vocab_size = len(chars)

# Count matrix
N = np.zeros((vocab_size, vocab_size), dtype=np.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# Normalize to get probabilities
P = (N + 1).astype(np.float32)  # Added 1 for Laplace smoothing
P /= P.sum(axis=1, keepdims=True)
```

### 1.2 The Neural Network Approach
We can represent the current character as a one-hot vector $\mathbf{x} \in \{0, 1\}^{\text{vocab\_size}}$. We pass this vector through a single linear layer with weights $W \in \mathbb{R}^{\text{vocab\_size} \times \text{vocab\_size}}$:

$$ \mathbf{z} = \mathbf{x} W $$

We interpret the outputs $\mathbf{z}$ as log-counts (logits). To convert them to probabilities, we pass them through the **softmax function**:

$$ p_c = \frac{e^{z_c}}{\sum_k e^{z_k}} $$

We train this network to minimize cross-entropy loss using gradient descent. Over time, the weights $W$ will converge such that $e^W$ matches the normalized count matrix $P$ exactly!

---

## Part 2: The Bengio-Style MLP Language Model

Predicting the next character using only one previous character yields poor text generation. To use a larger context window (e.g., 3 previous characters) without our transition table growing exponentially, we map characters to low-dimensional continuous vector spaces. This is the **MLP character language model** (based on Bengio et al., 2003).

```
Context (3 chars) -> [One-Hot Codes] -> [Embedding C Matrix] -> [Concatenate Embeddings] -> [Linear + tanh] -> [Linear] -> [Softmax] -> Next Char
```

### 2.1 Character Embeddings
Let $C \in \mathbb{R}^{\text{vocab\_size} \times d}$ be our embedding matrix, where $d$ is the embedding dimension. If the context contains 3 characters, we look up their vectors in $C$ and concatenate them into a single feature vector $\mathbf{e} \in \mathbb{R}^{3d}$.

### 2.2 Forward Pass
1. **Embedding Lookup:**
   $$ \mathbf{e} = [\mathbf{c}_1; \mathbf{c}_2; \mathbf{c}_3] $$
2. **Hidden Layer:**
   $$ \mathbf{h} = \tanh(\mathbf{e} W_1 + \mathbf{b}_1) $$
3. **Output Logits:**
   $$ \mathbf{z} = \mathbf{h} W_2 + \mathbf{b}_2 $$
4. **Loss Function (Cross-Entropy):**
   $$ L = -\log P(y \mid x) = -z_y + \log \sum_j e^{z_j} $$

---

## Part 3: Practice Exercises

### Exercise 1: Single-Layer Net vs. Count Matrix
1. Generate training pairs $(x, y)$ from a list of names for a bigram model.
2. Implement the single-layer neural network approach in PyTorch from scratch (no `nn.Linear` or `nn.CrossEntropyLoss`).
3. **Code it:** Train the model and show that the resulting probability distribution $e^W / \sum e^W$ matches the normalized count matrix $P$ to within numerical tolerance.

### Exercise 2: Implementing the MLP Model
1. Write a PyTorch model for the Bengio-style MLP character generator (using a context window of 3, embedding size of 2, and 100 hidden neurons).
2. **Code it:** Implement a generation loop that takes a context `...`, looks up embeddings, runs the forward pass, samples the next token, shifts the context window, and repeats until the end token `.` is generated.

---

## Self-Test Questions

1. **Why is the count-based approach infeasible if we increase the context window to 5 characters?** *(Because the count table size scales as $V^C$, where $V$ is the vocabulary size and $C$ is the context length. For $V = 27$ and $C = 5$, the table would have $27^5 \approx 14.3$ million cells, making counts extremely sparse and memory-intensive. MLP embeddings scale linearly with context size $C \times d$.)*
2. **What does the embedding matrix $C$ represent geometrically?** *(It maps discrete characters to coordinates in a continuous $d$-dimensional space. During training, characters that share similar semantic roles (e.g., vowels like 'a', 'e', 'o') are pushed closer together in this space.)*
3. **What is the mathematical definition of logits?** *(Logits are the raw, unnormalized outputs of a neural network layer ($\mathbf{z} = \mathbf{h} W + \mathbf{b}$) prior to applying a normalizing activation function like softmax.)*
4. **Why is it important to use `F.cross_entropy` instead of doing `softmax` followed by `log` manually?** *(Because computing softmax and log separately is numerically unstable. Large logits can cause overflow in `exp`, and probabilities near 0 cause `log` to yield negative infinity. `F.cross_entropy` uses the log-sum-exp trick to perform these operations safely.)*
