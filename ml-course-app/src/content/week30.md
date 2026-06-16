# Week 31: Attention from First Principles & Build GPT

> **Goal:** Deconstruct the core architecture of modern generative AI. You will derive and implement scaled dot-product attention, understand causal masking, build a multi-head attention module, and construct a decoder-only transformer (GPT-style) from scratch in PyTorch.

---

## Part 1: The Attention Mechanism

Traditional sequence models (LSTMs) compress history into a single vector. Attention allows the model to look back at all historical timesteps and dynamically focus on the most relevant information.

### 1.1 Query, Key, and Value Vectors
For each token at index $i$, we project its embedding into three vectors:
- **Query ($\mathbf{q}_i$):** What information the current token is looking for.
- **Key ($\mathbf{k}_i$):** What information the token contains.
- **Value ($\mathbf{v}_i$):** The actual content vector that will be aggregated.

### 1.2 Scaled Dot-Product Attention
Given queries $Q \in \mathbb{R}^{T \times d_k}$, keys $K \in \mathbb{R}^{T \times d_k}$, and values $V \in \mathbb{R}^{T \times d_v}$:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V $$

- **Why divide by $\sqrt{d_k}$?** If $d_k$ is large, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients (vanishing gradients). Scaling by $\sqrt{d_k}$ keeps the variance of the dot products equal to 1, stabilizing training.

### 1.3 Causal Masking
In a decoder-only language model (like GPT), we predict the next token given past tokens. The model must not "look ahead" into the future. We enforce this by applying a causal mask to the attention scores (logits) before the softmax step:

$$ M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases} $$
$$ \text{Scores} = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) $$

Adding $-\infty$ results in a softmax probability of exactly $0.0$ for all future tokens, blocking information leakage.

```python
import torch
import torch.nn.functional as F

T, dk = 4, 8
q = torch.randn(T, dk)
k = torch.randn(T, dk)
v = torch.randn(T, dk)

# Compute raw scores
scores = (q @ k.t()) / (dk ** 0.5)

# Apply causal mask: lower triangular matrix of ones
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))

# Softmax converts -inf to 0.0
weights = F.softmax(scores, dim=-1)
out = weights @ v
```

---

## Part 2: Multi-Head Attention and GPT Block

Rather than performing attention once, **Multi-Head Attention** splits the query, key, and value vectors into $h$ heads, performing attention in parallel in lower-dimensional spaces, and concatenating the outputs. This allows different heads to focus on different context locations (e.g., one head tracks syntax, another tracks pronouns).

```
Input -> [ LayerNorm ] -> [ Multi-Head Attention ] -> [ + Residual ] -> [ LayerNorm ] -> [ FeedForward MLP ] -> [ + Residual ] -> Output
```

Modern Transformers use **Pre-LayerNorm (Pre-LN)** residual blocks (applying LayerNorm *before* the attention/MLP layers), which stabilizes gradient flows and allows training of extremely deep networks.

---

## Part 3: Practice Exercises

### Exercise 1: Multi-Head Attention Layer
1. Write a custom PyTorch module `MultiHeadAttention(num_heads, head_dim)`.
2. It should project inputs into $Q, K, V$ matrices, split them into multiple heads, apply causal masking, and concatenate the outputs.
3. **Code it:** Verify that the output shape matches the input shape ($B \times T \times C$).

### Exercise 2: Assemble GPT
1. Build a PyTorch model `MiniGPT(vocab_size, n_embd, n_heads, n_layers, block_size)`.
2. Assemble it using token embeddings, positional embeddings, a stack of `Block` modules (MHA + MLP), a final `LayerNorm`, and a linear classification head.
3. **Code it:** Run a forward pass with dummy token inputs and verify it yields logits of shape ($B \times T \times \text{vocab\_size}$).

---

## Self-Test Questions

1. **Why is dividing by $\sqrt{d_k}$ mathematically necessary?** *(If components of two vectors $q$ and $k$ are independent random variables with mean 0 and variance 1, their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$. If $d_k$ is large, the variance is large, producing extreme values in softmax that saturate gradients. Dividing by $\sqrt{d_k}$ scales the variance back to 1.)*
2. **What is the difference between Self-Attention and Cross-Attention?** *(In Self-Attention, the Queries, Keys, and Values all originate from the same input sequence. In Cross-Attention (used in encoder-decoder translation models), the Queries come from the decoder sequence, while the Keys and Values come from the encoder output sequence.)*
3. **Why do we need Positional Embeddings in Transformers?** *(Unlike LSTMs, which process tokens sequentially, the attention equation is permutation-invariant. If you shuffle the input tokens, the attention output is identical (except shuffled). Positional embeddings add position vectors to token embeddings to encode sequence order.)*
4. **Why is Pre-LN preferred over Post-LN in modern architectures?** *(In Post-LN (used in the original Vaswani paper), normalization occurs after the residual addition. This causes gradients to grow exponentially near the output layers. Pre-LN normalizes inputs before layers, leaving the residual highway un-normalized, keeping gradient bounds stable across layers.)*
