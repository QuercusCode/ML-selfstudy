# Week 34: CS224n: Dependency Parsing & Seq2Seq Attention

> **Goal:** Bridge the gap between word embeddings and transformers. You will study how syntactic structures are resolved using dependency parsing, examine the encoder-decoder sequence-to-sequence (Seq2Seq) architecture, understand how attention resolves the Seq2Seq bottleneck, and visualize attention heatmaps.

---

## Part 1: Dependency Parsing

In natural language processing, understanding sentence structure is critical. **Dependency parsing** identifies grammatical relationships between words (e.g., finding the subject, object, and modifiers of a verb).

```
      root
       |
     [ eats ]
     /      \
  nsubj     obj
   /          \
[ She ]     [ apples ]
```

### 1.1 Transition-Based Dependency Parsing
A popular approach is transition-based parsing. The parser maintains:
- A **stack** containing words currently being processed (starts with `[ROOT]`).
- A **buffer** containing remaining words in the sentence.
- A set of **relations** (dependencies identified).

At each step, a classifier predicts one of three actions:
1. **SHIFT:** Move the first word of the buffer onto the stack.
2. **LEFT-ARC:** Create a dependency head $\leftarrow$ dependent between the top two words on the stack, and remove the second word.
3. **RIGHT-ARC:** Create a dependency head $\rightarrow$ dependent between the top two words on the stack, and remove the first word.

Modern parsers use neural networks to predict these transitions based on embeddings of words on the stack and buffer.

---

## Part 2: The Seq2Seq Bottleneck and Attention

Before the Transformer, sequence-to-sequence tasks (like machine translation) utilized recurrent Encoder-Decoder networks.

### 2.1 The Seq2Seq Bottleneck
- **Encoder:** An LSTM processes the input sentence token-by-token, compressing it into a single final hidden state vector $\mathbf{h}_T$ (the context vector).
- **Decoder:** Another LSTM takes the context vector $\mathbf{h}_T$ as its initial hidden state and generates the target translation word-by-word.
- **The Bottleneck:** Compressing a long sentence (e.g., 50 words) into a single fixed-size vector causes massive information loss. The decoder struggles to remember the beginning of the sentence.

### 2.2 Luong (Dot-Product) Attention
Attention solves the bottleneck by allowing the decoder to look at *all* encoder hidden states $\mathbf{h}_1, \dots, \mathbf{h}_{T_{\text{in}}}$ at each step.

Given the current decoder hidden state $\mathbf{s}_t$:
1. **Compute Alignment Scores:** Compare $\mathbf{s}_t$ to each encoder hidden state $\mathbf{h}_i$ using a dot product:
   $$ e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i $$
2. **Compute Attention Weights:** Normalize scores using softmax:
   $$ \alpha_{t,i} = \frac{e^{e_{t,i}}}{\sum_{k=1}^{T_{\text{in}}} e^{e_{t,k}}} $$
3. **Compute Context Vector:** Calculate the weighted sum of encoder hidden states:
   $$ \mathbf{c}_t = \sum_{i=1}^{T_{\text{in}}} \alpha_{t,i} \mathbf{h}_i $$
4. **Generate Output:** Combine $\mathbf{c}_t$ and $\mathbf{s}_t$ to predict the next word.

---

## Part 3: Visualizing Attention Heatmaps

An elegant property of attention is interpretability. We can plot the weights $\alpha_{t,i}$ as a 2D grid (heatmap) to see exactly which words the model focused on during translation.

```python
# Conceptual attention weight plotting
import matplotlib.pyplot as plt
import numpy as np

# Sample alignment matrix (French to English)
src_words = ["je", "mange", "une", "pomme"]
tgt_words = ["i", "eat", "an", "apple"]
alignment = np.array([
    [0.9, 0.0, 0.0, 0.1], # i -> je
    [0.0, 0.9, 0.0, 0.1], # eat -> mange
    [0.0, 0.0, 0.9, 0.1], # an -> une
    [0.0, 0.1, 0.0, 0.9]  # apple -> pomme
])

fig, ax = plt.subplots()
im = ax.imshow(alignment, cmap='viridis')
ax.set_xticks(np.arange(len(src_words)))
ax.set_yticks(np.arange(len(tgt_words)))
ax.set_xticklabels(src_words)
ax.set_yticklabels(tgt_words)
plt.colorbar(im)
plt.show()
```

---

## Part 4: Practice Exercises

### Exercise 1: Parsing Transition Verification
Given the sentence: `"She eats apples"` (initial buffer: `["She", "eats", "apples"]`).
Write out the sequence of stack, buffer, and relation states that leads to the correct parse tree (using SHIFT, LEFT-ARC, and RIGHT-ARC actions).

### Exercise 2: Luong Attention Implementation
1. Write a custom PyTorch module `LuongAttention(hidden_dim)` containing:
   - Dot-product scoring.
   - Softmax normalization.
   - Context vector weighted sum.
2. **Code it:** Pass dummy decoder hidden state tensor (shape $B \times 1 \times H$) and encoder output states (shape $B \times T \times H$) and verify that the output context vector shape is $B \times 1 \times H$.

---

## Self-Test Questions

1. **What is the difference between transition-based parsing and constituency parsing?** *(Transition-based parsing identifies grammatical relations between individual words (dependencies). Constituency parsing groups words into nested phrases based on phrase-structure rules (e.g., Noun Phrases, Verb Phrases).)*
2. **How does Seq2Seq attention solve the information bottleneck?** *(By allowing the decoder to query all intermediate hidden states of the encoder at each step, rather than forcing it to rely solely on a single compressed vector at the end of the encoder.)*
3. **What is the difference between Bahdanau (additive) and Luong (multiplicative) attention?** *(Bahdanau attention calculates scores using a small neural network: $score(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{v}_a^T \tanh(W_a [\mathbf{s}_t; \mathbf{h}_i])$. Luong attention uses a simple dot product: $score(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{s}_t^T W_a \mathbf{h}_i$, which is faster and easier to calculate in matrix form.)*
4. **If a translation attention heatmap displays a diagonal alignment, what does that tell you?** *(It indicates that the source and target languages share a similar word order, meaning the model reads and translates tokens sequentially from left to right.)*
