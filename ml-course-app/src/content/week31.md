# Week 32: Tokenizers & Byte-Pair Encoding (BPE)

> **Goal:** Study the critical interface between raw text and model inputs. You will understand character-level vs. word-level tokenization, trace the Byte-Pair Encoding (BPE) algorithm step-by-step, implement a subword merger from scratch, and analyze the trade-offs between vocabulary size and sequence length.

---

## Part 1: The Tokenization Problem

Neural networks require numerical inputs. We must map raw text (strings) into lists of integers (tokens).

### 1.1 Tokenization Approaches
- **Character-Level:** Treat each letter or punctuation mark as a token.
  - *Pros:* Small vocabulary size (~256 for ASCII); no Out-Of-Vocabulary (OOV) tokens.
  - *Cons:* Extremely long sequence lengths, making attention computations ($T^2$) very expensive.
- **Word-Level:** Split by spaces and treat words as tokens.
  - *Pros:* Short sequence lengths.
  - *Cons:* Massive vocabulary size (hundreds of thousands of words); cannot handle spelling errors or novel words (OOV tokens mapped to `[UNK]`).
- **Subword Tokenization (Modern Standard):** Split text into common character combinations (e.g., "tokenization" $\rightarrow$ "token", "ization"). Balances sequence length and vocabulary size.

---

## Part 2: Byte-Pair Encoding (BPE)

BPE is a compression algorithm adapted for subword tokenization (used by GPT-2, GPT-4, Llama).

### 2.1 The BPE Algorithm
1. **Initialize:** Define the vocabulary as all individual characters (and/or raw bytes) present in the training corpus. Represent the text corpus as a list of individual character tokens.
2. **Find Most Frequent Pair:** Iterate through the corpus and count all adjacent pairs of tokens (e.g., if `'h'` and `'e'` appear next to each other most frequently, their count is highest).
3. **Merge:** Add the merged pair `'he'` as a new token to the vocabulary, and replace all occurrences of `'h'` followed by `'e'` in the corpus with `'he'`.
4. **Repeat** steps 2 and 3 until the target vocabulary size is reached.

```python
# A single step of BPE pair counting
from collections import defaultdict

def get_stats(ids):
    # Counts frequencies of adjacent pairs of token IDs
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    # Replaces instances of 'pair' in 'ids' with 'new_id'
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
```

### 2.2 Byte-Level BPE
If the text contains foreign characters (e.g., emojis or non-English scripts), standard character-level BPE will create many `[UNK]` tokens. **Byte-Level BPE** maps text to UTF-8 bytes first. Since there are only 256 possible bytes, the base vocabulary is fixed at 256, allowing the tokenizer to represent *any* Unicode string without ever producing an unknown token.

---

## Part 3: The Vocabulary Size vs. Sequence Length Tradeoff

When configuring an LLM, your choice of target vocabulary size ($V$) has significant impacts:

| Vocabulary Size | Sequence Length | Embedding Matrix Memory | Softmax Computation |
|---|---|---|---|
| **Large (e.g., 100k)** | Shorter (fewer tokens/word) | High ($V \times d_e$ parameters) | Slow (computes scores over 100k classes) |
| **Small (e.g., 32k)** | Longer (more tokens/word) | Low | Fast |

---

## Part 4: Practice Exercises

### Exercise 1: BPE from Scratch
1. Given a small text corpus: `["lower", "newest", "widest"]`.
2. **Code it:** Write a script that runs 5 iterations of BPE. For each iteration, print the most frequent adjacent pair, add the merged token to the vocabulary, update the corpus, and display the growing vocabulary.

### Exercise 2: Tokenizing Biological Sequences
1. Take a list of 100 protein sequences (composed of 20 standard amino acid characters).
2. **Code it:** Build a BPE tokenizer. Train it with vocabulary sizes $V \in \{25, 100, 500\}$. For each size, compute and plot the average sequence length. Show how BPE creates "subword" motifs (common amino acid patterns) in the vocabulary.

---

## Self-Test Questions

1. **Why does Byte-Level BPE guarantee that we will never encounter an Out-Of-Vocabulary (OOV) token?** *(Because the base vocabulary contains all 256 possible bytes. Any string, regardless of language or formatting, can be encoded into UTF-8 bytes, meaning it can be represented using only tokens in our base vocabulary.)*
2. **What does a tokenizer do when it encounters an input that is not present in its merge rules?** *(It splits the input down to smaller subwords, eventually falling back to individual characters or raw bytes from its base vocabulary if no larger merges are defined.)*
3. **If we increase the vocabulary size of a model, does the attention context length increase or decrease?** *(The physical context window (e.g., 2048 tokens) remains the same. However, since a larger vocabulary compresses text into fewer tokens, the model can fit more actual words/information inside that same token window, effectively expanding the model's informational context.)*
4. **Why should you never share a tokenizer trained on English with a model being trained on Japanese?** *(Because the BPE merge rules trained on English will not capture common Japanese character patterns. The tokenizer will be forced to split Japanese text into individual characters or bytes, producing extremely long sequences that exhaust the model's context window and slow down inference.)*
