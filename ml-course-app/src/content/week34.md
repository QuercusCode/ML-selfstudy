# Week 35: CS224n: Transformers, Pretraining, & Large Models

> **Goal:** Study the pretraining paradigms that enable large language models. You will compare Encoder (BERT), Decoder (GPT), and Encoder-Decoder (T5) architectures, analyze their distinct pretraining objectives, and explore the mathematical properties of scaling laws.

---

## Part 1: Pretraining vs. Fine-Tuning Paradigm

Before 2018, models were trained from scratch on task-specific datasets, requiring massive amounts of labeled data. Modern NLP uses a two-step transfer learning paradigm:
1. **Pretraining:** Train a large model on a massive unlabeled text corpus (web scrape) using an unsupervised self-supervised objective. The model learns syntax, grammar, and general knowledge.
2. **Fine-Tuning:** Adapt the pretrained model to a specific target task (e.g., classification, extraction) using a small labeled dataset.

---

## Part 2: The Three Transformer Model Families

Depending on the attention masking and task design, transformers are divided into three main families:

```
BERT (Encoder-Only):     Input tokens  <-->  Bidirectional Attention  <-->  Outputs
GPT (Decoder-Only):      Input tokens  --->  Causal (Masked) Attention  --->  Outputs
T5 (Encoder-Decoder):    Input  --->  Encoder  --->  Cross-Attention  --->  Decoder  --->  Output
```

### 2.1 Encoder-Only Models (e.g., BERT)
- **Mechanism:** Bidirectional attention. Every token can attend to all other tokens in the sequence (past and future).
- **Pretraining Objective:** Masked Language Modeling (MLM). We randomly replace 15% of the input tokens with a special `[MASK]` token and train the model to predict the original tokens.
- **Best suited for:** Natural Language Understanding (NLU), sequence classification, Named Entity Recognition (NER).

### 2.2 Decoder-Only Models (e.g., GPT)
- **Mechanism:** Causal attention. Tokens can only attend to previous tokens in the sequence.
- **Pretraining Objective:** Causal Language Modeling (CLM). Predict the next token given all past tokens:
  $$ L = \sum_{t=1}^T \log P(w_t \mid w_{1}, \dots, w_{t-1}) $$
- **Best suited for:** Autoregressive text generation.

### 2.3 Encoder-Decoder Models (e.g., T5, BART)
- **Mechanism:** An encoder processes the input sequence bidirectionally; a decoder generates the target sequence causally using cross-attention over the encoder's outputs.
- **Pretraining Objective:** Span Corruption. Random spans of input tokens are replaced with unique sentinel tokens (e.g., `[toy, car]` $\rightarrow$ `[X]`), and the decoder must predict the corrupted spans.
- **Best suited for:** Summarization, translation, text-to-text mappings.

---

## Part 3: Scaling Laws

Research (Kaplan et al., 2020) demonstrated that cross-entropy loss scales as a power-law relationship with three main variables: parameters ($N$), dataset size ($D$), and compute ($C$):

$$ L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}; \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}; \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C} $$

- **Conclusion:** As long as we scale parameters, data, and compute in tandem, performance continues to improve predictably without saturation. This finding catalyzed the race to build massive models.

---

## Part 4: Practice Exercises

### Exercise 1: Masked Language Model Preprocessing
1. Write a Python function `mask_tokens(token_ids, mask_token_id)` that:
   - Takes a list of token IDs.
   - Randomly selects 15% of the indices.
   - For 80% of selected indices, replace with `mask_token_id`.
   - For 10% of selected indices, replace with a random token ID.
   - For 10% of selected indices, leave unchanged.
2. **Code it:** Run the function and print the inputs and targets for the MLM loss calculation.

### Exercise 2: Comparing Model Architectures
Create a reference summary markdown table comparing:
- **BERT, GPT, and T5** on:
  1. Attention Masking Type.
  2. Pretraining Objective.
  3. Classification Task setup (e.g., where do you attach the classification head?).
  4. Sequence-generation capability.

---

## Self-Test Questions

1. **Why can't we use bidirectional attention for autoregressive language generation?** *(Because during generation, we predict one token at a time. If the model had bidirectional attention, it would attend to the future tokens it is trying to predict, causing trivial copy-paste solutions during training.)*
2. **Why does T5 convert all tasks (including classification and regression) into text-to-text format?** *(It simplifies model design. By mapping classification ("Is this positive?") to text output ("positive"), a single encoder-decoder model can handle translation, summarization, classification, and question answering without changing its output head.)*
3. **What is the difference between BERT's MLM objective and GPT's CLM objective in terms of representation quality?** *(BERT learns bidirectional representations, meaning the embedding of a word is influenced by its left and right context (excellent for meaning representation). GPT learns unidirectional representations, meaning a word's representation only encodes past context (excellent for generation but less optimal for extraction tasks).)*
4. **What are the implications of scaling laws for compute budgets?** *(They show that model performance is highly predictable. Rather than running expensive trials blindly, teams can run small-scale experiments to fit power-law curves and extrapolate the exact parameter and data size needed to reach a target performance.)*
