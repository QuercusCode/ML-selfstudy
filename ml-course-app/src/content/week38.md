# Week 39: Protein Language Models (ESM-2)

> **Goal:** Transition from natural language to biological sequences. You will understand how protein language models (like ESM-2) learn representations from evolutionary sequences, extract per-residue and per-protein embeddings, and train downstream probe classifiers to predict biological properties.

---

## Part 1: Representation Learning on Biological Sequences

Proteins are linear chains of amino acids. We can represent a protein as a string where each character belongs to a 20-letter alphabet (the 20 standard amino acids).

```
Protein Sequence:  M  K  V  L  M  T  A  L  L  V  G  A  L  A  F
```

Just as large language models learn grammar and world knowledge by reading internet text, **Protein Language Models (PLMs)** learn the "grammar of biology" (protein folding, active sites, evolutionary constraints) by reading millions of raw, unlabeled protein sequences from databases like UniProt.

### 1.1 ESM-2 (Evolutionary Scale Modeling)
ESM-2 (developed by Meta AI) is a state-of-the-art PLM trained using a **Masked Language Modeling (MLM)** objective:
- Input sequences have 15% of their amino acids randomly masked out.
- The model uses bidirectional attention to predict the identity of the masked amino acids based on their surrounding context.
- To succeed, the model must implicitly learn how amino acids interact in 3D space, capturing structural and functional features directly from sequence.

---

## Part 2: Extracting Embeddings

We can use a pretrained ESM-2 model as a feature extractor.

### 2.1 Types of Embeddings
- **Per-Residue (Token-Level) Embeddings:** Output tensor of shape $L \times d$ (where $L$ is sequence length and $d$ is embedding dimension). Useful for token-level predictions like secondary structure (helix, sheet, loop) or active site locations.
- **Per-Protein (Sequence-Level) Embeddings:** A single vector of size $d$ representing the entire protein (usually computed by averaging the per-residue embeddings along the length dimension, called *mean pooling*). Useful for classification tasks like solubility, stability, or localization.

```python
import torch
from transformers import AutoTokenizer, EsmModel

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

sequence = "MKVLMTALLVGALAF"
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
# Per-residue embeddings: shape (1, L + 2, d) - includes CLS/EOS tokens
last_hidden_states = outputs.last_hidden_state

# Per-protein embedding (mean pooled over residues, omitting CLS/EOS)
protein_embedding = last_hidden_states[0, 1:-1].mean(dim=0)
```

---

## Part 3: Downstream Probing

Because biological datasets are often very small (e.g., only 100 proteins tested in a wet lab), fine-tuning a whole transformer will lead to severe overfitting.

Instead, we perform **downstream probing**:
1. Freeze the ESM-2 model.
2. Pass your dataset through ESM-2 to extract static embeddings for each protein.
3. Train a simple linear or ridge regression model (a **probe**) on top of these static embeddings.

Because the embeddings contain high-quality structural information, even a simple linear probe can predict complex properties like binding affinity or thermal stability.

---

## Part 4: Practice Exercises

### Exercise 1: Extracting Biological Embeddings
1. Load the `esm2_t6_8M_UR50D` model (the fast 8-million parameter variant).
2. **Code it:** Write a function `get_sequence_embedding(sequence)` that takes a string sequence, runs inference, performs mean pooling, and returns a 1D NumPy array representing the protein.

### Exercise 2: Solubility Predictor Probe
1. Create a simulated dataset: 50 soluble protein sequences and 50 insoluble protein sequences.
2. Extract the mean-pooled protein embeddings for each sequence using your function from Exercise 1.
3. **Code it:** Train a `RidgeClassifier` from scikit-learn on these embeddings and report the cross-validated classification accuracy.

---

## Self-Test Questions

1. **Why does ESM-2 use character-level tokenization instead of BPE?** *(Because BPE is designed to find common letter combinations in human languages. In biology, there are only 20 amino acid characters, and arbitrary subword grouping does not respect biological boundaries (like codons or domains), making character-level processing much more clean and natural.)*
2. **What does the `<cls>` token embedding represent in ESM-2?** *(The `<cls>` token is prepended to the start of the sequence. During self-attention, it aggregates information from all tokens in the sequence, meaning its final state can serve as a representation of the entire protein, similar to mean pooling.)*
3. **If you want to predict whether a specific mutation (e.g., mutating Alanine at position 12 to Valine) ruins a protein, which embeddings should you use?** *(You should use the per-residue embeddings at index 12. You can compare the embedding vector of the wild-type sequence with that of the mutated sequence, or train a classifier on the difference vector.)*
4. **Why are protein language model embeddings referred to as "implicit structure representations"?** *(Because although the model is trained only on sequence strings, to predict masked amino acids it must understand which residues are close in 3D space (e.g., forming disulfide bonds or hydrophobic cores). Pairwise attention weights in ESM-2 correlate highly with actual physical contact maps.)*
