# Week 41: Structure & Design Models: AlphaFold2, ProteinMPNN, & RFdiffusion

> **Goal:** Deconstruct the core models used in modern computational protein engineering. You will learn how AlphaFold2 predicts 3D structures from evolutionary data, understand the "inverse folding" problem solved by ProteinMPNN, and trace the de novo protein design pipeline using RFdiffusion.

---

## Part 1: Representing Proteins in 3D Space

To model protein structures, we must represent 3D geometry in a format neural networks can process.

- **Rigid Body Representation:** The backbone of each amino acid residue is defined by three main atoms: Nitrogen ($N$), Alpha Carbon ($C_\alpha$), and Carbon ($C$). We model this backbone as a rigid body (a "frame") with:
  - A translation vector $\mathbf{t} \in \mathbb{R}^3$ (position of $C_\alpha$ in space).
  - A rotation matrix $R \in \text{SO}(3)$ (orientation of the residue relative to a reference frame).
- **Distance Matrix:** A $L \times L$ matrix where cell $(i, j)$ represents the Euclidean distance between residue $i$ and residue $j$ in 3D space:
  $$ D_{ij} = \|\mathbf{t}_i - \mathbf{t}_j\|_2 $$

---

## Part 2: AlphaFold2 Structure Prediction

AlphaFold2 (Jumper et al., 2021) revolutionized structural biology by predicting 3D coordinates directly from sequence.

```
Input -> [ MSA + Templates ] -> [ Evoformer (MSA & Pair Interaction) ] -> [ Structure Module (IPA) ] -> 3D Coordinates
```

Key Innovations:
1. **Evoformer:** An attention block that simultaneously processes:
   - **MSA representation:** Evolutionary information (how residues co-vary across species).
   - **Pair representation:** Spatial information (which residues are close in 3D space).
   These representations continuously exchange information: covariance in MSA suggests proximity, which updates the pair representation.
2. **Invariant Point Attention (IPA):** Inside the Structure Module, IPA calculates attention over 3D coordinates in a way that is invariant to global rotations and translations of the protein, ensuring that moving the protein in space does not change the prediction.

---

## Part 3: Inverse Folding and ProteinMPNN

- **Structure Prediction (AlphaFold2):** Sequence $\rightarrow$ 3D Structure.
- **Inverse Folding (ProteinMPNN):** 3D Structure $\rightarrow$ Sequence.

Given a target 3D backbone shape (e.g., a designed binding pocket), we want to find a sequence of amino acids that will fold into this exact shape.

**ProteinMPNN** (Dauparas et al., 2022) models the backbone as a graph:
- **Nodes:** Residues (features encode positions, angles).
- **Edges:** Spatial relationships between neighboring residues.

Using a message-passing Graph Neural Network (GNN), the model calculates features over the graph and outputs probability distributions over the 20 amino acids autoregressively, token-by-token:

$$ P(s_i \mid s_{<i}, \text{backbone}) $$

---

## Part 4: De Novo Design with RFdiffusion

**RFdiffusion** (Watson et al., 2023) combines diffusion models with AlphaFold2's structural network to generate completely novel 3D backbones.

### 4.1 The Protein Design Pipeline
To design a functional protein (e.g., a new enzyme or binder) from scratch:
1. **RFdiffusion:** Starts with random coordinates in 3D space and gradually denoises them into a valid, novel protein backbone structure.
2. **ProteinMPNN:** Takes the generated 3D backbone and designs compatible amino acid sequences.
3. **AlphaFold2:** Evaluates the designed sequences by folding them. If the predicted structure matches the RFdiffusion target backbone, the design is validated and ready for wet-lab synthesis.

---

## Practice Exercises

### Exercise 1: Pairwise Distance Calculator
1. Given a batch of 3D coordinates representing $C_\alpha$ positions of a protein: shape $B \times L \times 3$.
2. **Code it:** Write a PyTorch function `compute_distance_matrix(coords)` that returns the pairwise Euclidean distance matrix of shape $B \times L \times L$ using vectorized operations (no loops).

### Exercise 2: GNN Message Passing Step
1. Given node features $H \in \mathbb{R}^{L \times d}$ and an edge index list of shape $2 \times E$.
2. **Code it:** Write a simple message-passing step in PyTorch: for each node, aggregate the features of its spatial neighbors (distance $< 10\text{Å}$) and update its node representation using a linear layer.

---

## Self-Test Questions

1. **Why does AlphaFold2 use MSA (Multiple Sequence Alignment) data?** *(Because evolution leaves clues. If two amino acids mutate together across different species (co-evolution), it implies they interact and are physically close in 3D space to maintain protein stability. The MSA encodes these physical constraints.)*
2. **What does it mean for an operation to be "rotationally invariant"?** *(An operation $f(X)$ is rotationally invariant if applying a rotation $R$ to the input coordinates does not change the output: $f(RX) = f(X)$. This is critical for structural biology models because a protein's biological properties are identical regardless of how it is rotated in space.)*
3. **Why is ProteinMPNN preferred over older physics-based models (like Rosetta) for sequence design?** *(Rosetta uses expensive physical forcefield calculations and stochastic search, taking hours to design a single sequence. ProteinMPNN is trained on thousands of native PDB structures, allowing it to design high-recovery sequences in seconds with higher experimental success rates.)*
4. **What is the difference between "hallucination" and "diffusion" in protein design?** *(Hallucination optimizes a sequence by running gradient descent on AlphaFold's confidence metrics, which is slow and tends to get stuck in local minima. Diffusion denoises coordinates directly in 3D space, exploring the structural landscape faster and producing more diverse and complex protein folds.)*
