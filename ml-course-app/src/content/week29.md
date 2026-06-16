# Week 30: WaveNet, Dilated Convolutions, & Block Project

> **Goal:** Explore non-recurrent sequence models. You will learn the mechanics of causal and dilated convolutions in WaveNet, see how they achieve exponential receptive fields, and consolidate the Deep Learning Foundations block by building a comprehensive sequence or vision paper-replication project.

---

## Part 1: Causal and Dilated Convolutions

Recurrent networks (LSTMs) cannot process timesteps in parallel. CNNs can, but standard convolutions look into both the past and the future of a sequence.

### 1.1 Causal Convolutions
To use convolutions for sequence generation, we must enforce temporal ordering. A convolution is **causal** if the output at timestep $t$ depends only on inputs from timesteps $t, t-1, \dots$ in the previous layer.
- **Implementation:** In PyTorch, we can implement 1D causal convolutions by applying standard `nn.Conv1d` and padding the input on the left by $(K - 1)$, where $K$ is the kernel size, then slicing off the right side of the output.

### 1.2 Dilated Convolutions
A standard causal convolution has a receptive field that grows linearly with network depth. To capture long-range dependencies (e.g., audio or long text), we would need hundreds of layers.

**Dilated Convolutions** introduce gaps in the kernel. A dilation factor $D$ means the filter applies to inputs spaced $D$ steps apart:

$$ Y_t = \sum_{k=0}^{K-1} X_{t - D \cdot k} K_k $$

By doubling the dilation factor at each subsequent layer ($D \in \{1, 2, 4, 8, 16, \dots\}$), the **receptive field grows exponentially** with network depth, allowing the model to capture thousands of steps of context with only a few layers.

```
Dilation D=4:  o   o   o   o   (Kernel applies to points with spaces of 4)
Dilation D=2:  o o o o o o o o
Dilation D=1:  ooooooooooooooo
Input:         xxxxxxxxxxxxxxx
```

---

## Part 2: The WaveNet Architecture

Developed by Google DeepMind (van den Oord et al., 2016), WaveNet uses stacked dilated causal convolutions to generate high-fidelity raw audio waveforms.

Key components:
1. **Gated Activation Units:** Instead of ReLU, WaveNet uses a gated activation:
   $$ \mathbf{z} = \tanh\left( W_{f,k} * \mathbf{x} \right) \odot \sigma\left( W_{g,k} * \mathbf{x} \right) $$
2. **Residual Connections:** The input to a block is added back to its output.
3. **Skip Connections:** Intermediate features from all layers are fed directly to the output layers to generate final predictions.

---

## Part 3: Deep Learning Foundations Block Project

To complete the Deep Learning Foundations block, you will consolidate your knowledge by replicating a core result or figure from a classic deep learning paper.

### 3.1 Recommended Projects
- **Sequence Generation:** Build a character-level WaveNet or LSTM model, train it on a large text/protein dataset, and analyze output properties.
- **Vision Classification:** Replicate a ResNet-style classifier on a custom dataset (microscopy, CIFAR-10), demonstrating data efficiency using transfer learning.
- **Generative:** Build a Variational Autoencoder (VAE) or Generative Adversarial Network (GAN) to generate new samples (e.g. MNIST digits).

### 3.2 Deliverable Requirements
1. **Github-style Repository:** Clean code with modules for model architecture, data loading, training loop, and evaluation.
2. **Weights & Biases Logs:** Run a hyperparameter sweep and log training curves.
3. **Written Report:** A short walkthrough documenting model choices, what you tested, and a comparison plot matching the paper's findings.

---

## Practice Exercises

### Exercise 1: Causal Padding Calculation
1. Given an input sequence of length 10.
2. We want to apply a 1D convolution with kernel size $K=3$ and dilation $D=2$.
3. **Code it:** Calculate the left padding required to make this convolution causal (so that output index $t$ depends only on input indexes $\leq t$). Write a short PyTorch script verifying your calculation.

### Exercise 2: Simple WaveNet Block
1. Write a custom PyTorch module `WaveNetBlock(channels, dilation)` containing:
   - Dilated causal convolution.
   - Gated activation unit (Tanh + Sigmoid).
   - Residual addition.
2. **Code it:** Implement the forward pass and print tensor shapes to verify consistency.

---

## Self-Test Questions

1. **Why does WaveNet perform better than LSTMs for sequence processing?** *(WaveNet uses convolutions, allowing it to process all timesteps of a sequence in parallel during training. LSTMs must process timesteps sequentially, preventing parallelization on GPUs.)*
2. **If we stack 4 layers with kernel size $K=2$ and dilation factors $D \in \{1, 2, 4, 8\}$, what is the receptive field size of the final output node?** *(Receptive field size is $1 + \sum_{l} (K - 1) D_l = 1 + (2-1)(1 + 2 + 4 + 8) = 16$ timesteps.)*
3. **What is the purpose of skip connections in WaveNet?** *(Because WaveNet gets very deep, skip connections allow the network to route low-level spatial features from early layers directly to the final classification layers, preventing features from being lost during deep feedforward operations.)*
4. **Why is autoregressive sampling slow in WaveNet during inference?** *(Because during generation, we must predict one sample at a time. The model needs the output at step $t$ to calculate the input for step $t+1$, creating a sequential bottleneck during generation even though training is parallel.)*
