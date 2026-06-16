# Week 24: Activations, Initialization, & BatchNorm

> **Goal:** Understand why deep neural networks fail to train and master the mechanisms used to stabilize them. You will study vanishing and exploding gradients, learn the mathematics behind Xavier and Kaiming initialization, understand the mechanics of Batch Normalization, and implement Dropout regularization.

---

## Part 1: Vanishing and Exploding Gradients

If we initialize a deep neural network incorrectly, our activations and gradients will either shrink to zero (vanish) or grow infinitely (explode) as we propagate through the layers.

### 1.1 The Mathematical Cause
Consider a simplified $L$-layer network without activations:

$$ \mathbf{a}^{[L]} = W^{[L]} W^{[L-1]} \dots W^{[1]} \mathbf{x} $$

If all weight matrices $W^{[l]}$ are scaled such that their eigenvalues are slightly larger than 1 (e.g., 1.5), then the activation size grows exponentially: $1.5^L \rightarrow \infty$. If they are slightly smaller than 1 (e.g., 0.5), they shrink exponentially: $0.5^L \rightarrow 0$.

During backprop, the gradient undergoes the same chain multiplication, leading to vanishing or exploding gradients. If gradients vanish, the weights in the early layers update extremely slowly, halting learning.

---

## Part 2: Weight Initialization

To prevent vanishing/exploding behaviors, we want the variance of the activations to remain constant across all layers.

### 2.1 Xavier (Glorot) Initialization
For a linear layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs followed by a symmetric activation function (e.g., Tanh), the weights should be sampled from a normal distribution with:

$$ W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) $$

### 2.2 Kaiming (He) Initialization
Symmetric initializers fail for **ReLU** activations because ReLU sets half of its inputs to zero (retaining only positive values). This cuts the variance of the output in half. To compensate, Kaiming initialization doubles the variance of the weights:

$$ W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right) $$

```python
import torch.nn as nn

# PyTorch default initialization for Linear layers is Kaiming-like, 
# but you can set it manually:
layer = nn.Linear(100, 50)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

---

## Part 3: Batch Normalization (BatchNorm)

Even with good initialization, the distribution of activations changes during training as weights update. This is called **internal covariate shift**. **Batch Normalization** stabilizes distributions by normalizing activations at each layer.

### 3.1 The BatchNorm Math
Given a mini-batch of activations $B = \{x^{(1)}, \dots, x^{(m)}\}$ for a specific feature, we:
1. **Compute Mean:**
   $$ \mu_B = \frac{1}{m} \sum_{i=1}^m x^{(i)} $$
2. **Compute Variance:**
   $$ \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_B)^2 $$
3. **Normalize:**
   $$ \hat{x}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$
   where $\epsilon > 0$ prevents division by zero.
4. **Scale and Shift:**
   $$ y^{(i)} = \gamma \hat{x}^{(i)} + \beta $$
   where $\gamma$ and $\beta$ are learnable parameters that allow the network to undo the normalization if doing so minimizes loss.

### 3.2 Training vs. Inference
- **During training:** We use the mean and variance of the current mini-batch.
- **During inference:** A single prediction does not have a mini-batch. We instead use a running average of the mean and variance computed during training:
  $$ \mu_{\text{run}} \leftarrow (1 - \text{momentum}) \mu_{\text{run}} + \text{momentum} \cdot \mu_B $$

---

## Part 4: Regularization — Dropout

Dropout is a simple regularization technique that prevents co-adaptation of features.

- **During training:** For each batch, we randomly zero out a fraction $p$ of activations. This forces the network to learn robust, redundant representations.
- **During inference:** No units are dropped. Instead, we scale down the activations by a factor of $1 - p$ (or use *inverted dropout* during training to divide by $1 - p$ so that no scaling is needed at test time).

---

## Practice Exercises

### Exercise 1: Visualizing Activation Collapse
1. Create a 10-layer MLP in PyTorch (each layer size 100).
2. Initialize weights using standard normal distribution scaled by 10 (exploding) and 0.01 (vanishing).
3. **Code it:** Pass a random input through the network and plot a histogram of the activation values at Layer 1, Layer 5, and Layer 10. Repeat the experiment using Kaiming initialization and compare the histograms.

### Exercise 2: Implementing a BatchNorm Layer
1. Write a custom PyTorch module `SimpleBatchNorm1d(num_features)` from scratch (do not use `nn.BatchNorm1d`).
2. **Code it:** Implement the forward pass showing both training mode (batch mean/std) and evaluation mode (running mean/std). Verify that it outputs the same values as PyTorch's native layer.

---

## Self-Test Questions

1. **Why does a standard normal distribution $\mathcal{N}(0, 1)$ cause exploding gradients in deep networks?** *(Because the variance of the sum of $n$ independent variables scales with $n$. If a layer has $n_{\text{in}} = 100$ inputs, the variance of the output activations will be $100$ times the input variance. Across multiple layers, this variance scales exponentially.)*
2. **Why does ReLU require a different initialization than Tanh?** *(Because ReLU maps all negative inputs to zero. Since half of the inputs are zeroed out on average, the variance of the output is halved. We must double the weight variance ($\frac{2}{n_{\text{in}}}$ instead of $\frac{1}{n_{\text{in}}}$) to compensate.)*
3. **Why does Batch Normalization act as a weak regularizer?** *(Because the mean and variance are calculated over mini-batches, which introduce small random fluctuations (noise) to the normalized values of each sample. This noise prevents overfitting, acting similarly to dropout.)*
4. **Why do we call `model.eval()` when evaluating a model that contains BatchNorm and Dropout?** *(Because during evaluation, we must disable dropout (keep all neurons active) and freeze BatchNorm (use running averages of mean and variance instead of batch statistics) to ensure deterministic, stable predictions.)*
