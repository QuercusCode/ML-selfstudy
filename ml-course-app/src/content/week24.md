# Week 25: Optimization, Tuning, & Backprop Ninja

> **Goal:** Bridge the gap between optimization theory and implementation. You will implement learning rate schedules, compare optimization algorithms (Momentum, RMSProp, Adam) in practice, and complete the "Backprop Ninja" exercise: manually deriving and coding the analytical gradients of a multi-layer network.

---

## Part 1: Modern Optimizers & LR Scheduling

Gradient updates require adjusting steps based on loss curvature.

### 1.1 Optimizers in Practice
- **SGD with Momentum:** Uses a moving average of past gradients to smooth out oscillations and accelerate along consistent directions:
  $$ \mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta) \nabla_\mathbf{\theta} J(\mathbf{\theta}) $$
  $$ \mathbf{\theta} \leftarrow \mathbf{\theta} - \alpha \mathbf{v}_t $$
- **RMSProp:** Scales step sizes inversely by the running average of squared gradients, dampening oscillations:
  $$ \mathbf{s}_t = \beta \mathbf{s}_{t-1} + (1-\beta) (\nabla_\mathbf{\theta} J(\mathbf{\theta}))^2 $$
  $$ \mathbf{\theta} \leftarrow \mathbf{\theta} - \frac{\alpha}{\sqrt{\mathbf{s}_t + \epsilon}} \nabla_\mathbf{\theta} J(\mathbf{\theta}) $$
- **Adam:** Combines both Momentum (1st moment) and RMSProp (2nd moment) with bias correction:
  $$ \mathbf{\theta} \leftarrow \mathbf{\theta} - \frac{\alpha}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}} \hat{\mathbf{v}}_t $$

### 1.2 Learning Rate Decay and Schedules
Starting with a high learning rate helps escape local minima, but we must decrease it as we converge to prevent overshooting the optimum.
- **Step Decay:** Multiplies the learning rate by a factor (e.g., $0.1$) every $N$ epochs.
- **Cosine Annealing:** Smoothly decreases the learning rate using a cosine curve:
  $$ \alpha_t = \alpha_{\text{min}} + \frac{1}{2}(\alpha_{\text{max}} - \alpha_{\text{min}})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right) $$

---

## Part 2: The "Backprop Ninja" Exercise

To build absolute confidence in deep learning, we must perform backpropagation manually—writing the analytical gradients of a complete network from scratch and verifying them against PyTorch's `autograd`.

Let's trace a single-layer step: $\mathbf{x} \rightarrow \mathbf{z} = \mathbf{x} W + \mathbf{b} \rightarrow \mathbf{a} = \tanh(\mathbf{z}) \rightarrow \text{loss } L$.

Given the incoming gradient from upstream layers $\mathbf{da} = \frac{\partial L}{\partial \mathbf{a}}$:

### 2.1 Backprop through Tanh
The derivative of $\tanh(z)$ is $1 - \tanh^2(z)$. Using the element-wise product ($\odot$):

$$ \mathbf{dz} = \frac{\partial L}{\partial \mathbf{z}} = \mathbf{da} \odot (1 - \mathbf{a}^2) $$

### 2.2 Backprop through Linear Layer
To find the gradients of the weights, bias, and input:

$$ \mathbf{dW} = \frac{\partial L}{\partial W} = \mathbf{x}^T \mathbf{dz} $$
$$ \mathbf{db} = \frac{\partial L}{\partial \mathbf{b}} = \sum_{\text{batch}} \mathbf{dz} $$
$$ \mathbf{dx} = \frac{\partial L}{\partial \mathbf{x}} = \mathbf{dz} W^T $$

```python
# A simple manual backprop implementation check
import torch

# Setup variables
x = torch.randn(5, 10, requires_grad=True)
W = torch.randn(10, 3, requires_grad=True)
b = torch.randn(1, 3, requires_grad=True)

# Forward pass
z = x @ W + b
a = torch.tanh(z)
# Dummy loss as sum
loss = a.sum()

# Backward pass via PyTorch
loss.backward()

# Manual calculations
da = torch.ones_like(a)                  # dLoss/da
dz = da * (1 - a**2)                     # dLoss/dz
dW = x.t() @ dz                          # dLoss/dW
db = dz.sum(0, keepdim=True)             # dLoss/db
dx = dz @ W.t()                          # dLoss/dx

# Verify values
print("Weights gradient match:", torch.allclose(W.grad, dW))
print("Inputs gradient match:", torch.allclose(x.grad, dx))
```

---

## Part 3: Practice Exercises

### Exercise 1: Cosine Decay Scheduler
1. Write a custom Python class `CosineDecay` that calculates the learning rate for a given epoch.
2. **Code it:** Integrate this scheduler into a PyTorch training loop. Plot the learning rate values over 100 epochs and verify that it matches the Cosine Annealing equation.

### Exercise 2: Backprop Ninja Challenge
1. Write an MLP containing: `Linear` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `Tanh` $\rightarrow$ `Linear` $\rightarrow$ `CrossEntropyLoss`.
2. **Code it:** Write the complete analytical backward pass from scratch (using only basic tensor operations, no `.backward()`). Log the maximum absolute difference between your manual gradients and PyTorch's gradients. It must be less than $10^{-7}$ to pass.

---

## Self-Test Questions

1. **Why does Adam use bias correction?** *(Because first and second moments ($\mathbf{v}_t, \mathbf{s}_t$) are initialized to zero. During early iterations, these moving averages are heavily biased towards zero. Bias correction divides by $(1 - \beta^t)$ to scale them up, resolving this initialization artifact.)*
2. **Under what conditions does SGD with Momentum outperform Adam?** *(In many image classification and language modeling tasks, SGD with Momentum generalizes better to unseen test data than Adam, provided the learning rate schedule is tuned carefully. Adam can converge rapidly but settle in flatter, less optimal minima.)*
3. **What is the shape of the gradient $\mathbf{dW}$ for a matrix multiplication $\mathbf{z} = \mathbf{x} W$?** *(If $\mathbf{x} \in \mathbb{R}^{m \times d}$ and $W \in \mathbb{R}^{d \times n}$, then $\mathbf{dz} \in \mathbb{R}^{m \times n}$. The gradient $\mathbf{dW}$ must have the same shape as $W$ ($d \times n$), which dictates the formula $\mathbf{dW} = \mathbf{x}^T \mathbf{dz}$.)*
4. **Why is it important to sum over the batch dimension when computing the bias gradient $\mathbf{db}$?** *(Because the bias vector $\mathbf{b}$ is broadcasted (replicated) across all $m$ samples in the batch. By the multivariate chain rule, the gradient with respect to a shared parameter is the sum of gradients across all operations that utilized it.)*
