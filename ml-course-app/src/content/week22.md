# Week 23: Deep Networks & the PyTorch Training Loop

> **Goal:** Transition from raw calculations to writing structured, modular PyTorch code. You will learn PyTorch's `nn.Module` architecture, implement a standard data pipeline using `Dataset` and `DataLoader`, build the canonical training/evaluation loops, and track your experiments using Weights & Biases (wandb).

---

## Part 1: PyTorch's Modular API (`nn.Module`)

In PyTorch, we define neural networks by subclassing `nn.Module`. This class automatically tracks all learnable parameters (weights and biases) added as attributes, allowing us to query and optimize them easily.

```python
import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Define the forward pass flow
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

---

## Part 2: Data Loading — `Dataset` and `DataLoader`

To train models efficiently on large datasets, we must load data in parallel batches rather than loading the entire dataset into memory.

- **`nn.utils.data.Dataset`:** An abstract class representing a dataset. We must implement:
  - `__len__(self)`: Returns the total number of samples.
  - `__getitem__(self, idx)`: Returns the sample and target at index `idx`.
- **`nn.utils.data.DataLoader`:** Wraps a `Dataset` and provides an iterator over batches, handling shuffling, batching, and parallel multiprocessing (`num_workers`).

```python
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## Part 3: The Canonical Training and Evaluation Loops

Training a model follows a strict sequence of steps repeated over several epochs.

```
For each epoch:
  Set model to train mode -> model.train()
  For each batch of (inputs, targets) in DataLoader:
    1. Zero gradients -> optimizer.zero_grad()
    2. Forward pass -> outputs = model(inputs)
    3. Calculate loss -> loss = criterion(outputs, targets)
    4. Backward pass -> loss.backward()
    5. Update parameters -> optimizer.step()
```

### 3.1 Why `optimizer.zero_grad()` is Required
By default, PyTorch **accumulates gradients** on every backward pass (adds them to existing gradients rather than overwriting them). If you do not call `zero_grad()` at the start of each step, the gradients from previous steps will accumulate, ruining the optimization.

### 3.2 The Evaluation Mode
Before validating or testing:
1. Call `model.eval()`. This disables layers that behave differently during training vs. inference (e.g., BatchNorm, Dropout).
2. Wrap predictions inside a `with torch.no_grad():` block. This prevents PyTorch from calculating gradients and building a computational graph, saving massive amounts of memory and speed.

---

## Part 4: Practice Exercises

### Exercise 1: Standard Training Loop Implementation
1. Generate a synthetic dataset with 10 features and 1,000 samples.
2. Build a 3-layer MLP classifier in PyTorch.
3. **Code it:** Write a complete training function from scratch that runs for 20 epochs, prints the training loss at each epoch, and correctly handles gradient zeroing and parameter updates.

### Exercise 2: Implementing Evaluation and Metrics
1. Split your synthetic dataset into training and validation sets.
2. **Code it:** Write an evaluation loop that runs after each training epoch. Compute and print the validation loss and accuracy. Make sure to call `model.eval()` and use the `no_grad` context manager.

---

## Self-Test Questions

1. **Why does PyTorch default to accumulating gradients?** *(Accumulating gradients makes it easy to implement virtual batch sizes. If your GPU memory can only fit 8 samples but you want a batch size of 32, you can run the forward/backward passes 4 times, accumulating gradients, and call `optimizer.step()` and `optimizer.zero_grad()` only once every 4 steps.)*
2. **What is the difference between `model.train()` / `model.eval()` and `torch.no_grad()`?** *(`model.train()` and `model.eval()` toggle model configurations for modules like Dropout and BatchNorm. `torch.no_grad()` is a context manager that disables autograd calculation entirely, saving memory regardless of the model mode.)*
3. **If your model's training loss decreases but validation loss increases, what is happening?** *(The model is overfitting to the training set. It is memorizing noise in the training data rather than generalizing, causing its performance on unseen validation data to degrade.)*
4. **What does `num_workers > 0` do in a PyTorch DataLoader?** *(It enables multi-process data loading. Instead of the main thread loading batches sequentially (blocking GPU calculation), background CPU worker processes fetch and queue batches in parallel, preventing data loading bottlenecks.)*
