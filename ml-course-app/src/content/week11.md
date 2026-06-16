# Week 12: Software Engineering for ML

> **Goal:** Stop living entirely in single Jupyter notebooks. Notebooks are great for exploration, but terrible for reproducible, scalable Machine Learning. It is time to learn environments, packaging, and testing—the scaffolding that keeps Deep Learning sane.

---

## Part 1: The Problem with Notebooks

Jupyter notebooks hold hidden state. If you run cell 4, then cell 2, then cell 5, the variables in memory are in an order that cannot be reproduced by simply running top-to-bottom.
When you share a notebook, it almost never runs on someone else's machine because they don't have the exact same package versions installed.

**The Solution:** Move core logic into `.py` files. Use notebooks *only* for importing that logic and plotting the results.

---

## Part 2: Environments and Dependencies

Never install packages globally on your computer. Always use a virtual environment.

### 2.1 The Virtual Environment
A virtual environment is simply an isolated folder that contains a specific Python executable and its own `site-packages` directory.

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate it (Mac/Linux)
source .venv/bin/activate
```

### 2.2 Managing Dependencies (`pyproject.toml`)
Historically, Python used `requirements.txt`. Today, the standard is the `pyproject.toml` file. It tells the ecosystem exactly what your project needs, how to build it, and what tools to use.

```toml
[project]
name = "my_ml_project"
version = "0.1.0"
dependencies = [
    "numpy>=1.24.0",
    "torch>=2.0.0",
    "pandas",
]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

Once this file exists, you can run `pip install -e .` to install your own project in "editable" mode. Now you can `import my_ml_project` from anywhere on your computer!

---

## Part 3: Project Structure

A professional ML repository doesn't have 50 `.py` files dumped in the root directory. It uses a structured layout, typically the `src/` layout.

```text
my_ml_project/
├── .venv/                  # Virtual environment (ignored by git)
├── data/                   # Raw data files (ignored by git)
├── notebooks/              # Exploration notebooks
├── src/                    # The actual Python package
│   └── my_ml_project/      
│       ├── __init__.py     
│       ├── data_loader.py  # Data ingestion logic
│       ├── model.py        # PyTorch model definitions
│       └── train.py        # The training loop
├── tests/                  # Unit tests
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Part 4: Testing with `pytest`

Machine Learning code fails silently. If you accidentally transpose a matrix, PyTorch might broadcast it incorrectly instead of crashing, and your model will simply learn garbage.
You must write unit tests.

`pytest` is the standard Python testing framework.

```python
# Save this in tests/test_model.py
import torch
from my_ml_project.model import SimpleMLP

def test_mlp_output_shape():
    # Setup
    model = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=2)
    dummy_batch = torch.randn(32, 10) # Batch of 32
    
    # Execution
    output = model(dummy_batch)
    
    # Assertion
    assert output.shape == (32, 2), f"Expected (32, 2), got {output.shape}"
```

To run all tests, simply type `pytest` in the terminal.

---

## Part 5: Experiment Tracking

If you run a training loop 50 times with different learning rates and architectures, you will forget which hyperparameters produced the best model.
Stop writing hyperparams down in a text file. Use an experiment tracker like **Weights & Biases (wandb)** or **MLflow**.

```python
import wandb

# 1. Initialize the run
wandb.init(
    project="my-first-model",
    config={
        "learning_rate": 0.001,
        "architecture": "MLP",
        "epochs": 10
    }
)

# 2. Inside your training loop, log metrics
for epoch in range(10):
    loss = train_epoch()
    accuracy = evaluate()
    
    # This automatically uploads a live dashboard to the cloud!
    wandb.log({"loss": loss, "accuracy": accuracy})

wandb.finish()
```

---

## Part 6: Practice Exercises

### Exercise 1: Scaffold a Project
1. Open your terminal and create a new directory `test_ml_project`.
2. Inside it, create the `src/` layout shown in Part 3.
3. Create a `pyproject.toml` file with `numpy` as a dependency.
4. Create a virtual environment, activate it, and run `pip install -e .`.

### Exercise 2: Write a Silent-Failure Test
Write a test to catch a silent broadcasting bug.
1. Create a function `add_bias(features, bias)` that returns `features + bias`.
2. Assume `features` is shape `(32, 10)` and `bias` is shape `(10,)`.
3. If someone accidentally passes a `bias` of shape `(32,)`, NumPy will broadcast it in unexpected ways instead of crashing!
4. **Code it:** Write a `pytest` test that asserts `add_bias` raises a `ValueError` if the shape of `bias` does not match the second dimension of `features`.

---

## Self-Test Questions
1. **Why should you put your code in a `src/` directory instead of the root folder?** *(It forces Python to test the *installed* version of your package rather than the raw files in the root directory, preventing bugs where your code works locally but fails when someone else installs it.)*
2. **What does `pip install -e .` do?** *(It installs the package in the current directory into your virtual environment in "editable" mode. If you change the code, you don't have to reinstall it for the changes to take effect.)*
3. **Why do we use an experiment tracker like `wandb` instead of just printing the loss to the console?** *(Because the console output is lost when you close the terminal. A tracker saves the exact hyperparameters, code version, and loss curves so every experiment is perfectly reproducible.)*

---
> ▣ **Foundations block exit criteria (End of Part II)**
> You're ready for Block B (Classical ML) when you can, from a blank page:
> 1. Explain backprop as the chain rule.
> 2. Reconstruct PCA via SVD.
> 3. Write a stable softmax from memory.
> 4. Vectorize a loop with broadcasting/einsum.
> 5. Scaffold a tested Python package from scratch.
