# Week 21: Build Autograd from Scratch & PyTorch Basics

> **Goal:** Deconstruct automatic differentiation by building a scalar-valued autograd engine (similar to Karpathy's micrograd) from scratch. You will understand how reverse-mode autodiff walks a computational graph using the chain rule, and transition to PyTorch's tensor-level autograd model.

---

## Part 1: The Computational Graph and Reverse-Mode Autodiff

In deep learning, neural networks are large mathematical expressions. We represent these expressions as **computational graphs**, where:
- **Nodes** represent variables (inputs, weights, biases, outputs).
- **Edges** represent mathematical operations (addition, multiplication, activations).

### 1.1 Reverse-Mode Automatic Differentiation
To train a network, we need the gradient of the loss with respect to every weight and bias. Instead of deriving these symbolically, **reverse-mode autodiff** computes gradients in two passes:
1. **Forward Pass:** Compute the output values of each node from the inputs.
2. **Backward Pass:** Starting at the final output node (the loss), walk backwards through the graph, applying the chain rule to compute the local gradient of the loss with respect to each node.

For a node $z = f(x, y)$, if we know the gradient of the loss $L$ with respect to the output $z$ ($\frac{\partial L}{\partial z}$), we can compute the gradients with respect to the inputs:

$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial x} \quad \text{and} \quad \frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial y} $$

---

## Part 2: Building a Scalar Autograd Engine (micrograd-style)

Let's build a tiny class `Value` that wraps a scalar number, tracks its operations, and calculates gradients.

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0          # derivative of the output with respect to this value
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  # function to compute gradient for this node's inputs

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
        out._backward = _backward
        return out
```

### 2.1 Implementing the Backward Pass (Topological Sort)
To compute the backward pass, we must ensure that we compute the gradient of a node only *after* we have computed the gradients of all its children in the graph. We achieve this using a **topological sort** (depth-first search):

```python
    def backward(self):
        # Build topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Go backward, setting the base derivative dL/dL = 1
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

---

## Part 3: PyTorch Autograd Basics

PyTorch generalizes this scalar computational graph to multi-dimensional arrays called **Tensors**.

### 3.1 Tensors and Gradients
When we create a tensor in PyTorch and set `requires_grad=True`, PyTorch starts tracking operations on it to build a dynamic computational graph:

```python
import torch

# Create tensors
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass: y = w * x + b
y = w * x + b

# Backward pass
y.backward()

# Print gradients dy/dw and dy/dx
print("dy/dw:", w.grad.item())  # Should be x = 3.0
print("dy/dx:", x.grad.item())  # Should be w = 2.0
```

---

## Part 4: Practice Exercises

### Exercise 1: Extend the Value Class
Using the `Value` class template:
1. Implement the subtraction (`__sub__`) and negation (`__neg__`) dunder methods.
2. Implement the power method (`__pow__`) supporting float powers ($x^n$, allowing division as $x \cdot y^{-1}$).
3. **Code it:** Write a test script verifying that your gradients for $f(a, b) = a^3 - 2b^2$ match values calculated using standard calculus equations.

### Exercise 2: Verify Against PyTorch
1. Write an equation $y = a \cdot b + \text{ReLU}(a - c)$.
2. Calculate the gradients of $y$ with respect to $a$, $b$, and $c$ using your custom `Value` engine.
3. **Code it:** Build the same graph in PyTorch using `torch.tensor` and verify that both the output and gradients match to 6 decimal places.

---

## Self-Test Questions

1. **What is a topological sort, and why is it necessary for the backward pass?** *(A topological sort orders nodes such that for every directed edge $u \rightarrow v$, node $u$ comes before $v$. In the backward pass, we must evaluate derivatives in reverse topological order so we never compute a node's gradient before accounting for all branches that depend on it.)*
2. **Why do we use `self.grad += ...` instead of `self.grad = ...` in the `_backward` functions?** *(Because if a variable is used multiple times in the graph (e.g., $y = x + x$), its gradients accumulate. By multivariate calculus, the total derivative is the sum of derivatives along all paths. Accumulating with `+=` correctly handles this dependency.)*
3. **What does `requires_grad=True` tell PyTorch?** *(It instructs PyTorch to allocate memory to store gradients for this tensor, and to track all operations involving it in the dynamic computational graph.)*
4. **What is the difference between static and dynamic computational graphs?** *(Static graphs (e.g., old TensorFlow) are defined and compiled once, then executed repeatedly with data feeds. Dynamic graphs (e.g., PyTorch) are constructed on the fly during the forward pass of each execution, allowing you to use standard Python control flow like `if` and `for` loops inside the network.)*
