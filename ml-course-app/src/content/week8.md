# Week 9: Idiomatic Python

> **Goal:** Write Python the way the ML ecosystem writes it. Understanding the Python Data Model, generators, decorators, and type hints separates researchers who write scripts from engineers who build systems.

---

## Part 1: The Python Data Model (Dunder Methods)

Python objects aren't just bags of attributes. They integrate deeply with the language via "dunder" (double underscore) methods.
When you write `len(dataset)`, Python doesn't look for a `.length` property. It calls `dataset.__len__()`.
When you write `batch[0]`, Python calls `batch.__getitem__(0)`.

This is why PyTorch datasets are so clean:

```python
class MyDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        # Called by len(dataset)
        return len(self.data)
        
    def __getitem__(self, idx):
        # Called by dataset[idx]
        return self.data[idx], self.labels[idx]

# Because we implemented dunder methods, our class acts like a built-in list!
dataset = MyDataset([1.2, 3.4, 5.6], [0, 1, 0])
print(f"Dataset length: {len(dataset)}")
print(f"First item: {dataset[0]}")
```

---

## Part 2: Generators and Laziness

Machine Learning datasets are huge. If you load 1,000,000 images into a Python list, your RAM will explode and your program will crash.

Instead of building a massive list in memory, we use **Generators** to yield one item at a time.
Any function with the `yield` keyword is a generator.

```python
# BAD: Loads everything into memory at once
def get_images_bad(paths):
    images = []
    for path in paths:
        images.append(load_image(path))
    return images # RAM crashes here

# GOOD: Yields one image, pauses, and waits for the loop to ask for the next
def get_images_good(paths):
    for path in paths:
        yield load_image(path) # RAM usage stays completely flat!
```

Generators are lazy. They don't compute the next value until you ask for it using `next()` or a `for` loop. This is the foundation of `DataLoader` logic.

---

## Part 3: First-Class Functions and Decorators

In Python, functions are just objects. You can pass them as arguments, return them from other functions, and assign them to variables.

A **Decorator** is a function that takes another function, adds some functionality, and returns it. It's a clean way to wrap code (like timing, logging, or checking `torch.no_grad()`).

```python
import time

def timer_decorator(func):
    # This wrapper replaces the original function
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs) # Call the original function
        end = time.time()
        print(f"Function {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# The @ syntax is syntactic sugar for: slow_task = timer_decorator(slow_task)
@timer_decorator
def slow_task():
    time.sleep(1.5)
    return "Done!"

slow_task() # Automatically prints the timing!
```

---

## Part 4: Type Hints and Dataclasses

Python is dynamically typed, which is great for fast prototyping but terrible for debugging a 50,000-line ML framework.
**Type hints** don't enforce types at runtime, but they allow IDEs to catch bugs instantly.

```python
from typing import List, Dict, Tuple

# We immediately know what this function expects and returns!
def calculate_loss(predictions: List[float], labels: List[int]) -> float:
    pass
```

### 4.1 Dataclasses
Stop writing massive `__init__` methods just to store state. Use `@dataclass`. It automatically generates `__init__`, `__repr__`, and `__eq__` for you.

```python
from dataclasses import dataclass

@dataclass
class HyperParameters:
    learning_rate: float
    batch_size: int
    optimizer: str = "adam" # Default value

# Instantiation is clean and prints beautifully
config = HyperParameters(0.001, 32)
print(config)
```

---

## Part 5: Practice Exercises

### Exercise 1: Custom Context Manager
Context managers (the `with` statement) ensure resources are cleaned up. Files close automatically: `with open('file.txt') as f:`.
You can build your own by implementing `__enter__` and `__exit__`.

1. Write a class `TimerContext`.
2. In `__enter__`, record `self.start_time`.
3. In `__exit__`, record the end time and print the elapsed time.
4. **Code it:** Use it like this:
```python
# with TimerContext():
#     time.sleep(1)
```

### Exercise 2: Generator Pipeline
Let's simulate a data processing pipeline.
1. Write a generator `read_lines()` that yields the numbers 1 through 10.
2. Write a second generator `square_numbers(iterable)` that takes the output of `read_lines` and yields the square of each number.
3. Write a `for` loop that prints the final squared numbers. Notice how memory is never filled with a list of 10 items.

---

## Self-Test Questions
1. **What is the difference between `return` and `yield`?** *(`return` terminates the function and passes back a value. `yield` pauses the function, passes back a value, and remembers its exact state so it can resume when `next()` is called.)*
2. **Why do we use `@dataclass` instead of regular classes for configs?** *(It removes boilerplate. It auto-generates the `__init__` and string representation methods based purely on the type-hinted attributes you define.)*
3. **If you implement `__len__` and `__getitem__`, what built-in Python behaviors does your object immediately gain?** *(It can be passed to `len()`, it can be indexed like `obj[2]`, and because it can be indexed, Python can automatically iterate over it in a `for` loop!)*
