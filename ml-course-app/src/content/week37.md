# Week 38: Reproduce GPT-2 (nanoGPT)

> **Goal:** Scale your from-scratch transformer to a production-level codebase. You will study the engineering optimizations in Karpathy's nanoGPT, implement binary data sharding, utilize mixed-precision training, apply gradient accumulation, and profile training throughput.

---

## Part 1: Binary Data Sharding

For large-scale language modeling, loading raw text files or tokenizing on the fly causes CPU bottlenecks. Instead, we pre-tokenize our dataset and save it as flat binary files.

- **Storage Format:** We map token strings to integer IDs using a BPE tokenizer (e.g., `tiktoken`'s GPT-2 encoding). We store these IDs in binary files using the `uint16` data type (supporting vocabulary sizes up to 65,535).
- **Memory Mapping:** During training, we read these binary files using `np.memmap`. This allows us to access segments of the dataset directly from disk without loading the entire multi-gigabyte file into RAM.

```python
import numpy as np
import tiktoken

# Raw text
text = "Pretraining transformers is highly efficient."

# Encode using GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(text)

# Convert to uint16 numpy array and save
arr = np.array(tokens, dtype=np.uint16)
with open("dataset.bin", "wb") as f:
    f.write(arr.tobytes())
```

---

## Part 2: Mixed-Precision Training

Standard PyTorch models use float32 (32-bit single-precision floating point) numbers for all weights, activations, and gradients.

**Mixed-Precision Training** calculates activations and gradients in 16-bit precision (float16 or bfloat16) while maintaining a master copy of the weights in float32 for stable updates.
- **Benefits:** Doubles calculation speed on tensor cores and halves memory usage.
- **bfloat16 vs. float16:** float16 has a narrow dynamic range, which can cause underflow (gradients shrinking to zero), requiring a `GradScaler`. bfloat16 shares the same dynamic range exponent as float32, making training highly stable without scaling.

```python
# PyTorch mixed precision example
import torch

# Instantiate scaler for float16
scaler = torch.cuda.amp.GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    # Run forward pass in mixed precision (bfloat16 or float16)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    # Scale loss and backprop
    scaler.scale(loss).backward()
    
    # Step optimizer and update scale
    scaler.step(optimizer)
    scaler.update()
```

---

## Part 3: Gradient Accumulation

GPT-2 was trained with a batch size of 524,288 tokens. If your GPU can only fit a batch size of 4,096 tokens before running out of memory (OOM), you cannot train the model stably.

**Gradient Accumulation** resolves this: we run the forward and backward passes multiple times, accumulating gradients, and call `optimizer.step()` only once every $N$ steps:

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward() # Accumulates gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Part 4: Practice Exercises

### Exercise 1: Dataset Sharding Script
1. Take a text file (e.g., a short book or article collection).
2. **Code it:** Write a Python script that tokenizes the text using `tiktoken` and splits it into `train.bin` (90%) and `val.bin` (10%) files. Print the final token counts for each file.

### Exercise 2: Training Loop with Optimizations
1. Write a training loop in PyTorch for your `MiniGPT` model from Week 31.
2. **Code it:** Integrate:
   - Mixed-precision training (`torch.cuda.amp.autocast`).
   - Gradient accumulation (accumulate over 4 steps).
   - Learning rate schedule (linear warmup followed by cosine decay).

---

## Self-Test Questions

1. **Why do we divide the loss by `accumulation_steps` when doing gradient accumulation?** *(Because gradients accumulate linearly. If we run 8 backward passes, the gradients will be 8 times larger than normal. Dividing the loss by 8 before running `backward()` ensures the final accumulated gradient magnitude matches the target batch scale.)*
2. **What is the difference between bfloat16 and float16 formats?** *(float16 uses 5 bits for the exponent and 10 bits for the fraction. bfloat16 uses 8 bits for the exponent (matching float32) and 7 bits for the fraction. bfloat16's larger exponent range prevents underflow, making training more stable.)*
3. **What does `torch.compile()` do to accelerate training?** *(It compiles PyTorch code into optimized CUDA kernels. It merges multiple element-wise operations (like ReLU following a bias add) into a single kernel (kernel fusion), reducing memory bandwidth overhead.)*
4. **Why is training throughput measured in tokens/sec instead of samples/sec?** *(Because sequence lengths can vary. A batch of 16 samples with sequence length 128 has the same computational size as a batch of 8 samples with sequence length 256. Measuring tokens/sec provides a normalized metric of GPU utilization.)*
