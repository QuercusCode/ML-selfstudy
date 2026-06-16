# Week 37: Parameter-Efficient Fine-Tuning: PEFT & LoRA

> **Goal:** Fine-tune large language models on consumer-grade hardware. You will understand the memory bottlenecks of full fine-tuning, derive the mathematics of Low-Rank Adaptation (LoRA), learn how parameters are updated, and study QLoRA quantization.

---

## Part 1: The Memory Bottleneck of Full Fine-Tuning

When training or fine-tuning a model, memory is consumed by more than just the model weights. The bulk of GPU RAM is occupied by **optimizer states** and **gradients**.

For a model with $N$ parameters trained using Adam:
- **Weights:** $4N$ bytes (in float32).
- **Gradients:** $4N$ bytes.
- **Optimizer States (Adam):** Tracks the first and second moments, requiring $8N$ bytes.
- **Total:** At least $16 \times N$ bytes, plus memory for activations during the forward pass.

For a 7-billion parameter model ($7\text{B}$), full fine-tuning requires over $112\text{GB}$ of GPU VRAM, making it impossible on standard consumer GPUs.

---

## Part 2: Low-Rank Adaptation (LoRA)

LoRA (Hu et al., 2021) freezes the pretrained weights and injects trainable rank decomposition matrices into each layer of the Transformer.

### 2.1 The Mathematics of LoRA
Consider a dense linear layer with pretrained weights $W_0 \in \mathbb{R}^{d \times k}$. The forward pass is:

$$ \mathbf{h} = W_0 \mathbf{x} $$

We want to learn a weight update matrix $\Delta W \in \mathbb{R}^{d \times k}$ such that the updated output is:

$$ \mathbf{h} = (W_0 + \Delta W) \mathbf{x} = W_0 \mathbf{x} + \Delta W \mathbf{x} $$

Instead of optimizing all $d \times k$ parameters in $\Delta W$, we decompose $\Delta W$ into two low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, where the rank $r \ll \min(d, k)$:

$$ \Delta W = \frac{\alpha}{r} B A $$

where $\alpha$ is a constant scaling hyperparameter.

```
       Input x (k x 1)
         /        \
        /          \
   [ W_0 ]       [ A ] (r x k)
   (d x k)          |
    Frozen       [ B ] (d x r)
      |          Trainable
      |             |
   W_0 * x   +   BA * x
        \          /
         \        /
       Output h (d x 1)
```

- **Initialization:** $A$ is initialized from a random Gaussian distribution, and $B$ is initialized to zero. This ensures that $\Delta W = 0$ at the start of training, so the model behavior is initially identical to $W_0$.
- **Parameter Savings:** If $d = 4096, k = 4096$, and rank $r = 8$, full fine-tuning trains $16.7$ million parameters. LoRA trains only $2 \times 4096 \times 8 = 65,536$ parameters—a **99.6% reduction** in trainable parameters.

### 2.2 Zero Inference Latency
During deployment, we can add the low-rank updates directly back to the frozen weights: $W_{\text{final}} = W_0 + \frac{\alpha}{r} BA$. This merges the adapters, yielding zero additional latency during inference.

---

## Part 3: QLoRA (Quantized LoRA)

QLoRA (Dettmers et al., 2023) scales parameter efficiency further by:
1. **4-bit NormalFloat (NF4):** Quantizing the frozen weights $W_0$ to 4-bit values.
2. **Double Quantization:** Quantizing the quantization constants to save additional memory.
3. **Paged Optimizers:** Using CPU RAM as a paging buffer to prevent GPU out-of-memory spikes during long gradient computations.

With QLoRA, a $7\text{B}$ model can be fine-tuned on a single $24\text{GB}$ GPU (like an RTX 3090/4090).

---

## Part 4: Practice Exercises

### Exercise 1: LoRA Layer from Scratch
1. Write a custom PyTorch class `LoRALinear(nn.Module)` that wraps a standard `nn.Linear` layer.
2. It should contain:
   - A frozen linear layer (`weight.requires_grad = False`).
   - Two small trainable parameter matrices $A$ and $B$ corresponding to rank $r$.
3. **Code it:** Implement the forward pass: $W_0 \mathbf{x} + \frac{\alpha}{r} B A \mathbf{x}$. Print the trainable parameter count and compare it to the base layer.

### Exercise 2: Fine-Tuning with PEFT Library
1. Install `peft` and `transformers`.
2. Load a classification model.
3. **Code it:** Use `get_peft_model` from the `peft` library along with `LoraConfig` to wrap your model. Print the count of trainable parameters vs. total parameters.

---

## Self-Test Questions

1. **Why does reducing the number of trainable parameters reduce GPU memory usage if the model weights still take up space?** *(Because we do not calculate or store gradients or optimizer states for frozen weights. For frozen weights, we only store their static float values, saving massive amounts of VRAM that would otherwise be allocated to Adam's first and second moments.)*
2. **What is the significance of the $\alpha$ parameter in LoRA?** *( $\alpha$ is a scaling factor that controls the influence of the low-rank update: $\frac{\alpha}{r}$. When we change the rank $r$, scaling by $\alpha$ ensures we do not need to re-tune our learning rate.)*
3. **Why do we initialize matrix $B$ to zero in LoRA?** *(Because if $B=0$, then $BA = 0$, meaning the weight update $\Delta W$ is initially zero. The model's behavior is unchanged from the original pretrained weights, ensuring stable training starts.)*
4. **Which layers in a Transformer are typically targeted for LoRA injection?** *(We usually inject LoRA into the key, query, and value projection layers ($W_q, W_k, W_v$) of the Multi-Head Attention blocks. For maximum performance, we can also inject them into the projection layer ($W_o$) and the feedforward MLP layers.)*
