# Week 40: Diffusion Model Fundamentals

> **Goal:** Study the generative paradigm behind modern structural biology and image generation. You will learn the mathematics of the forward diffusion process (noising), derive the reverse process (denoising), and implement a toy diffusion framework.

---

## Part 1: The Generative Diffusion Paradigm

Unlike GANs (which use a generator and discriminator game) or VAEs (which optimize a lower bound on data likelihood), **Diffusion Models** generate samples by learning to reverse a gradual noising process.

```
Forward Process (Noising):   x_0 (Data) --------> x_1 --------> x_2 --------> x_T (Pure Noise)
Reverse Process (Denoising): x_0 <-------- x_1 <-------- x_2 <-------- x_T (Pure Noise)
```

---

## Part 2: The Forward (Noising) Process

The forward process adds small amounts of Gaussian noise to a data point $\mathbf{x}_0$ over $T$ steps according to a variance schedule $\beta_1, \dots, \beta_T$:

$$ q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t \mid \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t I\right) $$

### 2.1 Direct Sampling at Any Step $t$
A critical property of the forward process is that we can sample $\mathbf{x}_t$ directly from the starting image $\mathbf{x}_0$ in one step, without calculating intermediate steps.

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Using substitution, we show:

$$ \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon} \quad \text{where} \quad \mathbf{\epsilon} \sim \mathcal{N}(0, I) $$

This allows us to write the distribution of $\mathbf{x}_t$ conditioned on $\mathbf{x}_0$:

$$ q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t \mid \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) I\right) $$

---

## Part 3: The Reverse (Denoising) Process

If we can reverse the forward process, we can start with pure Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(0, I)$ and reconstruct a sample from our target data distribution.

### 3.1 The Reverse Step
The reverse step transition $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ depends on the entire data distribution, so we approximate it using a neural network:

$$ p_\mathbf{\theta}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1} \mid \mathbf{\mu}_\mathbf{\theta}(\mathbf{x}_t, t), \mathbf{\Sigma}_\mathbf{\theta}(\mathbf{x}_t, t)\right) $$

### 3.2 Loss Function (Noise Prediction)
Instead of predicting the denoised image $\mathbf{x}_0$ directly, it is mathematically cleaner to train the neural network $\mathbf{\epsilon}_\mathbf{\theta}$ to predict the random noise vector $\mathbf{\epsilon}$ that was added to $\mathbf{x}_0$ to produce $\mathbf{x}_t$:

$$ L(\mathbf{\theta}) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{\epsilon}} \left[ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\mathbf{\theta}(\mathbf{x}_t, t) \|^2 \right] $$
$$ L(\mathbf{\theta}) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{\epsilon}} \left[ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\mathbf{\theta}\left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon}, t \right) \|^2 \right] $$

Once trained, we generate new samples by iteratively predicting noise, subtracting it, and adding scaled random noise back at each step from $T$ down to 1.

---

## Part 4: Practice Exercises

### Exercise 1: Forward Noising from Scratch
1. Write a Python function `forward_diffusion(x_0, t, beta_schedule)` that:
   - Takes a starting vector $\mathbf{x}_0$.
   - Computes $\bar{\alpha}_t$ for the specified timestep $t$.
   - Samples a random noise vector $\mathbf{\epsilon}$.
   - Outputs the noised vector $\mathbf{x}_t$ using the direct sampling formula.
2. **Code it:** Apply this to a 2D coordinates dataset representing a circle, and plot the distributions at $t \in \{0, 5, 20, 100\}$.

### Exercise 2: Implementing a U-Net Noise Predictor
1. In diffusion models, the network $\mathbf{\epsilon}_\mathbf{\theta}$ takes both the image $\mathbf{x}_t$ and the scalar timestep $t$.
2. **Code it:** Build a small PyTorch model that takes input shape $B \times C \times W \times H$ along with an integer timestep tensor $B \times 1$. Use sinusoidal positional embeddings to encode the timestep $t$ and add it to the hidden layers.

---

## Self-Test Questions

1. **Why is it easier to train a model to predict the noise $\mathbf{\epsilon}$ rather than the clean image $\mathbf{x}_0$?** *(Predicting the clean image directly requires the network to guess lost high-frequency details in a single step, which is highly non-linear. Predicting the noise aligns the network with predicting the local score function (gradients of the density), which is structurally smoother and easier to optimize.)*
2. **What role does the variance schedule $\beta_t$ play in training stability?** *( $\beta_t$ controls the noise rate. If $\beta_t$ is too large, the image degrades too quickly, and the model cannot learn the early denoising transitions. If it is too small, we need too many steps $T$ to reach pure noise, making training and sampling slow. We usually use linear or cosine schedules.)*
3. **What is the difference between DDPM and DDIM sampling?** *(DDPM (Denoising Diffusion Probabilistic Models) is stochastic and requires walking through all $T$ steps sequentially. DDIM (Denoising Diffusion Implicit Models) uses a deterministic reverse path, allowing us to skip steps and generate samples in $10 \times - 50 \times$ fewer steps without retraining the model.)*
4. **How do diffusion models relate to protein design (e.g., RFdiffusion)?** *(Instead of diffusing pixel values in an image, structural models diffuse the 3D coordinates (translations and rotations) of amino acid backbone residues. The model learns to denoise chaotic 3D coordinate clouds into valid, foldable protein structures.)*
