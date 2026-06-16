# Week 27: Convolutional Networks, Part 1

> **Goal:** Transition from fully connected layers to spatial convolutional layers. You will learn the mechanics of 2D convolutions, stride, and padding, understand how pooling layers introduce spatial invariance, calculate output shapes and parameter counts, and build a vision network in PyTorch.

---

## Part 1: Why Convolutions?

A standard Multi-Layer Perceptron (MLP) flattens 2D images into 1D vectors. This approach suffers from two major limitations:

1. **Parameter Explosion:** If an input image is size $1000 \times 1000 \times 3$ (3 megapixels), a single hidden layer with 1000 units would require $3 \times 10^9$ weights—an impossible memory footprint.
2. **Loss of Spatial Structure:** Flattening an image discards the spatial relationships between neighboring pixels. 

**Convolutional Neural Networks (CNNs)** solve these issues using two core principles:
- **Local Connectivity:** Each neuron connects to only a small local region of the input (defined by the kernel size).
- **Shared Weights:** The same set of weights (the kernel) slides across the entire image to detect features (e.g., edges) regardless of where they appear, introducing **translation invariance**.

---

## Part 2: Convolutions, Padding, and Stride

A convolutional layer applies a set of learnable filters (kernels) to the input.

### 2.1 The Convolution Operation
A 2D kernel $K \in \mathbb{R}^{f \times f}$ slides across an input image $X$. At each position, it computes the sum of element-wise products:

$$ Y_{ij} = \sum_{a=0}^{f-1} \sum_{b=0}^{f-1} X_{i+a, j+b} K_{ab} $$

### 2.2 Padding ($P$)
Applying convolutions directly shrinks the spatial dimensions at each layer, and pixels near the edges are rarely sampled compared to the center. To prevent this, we add a border of zero-pixels around the input.
- **Valid Convolution:** No padding ($P=0$). The output shrinks.
- **Same Convolution:** Padding is chosen such that the output shape matches the input shape:
  $$ P = \frac{f - 1}{2} \quad (\text{for odd kernel sizes } f) $$

### 2.3 Stride ($S$)
Stride defines the step size the kernel takes when sliding across the input. A stride of $S=2$ skips every other pixel, downsampling the output by a factor of 2.

### 2.4 The Output Dimension Formula
Given an input shape $W \times H$, a kernel size $K$, padding $P$, and stride $S$, the output dimension is:

$$ O = \lfloor \frac{W - K + 2P}{S} \rfloor + 1 $$

```python
# Output calculation helper
def get_conv_output_shape(W, K, P, S):
    return ((W - K + 2*P) // S) + 1
```

---

## Part 3: Pooling Layers

Pooling layers downsample the feature maps to reduce computational cost and introduce translation invariance.

- **Max Pooling:** Slides a window across the input and outputs the maximum value in that window. It extracts the most prominent features.
- **Average Pooling:** Outputs the average value in the window.

Unlike convolutional layers, pooling layers contain **no learnable parameters**. They only downsample the input.

```
Max Pooling (2x2 filter, stride 2)
[ 1  3 ] -> [ 3 ]
[ 2  0 ]
```

---

## Part 4: Practice Exercises

### Exercise 1: 2D Convolution from Scratch
1. Write a Python function `convolve2d(image, kernel, padding=0, stride=1)` from scratch using NumPy.
2. Do not use any deep learning libraries. Use nested loops for the sliding window.
3. **Code it:** Apply an edge-detection Sobel filter:
   $$ K = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} $$
   to a toy grayscale image and print the output.

### Exercise 2: Building a PyTorch CNN
1. Define a PyTorch CNN class `SimpleCNN` for CIFAR-10 (input shape $3 \times 32 \times 32$):
   - Conv2d: 3 channels $\rightarrow$ 16 channels, kernel size 3, padding 1, stride 1.
   - ReLU.
   - MaxPool2d: kernel size 2, stride 2.
   - Conv2d: 16 channels $\rightarrow$ 32 channels, kernel size 3, padding 1, stride 1.
   - ReLU.
   - MaxPool2d: kernel size 2, stride 2.
   - Linear: output 10 classes.
2. **Code it:** Instantiate the model, calculate the exact shape of the tensor before it enters the final `Linear` layer, and print the total number of learnable parameters.

---

## Self-Test Questions

1. **Why do convolutions require fewer parameters than fully connected layers?** *(Due to weight sharing and local connectivity. In a layer with 32 filters of size $3 \times 3 \times 3$, there are only $32 \times 27 = 864$ learnable weights, regardless of the input image size. A fully connected layer's weight count scales with the product of input and output dimensions.)*
2. **If an input is $64 \times 64 \times 3$, and we apply a $5 \times 5$ conv layer with 16 filters, stride 1, and no padding, what is the shape of the output volume?** *(Applying the formula: $O = \frac{64 - 5 + 0}{1} + 1 = 60$. The output volume shape is $60 \times 60 \times 16$.)*
3. **What is translation invariance, and how do convolutions and pooling achieve it?** *(Translation invariance means that if a feature appears in a different part of the image, the network can still identify it. Convolutions slide the same weights over the entire image, extracting features anywhere. Max pooling outputs the maximum local activation, meaning small shifts in the input do not alter the pooled output value.)*
4. **Why are odd kernel sizes ($3 \times 3, 5 \times 5$) preferred over even ones ($2 \times 2, 4 \times 4$)?** *(Odd kernel sizes have a clear "center pixel," which allows symmetric padding on all sides ($P = \frac{K-1}{2}$). Even kernels require asymmetric padding, which can introduce directional bias in the feature maps.)*
