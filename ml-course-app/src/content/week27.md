# Week 28: CNNs, Part 2: Architectures & Transfer Learning

> **Goal:** Leverage advanced convolutional neural network architectures and master transfer learning. You will understand how residual connections solve vanishing gradients in very deep networks, load pretrained backbones in PyTorch, and implement fine-tuning strategies to achieve high accuracy with limited data.

---

## Part 1: Advanced CNN Architectures

As vision models grew deeper, training became more difficult. New architectural patterns emerged to solve these training bottlenecks.

### 1.1 ResNet (Residual Networks)
Standard feed-forward networks pass activations sequentially: $\mathbf{a}^{[l+1]} = g(z(\mathbf{a}^{[l]}))$. In very deep networks (e.g., 50+ layers), optimization becomes so difficult that training error begins to increase.

ResNet introduces **shortcut connections** (or **skip connections**) that bypass one or more layers:

$$ \mathbf{a}^{[l+2]} = g\left( \mathbf{z}^{[l+2]} + \mathbf{a}^{[l]} \right) $$

If the bypassed layers learn nothing (i.e., weights shrink to zero), the block simply performs an **identity mapping**: $\mathbf{a}^{[l+2]} = g(\mathbf{a}^{[l]})$. This makes it extremely easy for the optimization algorithm to pass gradients through deep layers without vanishing, allowing ResNet architectures to scale to over 1000 layers.

### 1.2 Other Key Architectures
- **Inception:** Applies kernels of different sizes ($1\times1, 3\times3, 5\times5$) and max pooling in parallel at the same layer, concatenating the outputs. This allows the network to capture features at multiple spatial scales.
- **MobileNet:** Uses **depthwise separable convolutions** (splitting standard convolutions into a spatial depthwise convolution and a channel-wise pointwise convolution), reducing computational parameters by ~90% with minimal loss in accuracy.

---

## Part 2: Transfer Learning

In practice, you rarely train a large CNN from scratch on a small dataset because it will overfit. Instead, you utilize a network trained on a massive dataset (like ImageNet) and adapt it to your target task.

### 2.1 The Transfer Learning Process
1. **Load Pretrained Backbone:** Load a model (like ResNet50) trained on ImageNet.
2. **Replace Classifier Head:** Remove the final linear layer (which outputs 1000 classes for ImageNet) and replace it with a new linear layer matching your target number of classes.
3. **Decide Freezing Strategy:**
   - **Scenario A (Very small dataset):** Freeze all weights in the backbone model (`requires_grad = False`) and train *only* the new classifier head.
   - **Scenario B (Medium dataset):** Freeze the early layers of the backbone (which extract generic features like edges) and fine-tune the deeper layers along with the new head.
   - **Scenario C (Large dataset):** Fine-tune the entire model end-to-end using a very small learning rate.

```python
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet
resnet = models.resnet18(pretrained=True)

# Freeze backbone parameters
for param in resnet.parameters():
    param.requires_grad = False

# Replace the final fully connected layer (classifier head)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2) # e.g., binary classification
```

---

## Part 3: Practice Exercises

### Exercise 1: ResNet Block from Scratch
1. Write a custom PyTorch class `ResidualBlock(nn.Module)` representing a ResNet block.
2. It should contain: Conv2d $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ Conv2d $\rightarrow$ BatchNorm2d.
3. **Code it:** Implement the forward pass such that it adds the input tensor back to the output of the second batchnorm before applying the final ReLU activation. (Note: Handle the case where the input and output channel dimensions differ using a $1\times1$ convolution shortcut).

### Exercise 2: Fine-Tuning Comparison
1. Load a pretrained MobileNet model from `torchvision.models`.
2. **Code it:** Write a training script that compares two scenarios:
   - Training only the replaced classification head (backbone frozen).
   - Fine-tuning the entire model end-to-end.
   Compare the classification accuracy on a validation dataset after 3 epochs of training.

---

## Self-Test Questions

1. **Why do residual connections prevent vanishing gradients?** *(Because during backprop, the gradient of the residual block contains a direct path: $\frac{\partial \mathbf{a}^{[l+2]}}{\partial \mathbf{a}^{[l]}} = \frac{\partial \mathbf{z}^{[l+2]}}{\partial \mathbf{a}^{[l]}} + I$. The identity matrix $I$ acts as a gradient highway, passing gradients backwards intact even if the intermediate layer gradients are zero.)*
2. **If your custom target dataset consists of grayscale x-ray scans, is it still useful to use an ImageNet-pretrained backbone?** *(Yes. The early layers of ImageNet models learn primitive spatial features like edge detectors, blobs, and textures. These low-level features are highly general and transfer effectively to medical scans, even if the overall image distributions look completely different.)*
3. **What is the mathematical benefit of using $1\times1$ convolutions?** *($1\times1$ convolutions perform linear operations across the channel dimension ($C$) while preserving spatial size ($W \times H$). They are used to pool channels, change channel dimensions (up/down-sampling channels), and add non-linearities with very low parameter counts.)*
4. **Why do we use a smaller learning rate when fine-tuning a pretrained backbone compared to training a new classifier head?** *(Because the backbone weights are already highly optimized. A large learning rate would destroy these pretrained features, causing the model to forget its general vision representations (catastrophic forgetting).)*
