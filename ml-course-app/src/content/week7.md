# Week 8: Information Theory & Numerical Stability

> **Goal:** Understand Entropy, KL Divergence, and Cross-Entropy as measures of information. Then, learn the numerical tricks (like the log-sum-exp trick) that keep your neural network losses from exploding into `NaN`s.

---

## Part 1: Information Theory Essentials

Information Theory, founded by Claude Shannon, asks: *How do we quantify information?*

### 1.1 Surprise and Entropy
If a highly probable event happens (the sun rises), you get very little information. You aren't surprised. 
If a highly improbable event happens (it snows in the Sahara), you get a lot of information. You are very surprised.

The "surprise" of an event $x$ with probability $p(x)$ is measured in *bits*:
$$ \text{Surprise}(x) = -\log_2(p(x)) $$

**Entropy ($H$)** is the *expected* (average) surprise of a probability distribution $P$:
$$ H(P) = -\sum_x p(x) \log p(x) $$

- A biased coin that always lands Heads has $H = 0$. There is no uncertainty.
- A perfectly fair coin has $H = 1$ bit. Uncertainty is maximized.

### 1.2 Cross-Entropy
Suppose the true distribution of weather in your city is $P$. But you believe the distribution is $Q$.
Because you have the wrong belief, you will use a sub-optimal coding scheme to transmit weather reports. The expected number of bits you will need using your flawed belief $Q$ to encode events from the true distribution $P$ is the **Cross-Entropy**:

$$ H(P, Q) = -\sum_x p(x) \log q(x) $$

In Machine Learning:
- $P$ is the true data distribution (usually a one-hot vector, where $P(x_{true}) = 1$ and everything else is $0$).
- $Q$ is your neural network's predicted probability distribution.
- Minimizing Cross-Entropy forces $Q$ to become as close to $P$ as possible.

### 1.3 Kullback-Leibler (KL) Divergence
KL Divergence measures the *penalty* you pay for using the wrong distribution $Q$. It is exactly the difference between the Cross-Entropy and the true Entropy.

$$ D_{KL}(P || Q) = H(P, Q) - H(P) = \sum_x p(x) \log \frac{p(x)}{q(x)} $$

- $D_{KL} \geq 0$ always.
- $D_{KL} = 0$ if and only if $P = Q$.
- **It is not a true distance metric** because it is asymmetric: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$.

---

## Part 2: Numerical Stability in ML

Math on a chalkboard assumes real numbers have infinite precision. Computers use floating-point numbers with limited memory. This causes two massive headaches in ML:
1. **Overflow:** Numbers get too large and become `inf`.
2. **Underflow:** Numbers get too close to zero and become exactly `0.0`. Taking $\log(0)$ gives `-inf` or `NaN`.

### 2.1 The Softmax Catastrophe
The Softmax function turns arbitrary numbers (logits) into a probability distribution:
$$ \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

What happens if $z_1 = 1000$? The computer tries to calculate $e^{1000}$, which completely overflows `float64` precision. It returns `inf`, and $\text{inf} / \text{inf}$ evaluates to `NaN`. Your training crashes.

### 2.2 The Log-Sum-Exp Trick (Stable Softmax)
Notice that if we shift all the inputs by a constant $c$, the softmax output does not change!
$$ \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} e^{z_i}}{e^{-c} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

To prevent overflow, we simply set $c = \max(\mathbf{z})$.
Now, the largest value in the exponent will be $e^0 = 1$. Nothing can overflow.

When computing Cross-Entropy loss, we need to take the log of the softmax. 
$$ \log \sigma(\mathbf{z})_i = z_i - \max(\mathbf{z}) - \log\left(\sum_j e^{z_j - \max(\mathbf{z})}\right) $$
This calculation is called the **log-sum-exp** trick, and it is baked into `torch.nn.CrossEntropyLoss`, which is why you never pass softmax outputs directly to the loss function in PyTorch; you pass raw logits!

---

## Part 3: Practice Exercises

### Exercise 1: Information Theory Metrics
Let $P = [1.0, 0.0, 0.0]$ (A one-hot true label).
Let $Q_1 = [0.9, 0.05, 0.05]$ (A good prediction).
Let $Q_2 = [0.1, 0.8, 0.1]$ (A terrible prediction).

1. Calculate the Entropy of $P$ by hand.
2. Calculate the Cross-Entropy $H(P, Q_1)$ and $H(P, Q_2)$.
3. Because $P$ is one-hot, what do you notice about the relationship between Cross-Entropy and $-\log(q_{true})$?

### Exercise 2: Implementing Stable Softmax
Let's see overflow in action and fix it using the max trick.

```python
import numpy as np

logits = np.array([10.0, 20.0, 1000.0])

def naive_softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def stable_softmax(z):
    # Shift z by subtracting its maximum value
    shift_z = z - np.max(z)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z)

print("Naive Softmax:")
# This will likely output a warning and return array([ 0.,  0., nan])
print(naive_softmax(logits))

print("\nStable Softmax:")
# This should correctly return array([0., 0., 1.])
print(stable_softmax(logits))
```

---

## Self-Test Questions
1. **Why is KL Divergence not a true distance metric?** *(Because it is asymmetric. The penalty for using $Q$ when the truth is $P$ is not the same as using $P$ when the truth is $Q$.)*
2. **In PyTorch, `nn.CrossEntropyLoss` expects raw logits, not softmax probabilities. Why?** *(Because doing softmax followed by log is numerically unstable. PyTorch combines them into a single, fused LogSumExp operation under the hood to prevent underflow and overflow.)*
3. **If your model is perfectly confident and perfectly correct, what is the value of the Cross-Entropy Loss?** *(Zero. If $Q$ perfectly matches a one-hot $P$, then $-\log(1.0) = 0$.)*

---
> ▣ **Math block capstone (End of Part 1)**
> You have completed the theoretical foundations of Machine Learning! If you can confidently write out the matrix calculus chain rule for backpropagation, and understand how MLE leads to Cross-Entropy, you are ready for the programming block. Next week, we transition to Python mastery.
