# Week 29: Sequence Models: RNN, LSTM, & GRU

> **Goal:** Master recurrent neural architectures for processing sequential data (text, timeseries, biological sequences). You will understand the mathematical updates of standard RNNs, analyze why they fail over long sequences (BPTT vanishing gradients), and study the gating mechanisms of LSTMs and GRUs.

---

## Part 1: Recurrent Neural Networks (RNNs)

Feed-forward networks assume inputs are independent and have fixed sizes. **Recurrent Neural Networks (RNNs)** process inputs sequentially, passing a **hidden state** (memory) from one timestep to the next.

```
x_1 -> [ RNN Cell ] -> x_2 -> [ RNN Cell ] -> x_3 -> [ RNN Cell ]
            |                     |                     |
           h_1                   h_2                   h_3
```

### 1.1 The RNN Update Equations
At each timestep $t$, the RNN cell takes the input vector $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and computes the new hidden state $\mathbf{h}_t$:

$$ \mathbf{h}_t = \tanh\left( W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h \right) $$

The output at timestep $t$ is calculated from the hidden state:

$$ \mathbf{y}_t = W_{hy} \mathbf{h}_t + \mathbf{b}_y $$

---

## Part 2: The Vanishing Gradient Problem in RNNs

To train an RNN, we backpropagate gradients from the final timestep back through time. This is called **Backpropagation Through Time (BPTT)**.

### 2.1 Mathematical Bottleneck
The derivative of the loss at timestep $T$ with respect to the hidden state at timestep $t$ involves a chain of products:

$$ \frac{\partial L_T}{\partial \mathbf{h}_t} = \frac{\partial L_T}{\partial \mathbf{h}_T} \prod_{k=t+1}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} $$

Using the hidden state equation, the Jacobian matrix of the transition is:

$$ \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \operatorname{diag}\left( 1 - \mathbf{h}_k^2 \right) W_{hh}^T $$

If the weights in $W_{hh}$ are small, multiplying them repeatedly causes the gradient to shrink exponentially as $T - t$ grows. As a result, standard RNNs cannot learn dependencies that span more than 10–20 timesteps (they forget distant past information).

---

## Part 3: LSTMs (Long Short-Term Memory)

LSTMs solve vanishing gradients by introducing a **cell state** ($\mathbf{C}_t$) that acts as a linear gradient highway, regulated by three non-linear **gates**.

```
Input x_t  -------\
                   \
Prev Hidden h_t-1 --> [ GATES: Forget, Input, Output ]
                        |        |        |
Prev Cell C_t-1 --------+--------+--------+--------> New Cell C_t ---> New Hidden h_t
```

### 3.1 Gating Math
Let $\sigma(z) = \frac{1}{1 + e^{-z}}$ represent the sigmoid function (output range $[0, 1]$).
1. **Forget Gate:** Decides what fraction of the old cell state to discard:
   $$ \mathbf{f}_t = \sigma\left( W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f \right) $$
2. **Input Gate:** Decides which new values to write into the cell state:
   $$ \mathbf{i}_t = \sigma\left( W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i \right) $$
3. **Candidate State:** Generates new candidate values to add to the cell state:
   $$ \tilde{\mathbf{C}}_t = \tanh\left( W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c \right) $$
4. **Cell State Update:** Computes the new cell state using element-wise multiplication ($\odot$):
   $$ \mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t $$
5. **Output Gate:** Decides what fraction of the cell state to output to the hidden state:
   $$ \mathbf{o}_t = \sigma\left( W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o \right) $$
   $$ \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) $$

### 3.2 Why LSTMs Prevent Vanishing Gradients
The update for the cell state is linear: $\mathbf{C}_t = \mathbf{C}_{t-1} + \text{terms}$. If the forget gate $\mathbf{f}_t$ is close to 1, the gradient $\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}}$ is exactly 1. Gradients can flow backwards indefinitely along the cell state without shrinking exponentially.

---

## Part 4: Practice Exercises

### Exercise 1: LSTM Cell from Scratch
1. Write a custom PyTorch module `LSTMCell(input_dim, hidden_dim)` from scratch.
2. Define the weight matrices and bias vectors for the four internal gates.
3. **Code it:** Implement the forward pass equations from Part 3.1. Pass a dummy input tensor and verify that the output shapes match expected dimensions.

### Exercise 2: Character-Level Sequence Predictor
1. Prepare a text dataset (e.g., a few sentences or amino acid sequences).
2. Build an LSTM sequence predictor in PyTorch using `nn.LSTM`.
3. **Code it:** Write a script that trains the LSTM to predict the next character in the sequence, and print sample predictions at epoch 1, 10, and 50.

---

## Self-Test Questions

1. **What is the difference between the cell state ($C_t$) and the hidden state ($h_t$) in an LSTM?** *(The cell state is a linear memory vector that acts as a gradient highway, carrying long-term dependencies with minimal alteration. The hidden state is a filtered, non-linear projection of the cell state, used as the active output and passed to the next step's gating calculations.)*
2. **How does the Gated Recurrent Unit (GRU) simplify the LSTM architecture?** *(A GRU merges the cell state and hidden state into a single hidden state ($h_t$) and combines the forget and input gates into a single "update gate," reducing parameter counts and making calculations faster.)*
3. **What is gradient clipping, and why do we need it for RNN training?** *(Gradient clipping rescales gradients if their norm exceeds a specified threshold. We need it because although LSTMs prevent vanishing gradients, they are still susceptible to exploding gradients when sequence inputs are highly anomalous, causing optimization to diverge.)*
4. **Why are recurrent models difficult to parallelize on GPUs?** *(Because recurrent calculations are sequential: you cannot compute timestep $t$ until you have finished computing the hidden states for all previous timesteps $1, \dots, t-1$. This sequential dependency prevents parallel execution across time dimensions.)*
