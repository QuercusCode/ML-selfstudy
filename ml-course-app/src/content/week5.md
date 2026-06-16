# Week 6: Probability, Distributions, and Bayes

> **Goal:** Fluency with probability as the language of Machine Learning. Nearly every ML model is secretly a probabilistic model trying to manage uncertainty.

---

## Part 1: Probability vs Likelihood

The terms "probability" and "likelihood" are used interchangeably in casual conversation, but in statistics, they are fundamentally different concepts.

- **Probability:** You know the true parameters of the model (e.g., a coin has $P(Heads) = 0.5$). You want to predict the *data* (what is the chance of getting 3 heads in a row?). 
  *Parameters are fixed, Data is variable.*
- **Likelihood:** You have observed the *data* (e.g., you flipped 3 heads in a row). You want to evaluate different *parameters* (is this a fair coin, or is $P(Heads) = 0.9$?).
  *Data is fixed, Parameters are variable.*

Machine Learning is almost entirely about **Likelihood** (and maximizing it).

---

## Part 2: Key Probability Distributions

### 2.1 The Bernoulli and Binomial Distributions
If an event has two outcomes (1/0, Heads/Tails) with probability $p$, a single trial is a **Bernoulli** distribution.
- Mean: $p$
- Variance: $p(1-p)$

If you run $n$ independent Bernoulli trials and count the number of successes $k$, that is a **Binomial** distribution.
$$ P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} $$

### 2.2 The Gaussian (Normal) Distribution
The absolute king of distributions. Due to the Central Limit Theorem, the sum of many independent random variables tends toward a Gaussian, no matter what their original distribution was.
For a scalar variable $x$ with mean $\mu$ and variance $\sigma^2$:

$$ \mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) $$

**Multivariate Gaussian:**
For a vector $\mathbf{x} \in \mathbb{R}^k$, mean vector $\boldsymbol{\mu}$, and $k \times k$ covariance matrix $\Sigma$:

$$ \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right) $$

Notice the core exponent: $(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$. This is the **Mahalanobis Distance**—it measures how far $\mathbf{x}$ is from the mean, scaled by the covariance.

---

## Part 3: Bayes' Theorem

Bayes' Theorem is the mathematical formula for updating your beliefs based on new evidence.

$$ P(H | E) = \frac{P(E | H) P(H)}{P(E)} $$

Where:
- $H$ is the Hypothesis (or the parameters $\theta$)
- $E$ is the Evidence (or the Data $D$)

Rewritten for Machine Learning:
$$ P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)} $$

- **$P(\theta)$ (The Prior):** Your belief about the parameters *before* seeing the data. (e.g., "I think most coins are fair, so $P(\theta=0.5)$ is high.")
- **$P(D | \theta)$ (The Likelihood):** How probable is this exact data if the parameters were $\theta$?
- **$P(D)$ (The Evidence / Marginal Likelihood):** The total probability of seeing this data under *all possible* hypotheses. This is usually impossibly hard to compute, but it acts merely as a normalizing constant.
- **$P(\theta | D)$ (The Posterior):** Your updated belief about the parameters *after* seeing the data.

### 3.1 The Diagnostic Test Example
This is the canonical intuition-builder.
- A disease affects 1% of the population: Prior $P(Disease) = 0.01$.
- A test is 90% accurate at detecting the disease (Sensitivity): $P(+ | Disease) = 0.90$.
- The test has a 9% false positive rate: $P(+ | No Disease) = 0.09$.

You test positive. Do you have the disease?
We want the Posterior: $P(Disease | +)$.

$$ P(Disease | +) = \frac{P(+ | Disease) P(Disease)}{P(+)} $$
$$ P(+) = P(+ | Disease)P(Disease) + P(+ | No Disease)P(No Disease) $$
$$ P(+) = (0.90 \times 0.01) + (0.09 \times 0.99) = 0.009 + 0.0891 = 0.0981 $$

$$ P(Disease | +) = \frac{0.009}{0.0981} \approx 0.0917 $$

Even with a "90% accurate" test, you only have a **9.17%** chance of actually having the disease! This is the power of the Prior (the base rate). Because the disease is rare, the sheer volume of healthy people generates more false positives than true positives.

---

## Part 4: Practice Exercises

### Exercise 1: Bayes in Practice (Code)
Let's verify the diagnostic test mathematically using a simulation of 100,000 people.

```python
import numpy as np

n_people = 100000

# 1. Simulate the true disease state (1% prevalence)
has_disease = np.random.rand(n_people) < 0.01

# 2. Simulate the test results
# If they have the disease, 90% test positive
# If they don't, 9% test positive
test_positive = np.zeros(n_people, dtype=bool)

# For those with disease
test_positive[has_disease] = np.random.rand(has_disease.sum()) < 0.90
# For those without disease
test_positive[~has_disease] = np.random.rand((~has_disease).sum()) < 0.09

# 3. Calculate the empirical posterior
# Out of everyone who tested positive, how many actually have the disease?
total_positives = test_positive.sum()
true_positives = (has_disease & test_positive).sum()

posterior = true_positives / total_positives
print(f"Total positive tests: {total_positives}")
print(f"Actually have disease: {true_positives}")
print(f"P(Disease | Positive) = {posterior:.4f}")
# Should be very close to 0.0917!
```

### Exercise 2: Properties of Expectation and Variance
Let $X$ and $Y$ be independent random variables. $E[X] = 5$, $Var(X) = 2$. $E[Y] = 3$, $Var(Y) = 4$.
1. What is $E[2X + 3Y]$?
2. What is $Var(2X + 3Y)$? *(Hint: Variance scales quadratically, $Var(aX) = a^2 Var(X)$).*
3. What happens to $Var(X + Y)$ if $X$ and $Y$ are NOT independent but are positively correlated?

### Exercise 3: The Law of Large Numbers
The expected value of a fair die is 3.5. 
1. **Code it:** Simulate rolling a die. Compute the running average of the rolls. Plot the average as the number of rolls goes from 1 to 10,000. It should converge exactly to 3.5.

---

## Self-Test Questions
1. **What is the difference between Probability and Likelihood?** *(Probability fixes the parameters and predicts the data. Likelihood fixes the data and evaluates different parameters.)*
2. **In Bayes' Theorem, what is the 'Prior'?** *(Your belief about the distribution of parameters before seeing any new data. Often based on historical base rates.)*
3. **If you flip a coin 10 times and get 10 heads, the Maximum Likelihood Estimate says $P(Heads) = 1.0$. How does a Bayesian approach prevent this extreme conclusion?** *(By incorporating a Prior. A strong prior belief that coins are fair will overwhelm the small sample size of 10 flips, pulling the posterior probability back towards 0.5.)*
