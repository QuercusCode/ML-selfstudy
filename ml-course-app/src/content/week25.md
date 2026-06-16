# Week 26: Structuring ML Projects & Experiment Tracking

> **Goal:** Develop the strategic decision-making skills of a senior machine learning practitioner. You will learn how to structure datasets when training and evaluation distributions differ, construct systematic error analysis frameworks, and configure automated hyperparameter sweeps.

---

## Part 1: Dataset Splits and Distribution Mismatch

In classical machine learning, we often split datasets using a 70/30 or 80/20 ratio. In deep learning, where datasets frequently contain millions of samples, our splitting strategies must adjust.

### 1.1 Modern Splitting Ratios
If you have 1,000,000 samples, a 1% dev set (10,000 samples) and a 1% test set (10,000 samples) are large enough to yield high-precision performance estimates.
- **Modern Split:** 98% Train / 1% Dev / 1% Test.

### 1.2 Distribution Mismatch
Suppose you want to build a cell-segmentation app. You have:
- **Source Data (Web):** 100,000 high-quality images scraped from online medical articles.
- **Target Data (App):** 5,000 low-resolution, noisy images taken by users on mobile microscopes.

You must not mix these distributions randomly. If you do, your dev/test sets will contain web-scraped images, meaning you will optimize your model for web images rather than user app images.
- **Rule:** Make sure your dev set and test set come from the **target distribution** (mobile microscope images).

### 1.3 Diagnosing Mismatch with Train-Dev Splits
If your model performs poorly on the dev set, is it due to overfitting (high variance) or distribution mismatch? To find out, split a small portion of your training data into a **Train-Dev set**:

| Metric | Scenario A | Scenario B |
|---|---|---|
| **Train Error** | 1.0% | 1.0% |
| **Train-Dev Error** | 1.5% | 8.0% (High Variance) |
| **Dev Error** | 8.0% (Data Mismatch) | 8.5% |

In Scenario A, the model performs well on training-distribution data (both Train and Train-Dev) but fails on Dev, confirming a **distribution mismatch**.

---

## Part 2: Systematic Error Analysis

When a model fails to meet target performance, do not immediately start adding more layers or changing loss functions. Perform a **systematic error analysis** first.

### 2.1 The Error Analysis Framework
1. Collect a subset of validation examples that the model misclassified (e.g., 100 examples).
2. Manually inspect each example and categorize the errors in a table:

| Image ID | True Class | Predicted Class | Blurry | Low Contrast | Incorrect Label | Background Noise |
|---|---|---|:---:|:---:|:---:|:---:|
| 001 | Cat | Dog | ✓ | | | |
| 002 | Dog | Cat | | | ✓ | |
| 003 | Cat | Bird | ✓ | ✓ | | |
| **% of Errors**| | | **55%** | **15%** | **5%** | **10%** |

3. In this example, 55% of the errors are due to blurry images. Improving the model's robustness to blur (e.g., via data augmentation) is the most efficient path forward.

---

## Part 3: Hyperparameter Sweeps (wandb)

Tuning deep learning models involves searching a large space of hyperparameters (learning rate, batch size, dropout rate, weight decay).

### 3.1 Sweep Search Methods
- **Grid Search:** Evaluates every combination. Slow and scales poorly.
- **Random Search:** Samples configurations randomly. Highly effective because some parameters (like learning rate) are much more important than others (like dropout).
- **Bayesian Optimization:** Trains a Gaussian Process surrogate model to predict final accuracy based on parameters, choosing the next configuration to evaluate based on expected improvement.

```yaml
# A sample Weights & Biases (wandb) sweep configuration
program: train.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [32, 64, 128]
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
```

---

## Part 4: Practice Exercises

### Exercise 1: Error Analysis Tool
1. Train a classifier on MNIST.
2. **Code it:** Write a function that finds the top 10 misclassified images (highest loss on incorrect predictions). Plot these 10 images in a grid, displaying their true label, predicted label, and the prediction probabilities.

### Exercise 2: Mocking a Sweep
1. Write a training script `train.py` that takes command-line arguments `--lr` and `--batch_size`, trains a simple PyTorch model for 3 epochs, and prints the validation loss.
2. **Code it:** Write a search loop that runs the script 5 times, choosing parameters using a random search. Keep track of the parameters and output values, and print the best configuration found.

---

## Self-Test Questions

1. **Why must the dev set and test set come from the exact same distribution?** *(Because the dev set is your target target. If they differ, you will spend weeks optimizing your model to score well on the dev set, only to find it fails on the test set because you optimized for the wrong objective.)*
2. **What is the purpose of a Train-Dev split?** *(To isolate variance (overfitting to the training set) from distribution mismatch. If performance on the Train-Dev split is poor, the model has high variance. If Train-Dev is good but Dev is poor, the model is suffering from distribution mismatch.)*
3. **If you find that 10% of your dev set examples are labeled incorrectly, should you fix them?** *(It depends on how much it affects your metrics. If your model's total error is 15%, fixing a 10% labeling error will not change your conclusions. If your model's error is 1.2%, then 10% incorrect labels (0.12% absolute) represent a significant fraction of your remaining errors, making it critical to clean the labels.)*
4. **Why is Random Search generally preferred over Grid Search for hyperparameter tuning?** *(Because models are typically highly sensitive to a few key parameters (e.g., learning rate) and insensitive to others. Grid search wastes time evaluating redundant values of insensitive parameters, whereas Random Search explores unique values of all parameters.)*
