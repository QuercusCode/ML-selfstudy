# Week 42: Capstone Project & Continuous Growth

> **Goal:** Complete the curriculum by executing an original capstone project and establishing long-term habits for continuous growth in machine learning. You will learn how to scope projects, set baseline benchmarks, perform robustness checks, and build a framework for ongoing learning.

---

## Part 1: Scoping Your Capstone Project

The capstone is your opportunity to apply everything you've learned to a problem of your own choosing, ideally one that integrates with your active research or career goals.

### 1.1 Project Structure
A successful machine learning project follows five developmental phases:

```
1. Scope Problem & Data --> 2. Establish Baseline --> 3. Iterative Optimization --> 4. Robustness Checks --> 5. Document & Present
```

1. **Problem Definition:** Clearly state what you are predicting and why. Define your success metric (e.g., F1-score $> 0.85$ or RMSE $< 0.1$).
2. **Data Sourcing:** Locate or compile your dataset. Ensure you have at least a few hundred samples (for tabular/probing) or thousands (for deep learning).
3. **Establishing a Baseline:** Always start by training the simplest possible model (e.g., a dummy classifier, a linear regression, or a simple decision tree). A complex model is only useful if it beats this baseline significantly.
4. **Iterative Optimization:** Optimize your model systematically using:
   - Feature engineering and preprocessing pipelines.
   - Validation diagnostics (learning curves, error analysis tables).
   - Automated hyperparameter sweeps (wandb).
5. **Robustness Checks:** Evaluate your final model on a completely held-out test set. Perform sanity checks (e.g., verifying that the model does not predict coordinates that violate physical laws).

---

## Part 2: Long-Term Growth Habits

Completing this curriculum is only the beginning of your machine learning journey. The field evolves rapidly, and maintaining your skills requires establishing three durable habits:

### 2.1 One Paper Per Week
Read one machine learning paper deeply each week. 
- **Where to find papers:** Lil'Log (excellent deep-dives), The Annotated Transformer (papers as code), and the OpenBioML Discord community (for biological ML).
- **How to read:** Don't just scan the abstract. Trace the mathematical formulations, map them to the repository code, and understand the core architectural trade-offs.

### 2.2 One Project Per Quarter
Every three months, build something from scratch.
- Reproduce a figure from a newly published paper.
- Contribute to open-source libraries (e.g., Hugging Face, PyTorch, or community repositories).
- Write a tool to automate a tedious task in your lab or workflow.

### 2.3 One Domain Challenge Per Year
Participate in at least one competitive ML challenge annually (e.g., Kaggle competitions, CASP for protein structure, or benchmark challenges). This keeps your data cleaning, pipeline engineering, and validation discipline sharp against real-world, messy datasets.

---

## Part 3: Practice Exercises

### Exercise 1: Capstone Proposal
**Code it (in markdown):** Write a one-page proposal for your capstone project. Include:
1. The biological or practical problem you want to solve.
2. The dataset source (URL or collection plan).
3. Your proposed baseline model and final candidate model.
4. Your target evaluation metrics.

### Exercise 2: Baseline Comparison Script
1. Load a dataset.
2. **Code it:** Write a script that trains a dummy baseline model (e.g., `DummyClassifier` predicting the most frequent class) and your trained candidate model. Print the classification report for both and verify that your model outperforms random chance.

---

## Self-Test Questions

1. **Why is establishing a simple baseline model the first step in any machine learning project?** *(Because without a baseline, you cannot tell if the complexity of a deep model is actually providing value. If a simple linear regression achieves an $R^2$ of 0.82, and a massive neural network takes days to train to reach 0.83, the added complexity and cost are likely not justified.)*
2. **What does a "robustness check" mean in biological machine learning?** *(It means verifying that your model's predictions conform to biological reality. For example, if a model designs a protein backbone, a robustness check verifies that bond lengths and angles fall within physically allowed ranges (Ramachandran plots), rather than just score metrics.)*
3. **How do you keep up with new architectures without getting overwhelmed?** *(By focusing on foundations. Most "new" models are combinations of existing patterns (attention, message-passing, diffusion, residual connections). If you master these core building blocks, you can understand new papers in a few minutes by identifying which blocks they recombined.)*
4. **Why is writing clean, modular code as important as model accuracy?** *(Because research code must be reproducible. If your code is a disorganized collection of notebooks, other scientists cannot verify your findings, and you will struggle to adapt the model to new datasets or deploy it in production.)*
