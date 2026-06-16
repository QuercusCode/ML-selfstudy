# Week 36: The Hugging Face Ecosystem

> **Goal:** Master the industry-standard tooling for deep learning and transformers. You will learn the Hugging Face abstractions (Pipelines, Tokenizers, AutoClasses), map data transformations using the `datasets` library, and set up a fine-tuning pipeline using the `Trainer` API.

---

## Part 1: Hugging Face Core Abstractions

Hugging Face provides modular, open-source libraries that handle the entire lifecycle of transformer modeling.

### 1.1 The Pipeline API
The simplest way to use a model for inference. It wraps pre-processing, model forward pass, and post-processing into a single object:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love studying deep learning!")
print(result) # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 1.2 AutoClasses and Tokenizers
Under the hood, inference involves two main components:
1. **Tokenizer:** Converts text into numerical tensors (`input_ids` and `attention_mask`).
2. **Model:** The neural network weights.

Hugging Face uses **AutoClasses** to automatically fetch the correct architecture based on a model's name string:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load correct tokenizer and model head
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize input
inputs = tokenizer("I love studying deep learning!", return_tensors="pt")
print(inputs) # Contains 'input_ids' and 'attention_mask' tensors
```

---

## Part 2: The `datasets` Library

The `datasets` library provides fast, memory-mapped data loading and preprocessing.

Key methods:
- **`load_dataset`:** Downloads and caches datasets from the HF Hub.
- **`map`:** Applies preprocessing functions to every row in parallel using multiprocessing, caching outputs to disk to save RAM:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")

# Preprocessing map function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Preprocess dataset in parallel
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

---

## Part 3: The Trainer API

The `Trainer` class handles PyTorch training loops, device distribution (CPU/GPU/TPU), metrics, and checkpointing.

```python
from transformers import TrainingArguments, Trainer

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs"
)

# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

---

## Part 4: Practice Exercises

### Exercise 1: Tokenizer Inspection
1. Load the `bert-base-uncased` tokenizer.
2. Tokenize the sentence: `"Pretraining transformers is extremely efficient."`
3. **Code it:** Print the output token IDs, convert them back to a list of subword strings using `tokenizer.convert_ids_to_tokens`, and identify any subwords that contain special prefixes (like `##`).

### Exercise 2: Fine-Tuning a Custom Classifier
1. Load a text classification dataset from the Hub.
2. Load a base model `AutoModelForSequenceClassification` with 2 output labels.
3. **Code it:** Write a script that preprocesses the data using `map` and configures a `Trainer` to run for 1 epoch. Compute validation accuracy after training.

---

## Self-Test Questions

1. **What is the difference between `AutoModel` and `AutoModelForSequenceClassification`?** *(`AutoModel` returns the base transformer backbone, outputting raw hidden states (embeddings) for each token. `AutoModelForSequenceClassification` appends a task-specific linear head (classifier) on top of the backbone, outputting class logits.)*
2. **Why is the `attention_mask` tensor necessary?** *(Because mini-batches require sequences to have the same length. We pad shorter sequences with placeholder tokens (like `[PAD]`). The `attention_mask` is a binary tensor (1 for real tokens, 0 for pads) that prevents the attention mechanism from attending to placeholder pad tokens.)*
3. **What is the benefit of memory-mapping in the `datasets` library?** *(Memory-mapping allows datasets to be read directly from storage/disk rather than being fully loaded into RAM. This enables developers to preprocess and train on datasets that are larger than their system's memory without running out of RAM.)*
4. **How do you save and share a fine-tuned model on Hugging Face?** *(By calling `model.save_pretrained("./path")` to save weights locally, or calling `trainer.push_to_hub()` to upload the tokenizer, model weights, and config files directly to your Hugging Face account repository.)*
