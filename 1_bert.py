#!/usr/bin/env python3
"""
Train BERT classifiers on benchmarks and predict labels on fineweb-edu-80.
Simple, readable, waterfall structure.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/scratch/project_2011109/descriptors/data")
MODEL_NAME = "answerdotai/ModernBERT-large"
MAX_EXAMPLES_PER_CLASS = 5000
RANDOM_SEED = 42

# Benchmarks to train on
BENCHMARKS = ["imdb", "yelp", "bbc_news", "ag_news", "nemotron-cc-mix"]

# File paths
NEGATIVE_POOL_FILE = (
    BASE_DIR
    / "fineweb-edu-20/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)
TEST_FILE = (
    BASE_DIR
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)
OUTPUT_FILE = (
    BASE_DIR
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled_predicted.jsonl"
)
MODELS_DIR = BASE_DIR / "trained_models"
METRICS_DIR = BASE_DIR / "metrics"

# Create output directories
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_jsonl(filepath):
    """Load JSONL file into list of dicts."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """Save list of dicts to JSONL file."""
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_benchmark_data(benchmark_name):
    """Load positive examples from a benchmark."""
    filepath = (
        BASE_DIR
        / benchmark_name
        / "concatenated"
        / f"descriptors_{benchmark_name}_harmonized_labelled.jsonl"
    )
    data = load_jsonl(filepath)
    # Convert all labels to strings for consistency
    for ex in data:
        ex["label"] = str(ex["label"])
    return data


def sample_negatives(negative_pool, n_samples):
    """Sample n_samples from negative pool."""
    return random.sample(negative_pool, min(n_samples, len(negative_pool)))


def cap_examples_per_class(examples, max_per_class):
    """Cap number of examples per class label."""
    capped = []
    label_tracker = Counter()

    random.shuffle(examples)
    for ex in examples:
        label = ex["label"]
        if label_tracker[label] < max_per_class:
            capped.append(ex)
            label_tracker[label] += 1

    return capped


def split_data(examples, train_ratio=0.7, dev_ratio=0.15):
    """Split data into train/dev/test."""
    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))

    return {
        "train": examples[:train_end],
        "dev": examples[train_end:dev_end],
        "test": examples[dev_end:],
    }


def prepare_dataset(examples, tokenizer, label2id):
    """Convert examples to HuggingFace Dataset."""
    texts = [ex["document"] for ex in examples]
    labels = [label2id[ex["label"]] for ex in examples]

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

    dataset_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }

    return Dataset.from_dict(dataset_dict)


def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load negative pool (20k held-out)
print(f"\nLoading negative pool from {NEGATIVE_POOL_FILE}...")
negative_pool = load_jsonl(NEGATIVE_POOL_FILE)
print(f"Loaded {len(negative_pool)} negative examples")

# Load test set (80k)
print(f"\nLoading test set from {TEST_FILE}...")
test_data = load_jsonl(TEST_FILE)
print(f"Loaded {len(test_data)} test examples")

# ============================================================================
# TRAIN CLASSIFIERS FOR EACH BENCHMARK
# ============================================================================

for benchmark_name in BENCHMARKS:
    print("\n" + "=" * 80)
    print(f"PROCESSING BENCHMARK: {benchmark_name}")
    print("=" * 80)

    # Load positive examples
    print(f"\nLoading positive examples for {benchmark_name}...")
    positive_examples = load_benchmark_data(benchmark_name)
    print(f"Loaded {len(positive_examples)} positive examples")

    # Get class distribution
    label_counts = Counter([ex["label"] for ex in positive_examples])
    print(f"\nOriginal class distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # Cap examples per class
    print(f"\nCapping to max {MAX_EXAMPLES_PER_CLASS} per class...")
    positive_examples = cap_examples_per_class(
        positive_examples, MAX_EXAMPLES_PER_CLASS
    )
    label_counts = Counter([ex["label"] for ex in positive_examples])
    print(f"After capping:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # Calculate number of negatives needed (mean of positive classes)
    mean_positive = int(np.mean(list(label_counts.values())))
    n_negatives = min(mean_positive, MAX_EXAMPLES_PER_CLASS)
    print(f"\nSampling {n_negatives} negative examples (mean of positive classes)...")

    # Sample negatives and label them
    negative_examples = sample_negatives(negative_pool, n_negatives)
    for ex in negative_examples:
        ex["label"] = "negative"

    # Combine positive and negative
    all_examples = positive_examples + negative_examples
    print(f"Total training examples: {len(all_examples)}")

    # Get all unique labels
    all_labels = sorted(list(set([ex["label"] for ex in all_examples])))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(all_labels)

    print(f"\nLabels for this classifier: {all_labels}")
    print(f"Number of labels: {num_labels}")

    # Split into train/dev/test
    print("\nSplitting into train/dev/test (70/15/15)...")
    splits = split_data(all_examples)
    print(f"  Train: {len(splits['train'])}")
    print(f"  Dev:   {len(splits['dev'])}")
    print(f"  Test:  {len(splits['test'])}")

    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )

    # Prepare datasets
    print("\nTokenizing datasets...")
    train_dataset = prepare_dataset(splits["train"], tokenizer, label2id)
    dev_dataset = prepare_dataset(splits["dev"], tokenizer, label2id)
    test_dataset = prepare_dataset(splits["test"], tokenizer, label2id)

    # Training arguments
    output_dir = MODELS_DIR / benchmark_name
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=RANDOM_SEED,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    print("\nTraining model...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = trainer.predict(test_dataset)
    test_preds = test_predictions.predictions.argmax(-1)
    test_labels = test_predictions.label_ids

    # Get classification report
    report = classification_report(
        test_labels, test_preds, target_names=all_labels, output_dict=True
    )

    # Save metrics
    metrics_file = METRICS_DIR / f"{benchmark_name}_metrics.json"
    print(f"\nSaving metrics to {metrics_file}")
    with open(metrics_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\nTest Set Results:")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print("\nPer-class F1 scores:")
    for label in all_labels:
        if label in report:
            print(f"  {label:20s}: {report[label]['f1-score']:.4f}")

    # Save final model
    final_model_dir = MODELS_DIR / f"{benchmark_name}_final"
    print(f"\nSaving final model to {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # Predict on test set (80k)
    print(f"\nPredicting labels on 80k test set...")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare test data for prediction
    test_texts = [ex["document"] for ex in test_data]

    # Batch prediction
    batch_size = 32
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i : i + batch_size]
            batch_encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}

            outputs = model(**batch_encodings)
            predictions = outputs.logits.argmax(-1).cpu().numpy()
            all_predictions.extend(predictions)

    # Add predictions to test data
    pred_key = f"{benchmark_name}_label"
    for i, pred_id in enumerate(all_predictions):
        test_data[i][pred_key] = id2label[pred_id]

    print(f"Added predictions under key: '{pred_key}'")

# ============================================================================
# SAVE UPDATED TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

print(f"\nSaving updated test set to {OUTPUT_FILE}")
save_jsonl(test_data, OUTPUT_FILE)

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
print(f"\nUpdated test set: {OUTPUT_FILE}")
print(f"Models saved in: {MODELS_DIR}")
print(f"Metrics saved in: {METRICS_DIR}")
