#!/usr/bin/env python3
"""
Shared configuration and utilities for descriptor classifiers.
All paths, constants, data loading, and common operations live here.
"""

import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path("/scratch/project_2011109/descriptors/data")

# Input files
NEGATIVE_POOL_FILE = (
    BASE_DIR
    / "fineweb-edu-20/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)
TEST_FILE = (
    BASE_DIR
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)

# Single output file — all classifiers write predictions here
OUTPUT_FILE = (
    BASE_DIR
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled_predicted.jsonl"
)

# Model and metric directories (per classifier type)
MODELS_DIR_BERT = BASE_DIR / "trained_models" / "bert"
MODELS_DIR_LOGREG = BASE_DIR / "trained_models" / "logreg_descriptor"
MODELS_DIR_TFIDF = BASE_DIR / "trained_models" / "logreg_tfidf"
METRICS_DIR = BASE_DIR / "metrics"

# ============================================================================
# CONSTANTS
# ============================================================================

BERT_MODEL_NAME = "chandar-lab/NeoBERT"
MAX_EXAMPLES_PER_CLASS = 5000
RANDOM_SEED = 42
LOGREG_C = 1.0
TRAIN_RATIO = 0.7
DEV_RATIO = 0.15

# Benchmarks to train on
BENCHMARKS = ["yelp", "imdb"]

# ============================================================================
# SEEDING
# ============================================================================


def seed_everything(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# ============================================================================
# DATA I/O
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
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
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
    for ex in data:
        ex["label"] = str(ex["label"])
    return data


def load_test_data():
    """
    Load the test set. If the output file already exists (from a previous
    classifier run), load that instead so we accumulate predictions.
    """
    if OUTPUT_FILE.exists():
        print(f"  Loading existing predictions from {OUTPUT_FILE}")
        return load_jsonl(OUTPUT_FILE)
    else:
        print(f"  Loading base test set from {TEST_FILE}")
        return load_jsonl(TEST_FILE)


def save_test_data(data):
    """Save the test set to the single output file."""
    save_jsonl(data, OUTPUT_FILE)
    print(f"Saved predictions to {OUTPUT_FILE}")


# ============================================================================
# DATA PREPARATION
# ============================================================================


def sample_negatives(negative_pool, n_samples):
    """Sample n_samples from negative pool."""
    return random.sample(negative_pool, min(n_samples, len(negative_pool)))


def cap_examples_per_class(examples, max_per_class=MAX_EXAMPLES_PER_CLASS):
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


def split_data(examples, train_ratio=TRAIN_RATIO, dev_ratio=DEV_RATIO):
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


def prepare_benchmark_data(benchmark_name, negative_pool):
    """
    Full data preparation pipeline for a benchmark:
    load positives, cap, sample negatives, combine, build label maps, split.

    Returns: splits dict, all_labels list, label2id, id2label
    """
    # Load positive examples
    print(f"\nLoading positive examples for {benchmark_name}...")
    positive_examples = load_benchmark_data(benchmark_name)
    print(f"Loaded {len(positive_examples)} positive examples")

    # Class distribution
    label_counts = Counter([ex["label"] for ex in positive_examples])
    print(f"\nOriginal class distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # Cap
    print(f"\nCapping to max {MAX_EXAMPLES_PER_CLASS} per class...")
    positive_examples = cap_examples_per_class(positive_examples)
    label_counts = Counter([ex["label"] for ex in positive_examples])
    print(f"After capping:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # Sample negatives
    mean_positive = int(np.mean(list(label_counts.values())))
    n_negatives = min(mean_positive, MAX_EXAMPLES_PER_CLASS)
    print(f"\nSampling {n_negatives} negative examples (mean of positive classes)...")

    negative_examples = [
        copy.deepcopy(ex) for ex in sample_negatives(negative_pool, n_negatives)
    ]
    for ex in negative_examples:
        ex["label"] = "negative"

    # Combine
    all_examples = positive_examples + negative_examples
    print(f"Total examples: {len(all_examples)}")

    # Label maps
    all_labels = sorted(list(set([ex["label"] for ex in all_examples])))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    print(f"\nLabels: {all_labels} ({len(all_labels)} classes)")

    # Split
    print(
        f"\nSplitting into train/dev/test ({TRAIN_RATIO}/{DEV_RATIO}/{1 - TRAIN_RATIO - DEV_RATIO})..."
    )
    splits = split_data(all_examples)
    print(f"  Train: {len(splits['train'])}")
    print(f"  Dev:   {len(splits['dev'])}")
    print(f"  Test:  {len(splits['test'])}")

    return splits, all_labels, label2id, id2label


def print_classification_results(report, all_labels, set_name="Test"):
    """Print classification results summary."""
    print(f"\n{set_name} Set Results:")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"\nPer-class F1 scores:")
    for label in all_labels:
        if label in report:
            print(f"  {label:20s}: {report[label]['f1-score']:.4f}")
