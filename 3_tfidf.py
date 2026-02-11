#!/usr/bin/env python3
"""
Train Logistic Regression classifiers using TF-IDF on document text.
Simple, readable, waterfall structure.
If model already exists, skip training and just predict.
"""

import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/scratch/project_2011109/descriptors/data")
MAX_EXAMPLES_PER_CLASS = 5000
RANDOM_SEED = 42
C_VALUE = 1.0  # L2 regularization parameter

# Benchmarks to train on
BENCHMARKS = ["imdb"]

# File paths
NEGATIVE_POOL_FILE = (
    BASE_DIR
    / "fineweb-edu-20/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)
TEST_FILE = (
    BASE_DIR
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled_predicted_logreg.jsonl"
)
OUTPUT_FILE = (
    BASE_DIR
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled_predicted_tfidf.jsonl"
)
MODELS_DIR = BASE_DIR / "trained_models_tfidf"
METRICS_DIR = BASE_DIR / "metrics_tfidf"

# Create output directories
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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


def extract_texts(examples):
    """Extract document texts."""
    return [ex["document"] for ex in examples]


def extract_labels(examples, label2id):
    """Extract labels as integer array."""
    return np.array([label2id[ex["label"]] for ex in examples])


def model_exists(benchmark_name):
    """Check if trained model already exists."""
    model_file = MODELS_DIR / f"{benchmark_name}_model.pkl"
    vectorizer_file = MODELS_DIR / f"{benchmark_name}_vectorizer.pkl"
    labels_file = MODELS_DIR / f"{benchmark_name}_labels.json"
    return model_file.exists() and vectorizer_file.exists() and labels_file.exists()


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

    model_file = MODELS_DIR / f"{benchmark_name}_model.pkl"
    vectorizer_file = MODELS_DIR / f"{benchmark_name}_vectorizer.pkl"
    labels_file = MODELS_DIR / f"{benchmark_name}_labels.json"

    # Check if model already exists
    if model_exists(benchmark_name):
        print(f"\n✓ Model already exists at {model_file}")
        print("Skipping training, loading existing model for prediction...")

        # Load existing model
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Load vectorizer
        with open(vectorizer_file, "rb") as f:
            vectorizer = pickle.load(f)

        # Load label mappings
        with open(labels_file, "r") as f:
            labels_data = json.load(f)
            label2id = labels_data["label2id"]
            id2label = labels_data["id2label"]

        print(f"Loaded model with {len(vectorizer.vocabulary_)} features")
        print(f"Labels: {list(label2id.keys())}")

    else:
        print(f"\n✗ Model does not exist, will train from scratch...")

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
        print(
            f"\nSampling {n_negatives} negative examples (mean of positive classes)..."
        )

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
        id2label = {
            str(idx): label for label, idx in label2id.items()
        }  # Convert to string keys
        num_labels = len(all_labels)

        print(f"\nLabels for this classifier: {all_labels}")
        print(f"Number of labels: {num_labels}")

        # Split into train/dev/test
        print("\nSplitting into train/dev/test (70/15/15)...")
        splits = split_data(all_examples)
        print(f"  Train: {len(splits['train'])}")
        print(f"  Dev:   {len(splits['dev'])}")
        print(f"  Test:  {len(splits['test'])}")

        # Extract texts and labels
        train_texts = extract_texts(splits["train"])
        dev_texts = extract_texts(splits["dev"])
        test_texts = extract_texts(splits["test"])

        y_train = extract_labels(splits["train"], label2id)
        y_dev = extract_labels(splits["dev"], label2id)
        y_test = extract_labels(splits["test"], label2id)

        # Build TF-IDF vectorizer from training data
        print("\nBuilding TF-IDF vectorizer from training data...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        # Fit on training data and transform
        print("Fitting TF-IDF on training data...")
        X_train = vectorizer.fit_transform(train_texts)
        X_dev = vectorizer.transform(dev_texts)
        X_test = vectorizer.transform(test_texts)

        print(f"Training features shape: {X_train.shape}")
        print(f"Dev features shape: {X_dev.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

        # Train logistic regression
        print(f"\nTraining Logistic Regression (C={C_VALUE})...")
        model = LogisticRegression(
            C=C_VALUE, max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1, verbose=1
        )
        model.fit(X_train, y_train)

        # Evaluate on dev set
        print("\nEvaluating on dev set...")
        y_dev_pred = model.predict(X_dev)
        dev_acc = accuracy_score(y_dev, y_dev_pred)
        print(f"Dev accuracy: {dev_acc:.4f}")

        # Evaluate on test set
        print("\nEvaluating on test set...")
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Get classification report
        report = classification_report(
            y_test, y_test_pred, target_names=all_labels, output_dict=True
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

        # Save model and vectorizer
        print(f"\nSaving model to {model_file}")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print(f"Saving vectorizer to {vectorizer_file}")
        with open(vectorizer_file, "wb") as f:
            pickle.dump(vectorizer, f)

        print(f"Saving labels to {labels_file}")
        with open(labels_file, "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f)

    # Predict on test set (80k)
    print(f"\nPredicting labels on 80k test set...")
    test_texts_80k = extract_texts(test_data)
    X_test_80k = vectorizer.transform(test_texts_80k)
    y_pred_80k = model.predict(X_test_80k)

    # Add predictions to test data
    pred_key = f"{benchmark_name}_label_tfidf"
    for i, pred_id in enumerate(y_pred_80k):
        test_data[i][pred_key] = id2label[str(pred_id)]

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
