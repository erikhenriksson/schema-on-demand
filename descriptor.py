#!/usr/bin/env python3
"""
Train Logistic Regression on multi-hot harmonized descriptors.
Predict labels + probabilities on the fineweb-edu-80 test set.
Writes to the shared output file.
"""

import json
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from config import (
    BENCHMARKS,
    LOGREG_C,
    METRICS_DIR,
    MODELS_DIR_LOGREG,
    NEGATIVE_POOL_FILE,
    RANDOM_SEED,
    load_jsonl,
    load_test_data,
    prepare_benchmark_data,
    print_classification_results,
    save_test_data,
    seed_everything,
)

seed_everything()

MODELS_DIR_LOGREG.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DESCRIPTOR-SPECIFIC HELPERS
# ============================================================================


def build_descriptor_vocabulary(examples):
    """Build vocabulary of all unique descriptors in training data."""
    vocab = set()
    for ex in examples:
        vocab.update(ex["harmonized_descriptors"])
    return sorted(list(vocab))


def descriptors_to_multihot(examples, vocab):
    """Convert descriptor lists to multi-hot encoded sparse vectors."""
    vocab_to_idx = {desc: idx for idx, desc in enumerate(vocab)}
    X = sp.lil_matrix((len(examples), len(vocab)), dtype=np.float32)

    for i, ex in enumerate(examples):
        for desc in ex["harmonized_descriptors"]:
            if desc in vocab_to_idx:
                X[i, vocab_to_idx[desc]] = 1.0

    return X.tocsr()


def extract_labels(examples, label2id):
    """Extract labels as integer array."""
    return np.array([label2id[ex["label"]] for ex in examples])


def model_exists(benchmark_name):
    """Check if trained model already exists."""
    model_file = MODELS_DIR_LOGREG / f"{benchmark_name}_model.pkl"
    vocab_file = MODELS_DIR_LOGREG / f"{benchmark_name}_vocab.json"
    return model_file.exists() and vocab_file.exists()


# ============================================================================
# MAIN
# ============================================================================

print("=" * 80)
print("LOGISTIC REGRESSION ON HARMONIZED DESCRIPTORS")
print("=" * 80)

print("\nLoading negative pool...")
negative_pool = load_jsonl(NEGATIVE_POOL_FILE)
print(f"Loaded {len(negative_pool)} negative examples")

print("\nLoading test set...")
test_data = load_test_data()
print(f"Loaded {len(test_data)} test examples")

# ============================================================================

for benchmark_name in BENCHMARKS:
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {benchmark_name} (LogReg Descriptors)")
    print("=" * 80)

    model_file = MODELS_DIR_LOGREG / f"{benchmark_name}_model.pkl"
    vocab_file = MODELS_DIR_LOGREG / f"{benchmark_name}_vocab.json"

    if model_exists(benchmark_name):
        print(f"\n✓ Model exists at {model_file}, loading...")

        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(vocab_file, "r") as f:
            vocab_data = json.load(f)
            vocab = vocab_data["vocab"]
            label2id = vocab_data["label2id"]
            id2label = {int(k): v for k, v in vocab_data["id2label"].items()}
            all_labels = sorted(label2id.keys())

        print(f"Loaded: {len(vocab)} descriptors, labels: {all_labels}")

    else:
        print(f"\n✗ Training from scratch...")

        splits, all_labels, label2id, id2label = prepare_benchmark_data(
            benchmark_name, negative_pool
        )

        # Build vocabulary from training data only
        print("\nBuilding descriptor vocabulary from training data...")
        vocab = build_descriptor_vocabulary(splits["train"])
        print(f"Vocabulary size: {len(vocab)}")

        # Encode
        print("\nEncoding descriptors as multi-hot sparse vectors...")
        X_train = descriptors_to_multihot(splits["train"], vocab)
        X_dev = descriptors_to_multihot(splits["dev"], vocab)
        X_test = descriptors_to_multihot(splits["test"], vocab)

        y_train = extract_labels(splits["train"], label2id)
        y_dev = extract_labels(splits["dev"], label2id)
        y_test = extract_labels(splits["test"], label2id)

        print(f"Train shape: {X_train.shape}")

        # Train
        print(f"\nTraining Logistic Regression (C={LOGREG_C})...")
        model = LogisticRegression(
            C=LOGREG_C, max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1, verbose=1
        )
        model.fit(X_train, y_train)

        # Evaluate dev
        print("\nDev set...")
        y_dev_pred = model.predict(X_dev)
        print(f"Dev accuracy: {accuracy_score(y_dev, y_dev_pred):.4f}")

        # Evaluate test
        print("\nTest set...")
        y_test_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_test_pred, target_names=all_labels, output_dict=True
        )

        metrics_file = METRICS_DIR / f"{benchmark_name}_logreg_descriptor_metrics.json"
        print(f"\nSaving metrics to {metrics_file}")
        with open(metrics_file, "w") as f:
            json.dump(report, f, indent=2)

        print_classification_results(report, all_labels)

        # Save model + vocab
        MODELS_DIR_LOGREG.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving model to {model_file}")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print(f"Saving vocabulary to {vocab_file}")
        with open(vocab_file, "w") as f:
            json.dump(
                {
                    "vocab": vocab,
                    "label2id": label2id,
                    "id2label": {str(k): v for k, v in id2label.items()},
                },
                f,
            )

    # ------------------------------------------------------------------
    # Predict on 80k test set — labels + probabilities
    # ------------------------------------------------------------------
    print(f"\nPredicting on 80k test set...")
    X_test_80k = descriptors_to_multihot(test_data, vocab)
    y_pred = model.predict(X_test_80k)
    y_probs = model.predict_proba(X_test_80k)

    label_key = f"{benchmark_name}_descriptor_label"
    probs_key = f"{benchmark_name}_descriptor_probs"
    class_names = [id2label[i] for i in range(len(id2label))]

    for i, (pred_id, probs) in enumerate(zip(y_pred, y_probs)):
        test_data[i][label_key] = id2label[int(pred_id)]
        test_data[i][probs_key] = {
            cls: round(float(probs[j]), 6) for j, cls in enumerate(class_names)
        }

    print(f"Added: '{label_key}', '{probs_key}'")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)
save_test_data(test_data)
print("LogReg descriptor classifier done.")
