#!/usr/bin/env python3
"""
Train Logistic Regression on TF-IDF features from raw document text.
Predict labels + probabilities on the fineweb-edu-80 test set.
Writes to the shared output file.
"""

import os

os.environ["WANDB_DISABLED"] = "true"
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from config import (
    BENCHMARKS,
    LOGREG_C,
    METRICS_DIR,
    MODELS_DIR_TFIDF,
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

MODELS_DIR_TFIDF.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TFIDF-SPECIFIC HELPERS
# ============================================================================


def extract_labels(examples, label2id):
    """Extract labels as integer array."""
    return np.array([label2id[ex["label"]] for ex in examples])


def model_exists(benchmark_name):
    """Check if trained model already exists."""
    model_file = MODELS_DIR_TFIDF / f"{benchmark_name}_model.pkl"
    vectorizer_file = MODELS_DIR_TFIDF / f"{benchmark_name}_tfidf.pkl"
    return model_file.exists() and vectorizer_file.exists()


# ============================================================================
# MAIN
# ============================================================================

print("=" * 80)
print("LOGISTIC REGRESSION ON TF-IDF FEATURES")
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
    print(f"BENCHMARK: {benchmark_name} (LogReg TF-IDF)")
    print("=" * 80)

    model_file = MODELS_DIR_TFIDF / f"{benchmark_name}_model.pkl"
    vectorizer_file = MODELS_DIR_TFIDF / f"{benchmark_name}_tfidf.pkl"
    labels_file = MODELS_DIR_TFIDF / f"{benchmark_name}_labels.json"

    if model_exists(benchmark_name):
        print(f"\n✓ Model exists at {model_file}, loading...")

        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_file, "rb") as f:
            vectorizer = pickle.load(f)
        with open(labels_file, "r") as f:
            label_data = json.load(f)
            label2id = label_data["label2id"]
            id2label = {int(k): v for k, v in label_data["id2label"].items()}
            all_labels = sorted(label2id.keys())

        print(f"Loaded: vocab size {len(vectorizer.vocabulary_)}, labels: {all_labels}")

    else:
        print(f"\n✗ Training from scratch...")

        splits, all_labels, label2id, id2label = prepare_benchmark_data(
            benchmark_name, negative_pool
        )

        # Fit TF-IDF on training data
        print("\nFitting TF-IDF vectorizer on training data...")
        vectorizer = TfidfVectorizer(
            max_features=50000, sublinear_tf=True, min_df=2, ngram_range=(1, 2)
        )
        X_train = vectorizer.fit_transform([ex["document"] for ex in splits["train"]])
        X_dev = vectorizer.transform([ex["document"] for ex in splits["dev"]])
        X_test = vectorizer.transform([ex["document"] for ex in splits["test"]])

        y_train = extract_labels(splits["train"], label2id)
        y_dev = extract_labels(splits["dev"], label2id)
        y_test = extract_labels(splits["test"], label2id)

        print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
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

        metrics_file = METRICS_DIR / f"{benchmark_name}_logreg_tfidf_metrics.json"
        print(f"\nSaving metrics to {metrics_file}")
        with open(metrics_file, "w") as f:
            json.dump(report, f, indent=2)

        print_classification_results(report, all_labels)

        # Save model + vectorizer + labels
        MODELS_DIR_TFIDF.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving model to {model_file}")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print(f"Saving vectorizer to {vectorizer_file}")
        with open(vectorizer_file, "wb") as f:
            pickle.dump(vectorizer, f)

        print(f"Saving label mappings to {labels_file}")
        with open(labels_file, "w") as f:
            json.dump(
                {
                    "label2id": label2id,
                    "id2label": {str(k): v for k, v in id2label.items()},
                },
                f,
            )

    # ------------------------------------------------------------------
    # Predict on 80k test set — labels + probabilities
    # ------------------------------------------------------------------
    print(f"\nPredicting on 80k test set...")
    X_test_80k = vectorizer.transform([ex["document"] for ex in test_data])
    y_pred = model.predict(X_test_80k)
    y_probs = model.predict_proba(X_test_80k)

    label_key = f"{benchmark_name}_tfidf_label"
    probs_key = f"{benchmark_name}_tfidf_probs"
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
print("LogReg TF-IDF classifier done.")
