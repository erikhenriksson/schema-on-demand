#!/usr/bin/env python3
"""
Evaluate retrieval performance: Descriptor vs TF-IDF vs Random baseline.
Treat BERT predictions as gold labels.
Simple, readable, waterfall structure.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/scratch/project_2011109/descriptors/data")
RANDOM_SEED = 42

# Benchmarks to evaluate
BENCHMARKS = ["yelp", "bbc_news", "imdb"]

# File paths
TEST_FILE = (
    BASE_DIR
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled_predicted_tfidf.jsonl"
)
RESULTS_DIR = BASE_DIR / "retrieval_results"

# Create output directory
RESULTS_DIR.mkdir(exist_ok=True)

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


def get_all_classes_for_benchmark(data, benchmark_name):
    """Get all unique classes for a benchmark (excluding 'negative')."""
    gold_key = f"{benchmark_name}_label"
    classes = set()
    for ex in data:
        label = str(ex[gold_key])
        if label != "negative":
            classes.add(label)
    return sorted(list(classes))


def evaluate_retrieval(gold_labels, pred_labels, target_class):
    """
    Evaluate retrieval for a specific class.

    Returns:
        precision, recall, f1, n_retrieved, n_gold, reduction_rate
    """
    total_docs = len(gold_labels)

    # Binary classification: is it target_class or not?
    gold_binary = np.array([1 if g == target_class else 0 for g in gold_labels])
    pred_binary = np.array([1 if p == target_class else 0 for p in pred_labels])

    # Count metrics
    n_retrieved = np.sum(pred_binary)  # How many we retrieved
    n_gold = np.sum(gold_binary)  # How many actually exist

    # Calculate metrics (handle edge cases)
    if n_retrieved == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    elif n_gold == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = precision_score(gold_binary, pred_binary, zero_division=0)
        recall = recall_score(gold_binary, pred_binary, zero_division=0)
        f1 = f1_score(gold_binary, pred_binary, zero_division=0)

    # Reduction rate
    reduction_rate = 1.0 - (n_retrieved / total_docs)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_retrieved": int(n_retrieved),
        "n_gold": int(n_gold),
        "reduction_rate": float(reduction_rate),
        "n_total": int(total_docs),
    }


def create_random_predictions(gold_labels, all_classes):
    """Create random predictions with same class distribution as gold."""
    return [random.choice(all_classes) for _ in gold_labels]


# ============================================================================
# MAIN EVALUATION
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

print(f"\nLoading test set from {TEST_FILE}...")
test_data = load_jsonl(TEST_FILE)
print(f"Loaded {len(test_data)} test examples")

# ============================================================================
# EVALUATE EACH BENCHMARK
# ============================================================================

for benchmark_name in BENCHMARKS:
    print("\n" + "=" * 80)
    print(f"EVALUATING BENCHMARK: {benchmark_name}")
    print("=" * 80)

    # Keys for this benchmark
    gold_key = f"{benchmark_name}_label"
    descriptor_key = f"{benchmark_name}_label_descriptor"
    tfidf_key = f"{benchmark_name}_label_tfidf"

    # Extract labels (convert to strings)
    gold_labels = [str(ex[gold_key]) for ex in test_data]
    descriptor_labels = [str(ex[descriptor_key]) for ex in test_data]
    tfidf_labels = [str(ex[tfidf_key]) for ex in test_data]

    # Get all classes (excluding 'negative')
    all_classes = get_all_classes_for_benchmark(test_data, benchmark_name)
    print(f"\nClasses to evaluate: {all_classes}")
    print(f"Number of classes: {len(all_classes)}")

    # Get gold label distribution
    gold_counter = Counter(gold_labels)
    print(f"\nGold label distribution (BERT):")
    for cls in all_classes:
        count = gold_counter[cls]
        pct = (count / len(gold_labels)) * 100
        print(f"  {cls:20s}: {count:6d} ({pct:5.2f}%)")

    # Create random baseline predictions
    print("\nGenerating random baseline predictions...")
    random_labels = create_random_predictions(gold_labels, all_classes)

    # Results storage
    results = {
        "benchmark": benchmark_name,
        "n_total_docs": len(test_data),
        "n_classes": len(all_classes),
        "classes": all_classes,
        "gold_distribution": {cls: int(gold_counter[cls]) for cls in all_classes},
        "per_class_results": {},
    }

    # Evaluate each class
    print("\n" + "-" * 80)
    print("PER-CLASS EVALUATION")
    print("-" * 80)

    for target_class in all_classes:
        print(f"\nEvaluating class: {target_class}")

        # Evaluate descriptor-based retrieval
        descriptor_metrics = evaluate_retrieval(
            gold_labels, descriptor_labels, target_class
        )
        print(f"  Descriptor-based:")
        print(f"    Precision: {descriptor_metrics['precision']:.4f}")
        print(f"    Recall:    {descriptor_metrics['recall']:.4f}")
        print(f"    F1:        {descriptor_metrics['f1']:.4f}")
        print(
            f"    Retrieved: {descriptor_metrics['n_retrieved']} / {descriptor_metrics['n_gold']} gold"
        )
        print(f"    Reduction: {descriptor_metrics['reduction_rate'] * 100:.2f}%")

        # Evaluate TF-IDF retrieval
        tfidf_metrics = evaluate_retrieval(gold_labels, tfidf_labels, target_class)
        print(f"  TF-IDF:")
        print(f"    Precision: {tfidf_metrics['precision']:.4f}")
        print(f"    Recall:    {tfidf_metrics['recall']:.4f}")
        print(f"    F1:        {tfidf_metrics['f1']:.4f}")
        print(
            f"    Retrieved: {tfidf_metrics['n_retrieved']} / {tfidf_metrics['n_gold']} gold"
        )
        print(f"    Reduction: {tfidf_metrics['reduction_rate'] * 100:.2f}%")

        # Evaluate random baseline
        random_metrics = evaluate_retrieval(gold_labels, random_labels, target_class)
        print(f"  Random Baseline:")
        print(f"    Precision: {random_metrics['precision']:.4f}")
        print(f"    Recall:    {random_metrics['recall']:.4f}")
        print(f"    F1:        {random_metrics['f1']:.4f}")
        print(
            f"    Retrieved: {random_metrics['n_retrieved']} / {random_metrics['n_gold']} gold"
        )
        print(f"    Reduction: {random_metrics['reduction_rate'] * 100:.2f}%")

        # Store results
        results["per_class_results"][target_class] = {
            "descriptor": descriptor_metrics,
            "tfidf": tfidf_metrics,
            "random": random_metrics,
        }

    # Calculate macro-averaged metrics
    print("\n" + "-" * 80)
    print("MACRO-AVERAGED METRICS")
    print("-" * 80)

    descriptor_avg = {
        "precision": np.mean(
            [
                results["per_class_results"][cls]["descriptor"]["precision"]
                for cls in all_classes
            ]
        ),
        "recall": np.mean(
            [
                results["per_class_results"][cls]["descriptor"]["recall"]
                for cls in all_classes
            ]
        ),
        "f1": np.mean(
            [
                results["per_class_results"][cls]["descriptor"]["f1"]
                for cls in all_classes
            ]
        ),
        "reduction_rate": np.mean(
            [
                results["per_class_results"][cls]["descriptor"]["reduction_rate"]
                for cls in all_classes
            ]
        ),
    }

    tfidf_avg = {
        "precision": np.mean(
            [
                results["per_class_results"][cls]["tfidf"]["precision"]
                for cls in all_classes
            ]
        ),
        "recall": np.mean(
            [
                results["per_class_results"][cls]["tfidf"]["recall"]
                for cls in all_classes
            ]
        ),
        "f1": np.mean(
            [results["per_class_results"][cls]["tfidf"]["f1"] for cls in all_classes]
        ),
        "reduction_rate": np.mean(
            [
                results["per_class_results"][cls]["tfidf"]["reduction_rate"]
                for cls in all_classes
            ]
        ),
    }

    random_avg = {
        "precision": np.mean(
            [
                results["per_class_results"][cls]["random"]["precision"]
                for cls in all_classes
            ]
        ),
        "recall": np.mean(
            [
                results["per_class_results"][cls]["random"]["recall"]
                for cls in all_classes
            ]
        ),
        "f1": np.mean(
            [results["per_class_results"][cls]["random"]["f1"] for cls in all_classes]
        ),
        "reduction_rate": np.mean(
            [
                results["per_class_results"][cls]["random"]["reduction_rate"]
                for cls in all_classes
            ]
        ),
    }

    print(f"\nDescriptor-based (macro-avg):")
    print(f"  Precision: {descriptor_avg['precision']:.4f}")
    print(f"  Recall:    {descriptor_avg['recall']:.4f}")
    print(f"  F1:        {descriptor_avg['f1']:.4f}")
    print(f"  Reduction: {descriptor_avg['reduction_rate'] * 100:.2f}%")

    print(f"\nTF-IDF (macro-avg):")
    print(f"  Precision: {tfidf_avg['precision']:.4f}")
    print(f"  Recall:    {tfidf_avg['recall']:.4f}")
    print(f"  F1:        {tfidf_avg['f1']:.4f}")
    print(f"  Reduction: {tfidf_avg['reduction_rate'] * 100:.2f}%")

    print(f"\nRandom Baseline (macro-avg):")
    print(f"  Precision: {random_avg['precision']:.4f}")
    print(f"  Recall:    {random_avg['recall']:.4f}")
    print(f"  F1:        {random_avg['f1']:.4f}")
    print(f"  Reduction: {random_avg['reduction_rate'] * 100:.2f}%")

    # Store macro averages
    results["macro_averaged"] = {
        "descriptor": descriptor_avg,
        "tfidf": tfidf_avg,
        "random": random_avg,
    }

    # Save results
    results_file = RESULTS_DIR / f"{benchmark_name}_retrieval_results.json"
    print(f"\nSaving results to {results_file}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
print(f"\nResults saved in: {RESULTS_DIR}")
print("\nFiles created:")
for benchmark_name in BENCHMARKS:
    print(f"  {benchmark_name}_retrieval_results.json")
