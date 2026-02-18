#!/usr/bin/env python3
"""
Retrieval analysis: how well can the descriptor model retrieve BERT-labeled
documents per class?

For each class, sweep a threshold on the descriptor probability and compute:
  - Precision: of retrieved docs, how many have the correct BERT gold label?
  - Recall: of all BERT gold docs for this class, how many are retrieved?
  - Dataset reduction: what fraction of the full dataset is retained?

Produces per-class precision-recall curves and a summary table of operating
points (e.g. recall=0.9, 0.8, 0.7).

Treats BERT labels as gold. Expects the shared output file to contain both
`{benchmark}_bert_label`, `{benchmark}_bert_probs` and
`{benchmark}_descriptor_label`, `{benchmark}_descriptor_probs`.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    BENCHMARKS,
    METRICS_DIR,
    OUTPUT_FILE,
    load_jsonl,
    seed_everything,
)

seed_everything()

ANALYSIS_DIR = METRICS_DIR / "retrieval_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Which classifier to evaluate as the retrieval proxy
PROXY_METHOD = "descriptor"  # change to "tfidf" to evaluate that one instead
GOLD_METHOD = "bert"

# Recall targets for the summary table
RECALL_TARGETS = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]

# ============================================================================
# HELPERS
# ============================================================================


def compute_pr_curve(scores, gold_labels, target_class, n_total):
    """
    Sweep threshold on scores for target_class and compute precision, recall,
    and dataset retention rate at each threshold.

    Args:
        scores: array of proxy probabilities for target_class (length n_total)
        gold_labels: array of BERT gold labels (length n_total)
        target_class: the class we're trying to retrieve
        n_total: total dataset size

    Returns:
        thresholds, precisions, recalls, retention_rates (all arrays)
    """
    # Sort by descending score
    order = np.argsort(-scores)
    sorted_gold = np.array(gold_labels)[order]
    sorted_scores = scores[order]

    # True positives: gold label matches target class
    is_positive = (sorted_gold == target_class).astype(np.float64)
    total_positives = is_positive.sum()

    if total_positives == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Cumulative TP and retrieved counts
    cum_tp = np.cumsum(is_positive)
    cum_retrieved = np.arange(1, len(is_positive) + 1, dtype=np.float64)

    precisions = cum_tp / cum_retrieved
    recalls = cum_tp / total_positives
    retention_rates = cum_retrieved / n_total
    thresholds = sorted_scores

    return thresholds, precisions, recalls, retention_rates


def find_operating_point(
    recalls, precisions, retention_rates, thresholds, target_recall
):
    """Find the first point where recall >= target_recall."""
    idx = np.searchsorted(recalls, target_recall)
    if idx >= len(recalls):
        return None
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precisions[idx]),
        "recall": float(recalls[idx]),
        "retention_rate": float(retention_rates[idx]),
        "dataset_reduction": float(1.0 - retention_rates[idx]),
        "n_retrieved": int(idx + 1),
    }


# ============================================================================
# MAIN
# ============================================================================

print("=" * 80)
print("RETRIEVAL ANALYSIS: DESCRIPTOR vs BERT (GOLD)")
print("=" * 80)

print(f"\nLoading predictions from {OUTPUT_FILE}...")
data = load_jsonl(OUTPUT_FILE)
print(f"Loaded {len(data)} examples")
n_total = len(data)

# ============================================================================

for benchmark_name in BENCHMARKS:
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {benchmark_name}")
    print("=" * 80)

    gold_key = f"{benchmark_name}_{GOLD_METHOD}_label"
    proxy_probs_key = f"{benchmark_name}_{PROXY_METHOD}_probs"

    # Validate keys exist
    if gold_key not in data[0] or proxy_probs_key not in data[0]:
        print(f"ERROR: Missing keys. Need '{gold_key}' and '{proxy_probs_key}'.")
        print(f"Available keys: {list(data[0].keys())}")
        continue

    # Extract gold labels and class list
    gold_labels = np.array([ex[gold_key] for ex in data])
    classes = sorted(set(gold_labels))

    # Skip the "negative" class — we only care about retrieving benchmark classes
    target_classes = [c for c in classes if c != "negative"]

    print(f"\nGold label distribution:")
    for cls in classes:
        count = (gold_labels == cls).sum()
        print(f"  {cls:20s}: {count:6d} ({100 * count / n_total:.1f}%)")

    print(f"\nTarget classes for retrieval: {target_classes}")

    # ------------------------------------------------------------------
    # Per-class analysis
    # ------------------------------------------------------------------

    all_results = {}
    fig, axes = plt.subplots(
        1, len(target_classes), figsize=(7 * len(target_classes), 6)
    )
    if len(target_classes) == 1:
        axes = [axes]

    for ax, target_class in zip(axes, target_classes):
        print(f"\n--- Class: {target_class} ---")

        # Extract proxy scores for this class
        scores = np.array([ex[proxy_probs_key][target_class] for ex in data])

        thresholds, precisions, recalls, retention_rates = compute_pr_curve(
            scores, gold_labels, target_class, n_total
        )

        if len(thresholds) == 0:
            print(f"  No gold examples for class '{target_class}', skipping.")
            continue

        # Find operating points
        print(f"\n  Operating points (target recall → achieved):")
        print(
            f"  {'Recall':>8s}  {'Precision':>10s}  {'Retained':>10s}  {'Reduction':>10s}  {'N Retrieved':>12s}  {'Threshold':>10s}"
        )
        print(
            f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 12}  {'-' * 10}"
        )

        class_results = []
        for target_recall in RECALL_TARGETS:
            op = find_operating_point(
                recalls, precisions, retention_rates, thresholds, target_recall
            )
            if op is None:
                print(f"  {target_recall:>8.2f}  {'N/A':>10s}")
                class_results.append(
                    {"target_recall": target_recall, "achievable": False}
                )
            else:
                print(
                    f"  {op['recall']:>8.3f}  {op['precision']:>10.3f}  "
                    f"{op['retention_rate']:>10.3f}  {op['dataset_reduction']:>10.3f}  "
                    f"{op['n_retrieved']:>12d}  {op['threshold']:>10.4f}"
                )
                class_results.append(
                    {"target_recall": target_recall, "achievable": True, **op}
                )

        all_results[target_class] = class_results

        # Plot precision-recall curve
        ax.plot(recalls, precisions, "b-", linewidth=1.5, label="Precision vs Recall")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Class: {target_class}", fontsize=14)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

        # Add retention rate on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            recalls,
            retention_rates,
            "r--",
            linewidth=1.0,
            alpha=0.7,
            label="Dataset retained",
        )
        ax2.set_ylabel("Fraction of dataset retained", color="red", fontsize=11)
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(axis="y", labelcolor="red")

        # Mark operating points
        for op_data in class_results:
            if op_data.get("achievable"):
                ax.plot(op_data["recall"], op_data["precision"], "ko", markersize=4)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    plt.suptitle(
        f"{benchmark_name}: {PROXY_METHOD} retrieval vs {GOLD_METHOD} gold labels",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save plot
    plot_file = ANALYSIS_DIR / f"{benchmark_name}_{PROXY_METHOD}_retrieval_pr.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved PR curve plot to {plot_file}")

    # Save detailed results as JSON
    results_file = (
        ANALYSIS_DIR / f"{benchmark_name}_{PROXY_METHOD}_retrieval_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(
            {
                "benchmark": benchmark_name,
                "proxy_method": PROXY_METHOD,
                "gold_method": GOLD_METHOD,
                "n_total": n_total,
                "classes": all_results,
            },
            f,
            indent=2,
        )
    print(f"Saved detailed results to {results_file}")

# ============================================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"\nAll outputs in: {ANALYSIS_DIR}")
