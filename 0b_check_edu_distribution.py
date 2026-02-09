#!/usr/bin/env python3
"""
Check label distribution in the 20% and 80% splits.
"""

import json
from collections import Counter
from pathlib import Path

# Paths
base_dir = Path("/scratch/project_2011109/descriptors/data")
file_20 = (
    base_dir
    / "fineweb-edu-20/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)
file_80 = (
    base_dir
    / "fineweb-edu-80/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)
file_original = (
    base_dir
    / "fineweb-edu/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)


def get_label_distribution(filepath):
    """Read JSONL and return label counts."""
    labels = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            labels.append(data["label"])
    return Counter(labels)


def print_distribution(name, counter, total):
    """Pretty print label distribution."""
    print(f"\n{name}:")
    print(f"  Total: {total}")
    for label, count in sorted(counter.items()):
        pct = (count / total) * 100
        print(f"  {label:15s}: {count:6d} ({pct:5.2f}%)")


# Analyze all three files
print("Analyzing label distributions...")

original_dist = get_label_distribution(file_original)
dist_20 = get_label_distribution(file_20)
dist_80 = get_label_distribution(file_80)

original_total = sum(original_dist.values())
total_20 = sum(dist_20.values())
total_80 = sum(dist_80.values())

print_distribution("Original (100k)", original_dist, original_total)
print_distribution("20% split", dist_20, total_20)
print_distribution("80% split", dist_80, total_80)

# Sanity check
print("\n" + "=" * 50)
print("Sanity check:")
print(f"20% + 80% = {total_20 + total_80} (should be {original_total})")
assert total_20 + total_80 == original_total, "Split sizes don't add up!"
print("✓ Split sizes correct")
