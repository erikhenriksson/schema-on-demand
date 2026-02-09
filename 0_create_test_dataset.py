#!/usr/bin/env python3
"""
Split fineweb-edu dataset into 20% and 80% samples.
"""

import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("/scratch/project_2011109/descriptors/data")
input_file = (
    base_dir
    / "fineweb-edu/concatenated/descriptors_fineweb-edu_harmonized_labelled.jsonl"
)

output_20_dir = base_dir / "fineweb-edu-20/concatenated"
output_80_dir = base_dir / "fineweb-edu-80/concatenated"

output_20_file = output_20_dir / "descriptors_fineweb-edu_harmonized_labelled.jsonl"
output_80_file = output_80_dir / "descriptors_fineweb-edu_harmonized_labelled.jsonl"

# Create output directories
output_20_dir.mkdir(parents=True, exist_ok=True)
output_80_dir.mkdir(parents=True, exist_ok=True)

# Read all lines
print(f"Reading {input_file}...")
with open(input_file, "r") as f:
    lines = f.readlines()

total = len(lines)
print(f"Total lines: {total}")

# Shuffle and split
random.shuffle(lines)
split_idx = 20_000

lines_20 = lines[:split_idx]
lines_80 = lines[split_idx:]

# Write 20% sample
print(f"Writing {len(lines_20)} lines to {output_20_file}...")
with open(output_20_file, "w") as f:
    f.writelines(lines_20)

# Write 80% sample
print(f"Writing {len(lines_80)} lines to {output_80_file}...")
with open(output_80_file, "w") as f:
    f.writelines(lines_80)

print("Done!")
print(f"20% sample: {output_20_file} ({len(lines_20)} lines)")
print(f"80% sample: {output_80_file} ({len(lines_80)} lines)")
