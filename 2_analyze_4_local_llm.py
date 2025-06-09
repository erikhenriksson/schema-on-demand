import argparse
import json
import os
import sys
from typing import List

import dspy
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
SOURCE_FILE = "data/processed/descriptors_with_explainers.jsonl"
OUTPUT_DIR = "results/shards"
BATCH_SIZE = 25
MODEL_NAME = "meta-llama/Llama-3.3-8B-Instruct"
NUM_SHARDS = 4


class DescriptorScore(BaseModel):
    descriptor: str = Field(description="The descriptor being evaluated")
    justification: str = Field(
        description="Brief justification for the score (max 50 words)"
    )
    score: int = Field(description="Educational value score from 1-5", ge=1, le=5)


class LocalLlama(dspy.LM):
    def __init__(self, model_name, device_map="auto"):
        super().__init__(model_name)
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded on devices: {self.model.hf_device_map}")

    def __call__(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return [response]


class BatchScoring(dspy.Signature):
    """Evaluate educational value of descriptors using a 5-point scoring system.

    Scoring criteria:
    1 point: Basic information relevant to education, might include non-academic content
    2 points: Addresses educational elements but doesn't align closely with standards
    3 points: Appropriate for education, introduces key concepts, coherent but may have limitations
    4 points: Highly relevant for grade school education, clear presentation, substantial content
    5 points: Outstanding educational value, perfect for primary/grade school teaching
    """

    descriptors: List[str] = dspy.InputField(
        desc="List of content descriptors to evaluate for educational value"
    )
    evaluations: List[DescriptorScore] = dspy.OutputField(
        desc="Educational scores with justifications for each descriptor"
    )


class DescriptorScorer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use Predict with structured output
        self.scorer = dspy.Predict(BatchScoring)

    def forward(self, descriptors: List[str]):
        result = self.scorer(descriptors=descriptors)
        return result.evaluations


def get_all_descriptors():
    """Get all unique descriptors"""
    descriptors = set()
    with open(SOURCE_FILE, "r") as f:
        for line in f:
            doc = json.loads(line.strip())
            for descriptor in doc["descriptors"]:
                normalized = descriptor.lower().strip()
                if normalized:
                    descriptors.add(normalized)
    return sorted(list(descriptors))


def get_shard_descriptors(all_descriptors, shard_id):
    """Get descriptors for this shard"""
    shard_size = len(all_descriptors) // NUM_SHARDS
    start_idx = shard_id * shard_size
    if shard_id == NUM_SHARDS - 1:  # Last shard gets remainder
        end_idx = len(all_descriptors)
    else:
        end_idx = (shard_id + 1) * shard_size
    return all_descriptors[start_idx:end_idx]


def get_processed_descriptors(output_file):
    """Get already processed descriptors"""
    if not os.path.exists(output_file):
        return set()

    processed = set()
    with open(output_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            processed.add(data["descriptor"])
    return processed


def process_batch(scorer, descriptors_batch):
    """Process a batch of descriptors"""
    try:
        evaluations = scorer(descriptors_batch)

        # Convert Pydantic models to dicts
        results = []
        for eval_item in evaluations:
            if isinstance(eval_item, DescriptorScore):
                results.append(
                    {
                        "descriptor": eval_item.descriptor,
                        "justification": eval_item.justification[:200],
                        "score": eval_item.score,
                    }
                )
            elif isinstance(eval_item, dict):
                results.append(
                    {
                        "descriptor": eval_item.get("descriptor", "unknown"),
                        "justification": eval_item.get(
                            "justification", "no justification"
                        )[:200],
                        "score": eval_item.get("score", 3),
                    }
                )

        return results
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_id", type=int, help="Shard ID (0-3)")
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= NUM_SHARDS:
        print(f"Shard ID must be 0-{NUM_SHARDS - 1}")
        sys.exit(1)

    print(f"Starting shard {args.shard_id}")

    # Setup model
    print("Loading local Llama model...")
    local_lm = LocalLlama(MODEL_NAME)
    dspy.settings.configure(lm=local_lm)

    # Get descriptors for this shard
    all_descriptors = get_all_descriptors()
    shard_descriptors = get_shard_descriptors(all_descriptors, args.shard_id)
    print(f"Shard {args.shard_id}: {len(shard_descriptors)} descriptors")

    # Setup output file
    output_file = f"{OUTPUT_DIR}/shard_{args.shard_id}.jsonl"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get already processed
    processed = get_processed_descriptors(output_file)
    remaining = [d for d in shard_descriptors if d not in processed]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All descriptors already processed!")
        return

    # Process in batches
    scorer = DescriptorScorer()

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} descriptors)"
        )

        results = process_batch(scorer, batch)

        if results:
            with open(output_file, "a") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            print(f"Saved {len(results)} results")
        else:
            print("No results from this batch")

    print(f"Shard {args.shard_id} complete!")


if __name__ == "__main__":
    main()
