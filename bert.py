#!/usr/bin/env python3
"""
Train a BERT (NeoBERT) classifier on benchmarks and predict labels + probabilities
on the fineweb-edu-80 test set. Writes to the shared output file.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    BENCHMARKS,
    BERT_MODEL_NAME,
    METRICS_DIR,
    MODELS_DIR_BERT,
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

MODELS_DIR_BERT.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# BERT-SPECIFIC HELPERS
# ============================================================================


def prepare_dataset(examples, tokenizer, label2id):
    """Convert examples to HuggingFace Dataset."""
    texts = [ex["document"] for ex in examples]
    labels = [label2id[ex["label"]] for ex in examples]

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}


def model_exists(benchmark_name):
    """Check if trained model already exists."""
    final_model_dir = MODELS_DIR_BERT / f"{benchmark_name}_final"
    return final_model_dir.exists() and (final_model_dir / "config.json").exists()


# ============================================================================
# MAIN
# ============================================================================

print("=" * 80)
print("BERT CLASSIFIER")
print("=" * 80)

# Load shared data
print("\nLoading negative pool...")
negative_pool = load_jsonl(NEGATIVE_POOL_FILE)
print(f"Loaded {len(negative_pool)} negative examples")

print("\nLoading test set...")
test_data = load_test_data()
print(f"Loaded {len(test_data)} test examples")

# ============================================================================

for benchmark_name in BENCHMARKS:
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {benchmark_name} (BERT)")
    print("=" * 80)

    final_model_dir = MODELS_DIR_BERT / f"{benchmark_name}_final"

    if model_exists(benchmark_name):
        print(f"\n✓ Model exists at {final_model_dir}, loading...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(final_model_dir), trust_remote_code=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(final_model_dir), trust_remote_code=True
        )
        id2label = model.config.id2label
        label2id = model.config.label2id
        all_labels = sorted(label2id.keys())
        print(f"Labels: {all_labels}")

    else:
        print(f"\n✗ Training from scratch...")

        splits, all_labels, label2id, id2label = prepare_benchmark_data(
            benchmark_name, negative_pool
        )

        # Load tokenizer and model
        print(f"\nLoading {BERT_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(
            BERT_MODEL_NAME, trust_remote_code=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=len(all_labels),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
        )

        # Tokenize
        print("\nTokenizing datasets...")
        train_dataset = prepare_dataset(splits["train"], tokenizer, label2id)
        dev_dataset = prepare_dataset(splits["dev"], tokenizer, label2id)
        test_dataset = prepare_dataset(splits["test"], tokenizer, label2id)

        # Training
        output_dir = MODELS_DIR_BERT / benchmark_name
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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("\nTraining...")
        trainer.train()

        # Evaluate
        print("\nEvaluating on test set...")
        test_predictions = trainer.predict(test_dataset)
        test_preds = test_predictions.predictions.argmax(-1)
        test_labels = test_predictions.label_ids

        report = classification_report(
            test_labels, test_preds, target_names=all_labels, output_dict=True
        )

        metrics_file = METRICS_DIR / f"{benchmark_name}_bert_metrics.json"
        print(f"\nSaving metrics to {metrics_file}")
        with open(metrics_file, "w") as f:
            json.dump(report, f, indent=2)

        print_classification_results(report, all_labels)

        # Save model
        print(f"\nSaving model to {final_model_dir}")
        final_model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

    # ------------------------------------------------------------------
    # Predict on 80k test set — labels + probabilities
    # ------------------------------------------------------------------
    print(f"\nPredicting on 80k test set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_texts = [ex["document"] for ex in test_data]
    batch_size = 32
    all_pred_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i : i + batch_size]
            batch_enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}

            outputs = model(**batch_enc)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            all_pred_labels.extend(preds)
            all_pred_probs.extend(probs)

            if (i // batch_size) % 100 == 0:
                print(f"  Processed {i}/{len(test_texts)} examples...")

    # Write predictions: label + per-class probabilities
    label_key = f"{benchmark_name}_bert_label"
    probs_key = f"{benchmark_name}_bert_probs"

    # Build ordered class list for probability dict
    n_classes = len(id2label)
    class_names = [id2label[i] for i in range(n_classes)]

    for i, (pred_id, probs) in enumerate(zip(all_pred_labels, all_pred_probs)):
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
print("BERT classifier done.")
