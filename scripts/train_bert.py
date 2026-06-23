"""
Fine-tune DistilBERT for PII token classification.

Designed to run on Google Colab (GPU). Can also run locally on CPU/MPS
for quick smoke-testing with a tiny subset before a full Colab run.

Usage:
    python scripts/train_bert.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate as hf_evaluate

from data.generator import Entity
from data.tokenize_align import align_labels_with_tokens, LABEL_LIST, LABEL_TO_ID, ID_TO_LABEL

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("models/pii-distilbert")
MAX_LENGTH = 128


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_dataset(items: list[dict], tokenizer) -> Dataset:
    """Tokenize and align labels for a list of raw examples."""
    all_encodings = []
    for item in items:
        entities = [Entity(**e) for e in item["entities"]]
        encoding = align_labels_with_tokens(item["text"], entities, tokenizer, max_length=MAX_LENGTH)
        all_encodings.append(encoding)

    # convert list of dicts into a dict of lists, which is what
    # datasets.Dataset.from_dict expects
    keys = all_encodings[0].keys()
    data_dict = {key: [enc[key] for enc in all_encodings] for key in keys}
    return Dataset.from_dict(data_dict)

def compute_metrics(eval_pred):
    """
    Compute seqeval-based precision/recall/F1 for the Trainer's evaluation
    loop. Converts label IDs back to BIO strings, ignoring -100 positions.
    """
    seqeval = hf_evaluate.load("seqeval")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [ID_TO_LABEL[l] for l in label_row if l != -100]
        for label_row in labels
    ]
    pred_labels = [
        [ID_TO_LABEL[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=pred_labels, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    print("Loading and tokenizing datasets...")
    train_items = load_split("train")
    val_items = load_split("val")

    train_dataset = build_dataset(train_items, tokenizer)
    val_dataset = build_dataset(val_items, tokenizer)

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Done.")


if __name__ == "__main__":
    main()