"""
Evaluate the fine-tuned DistilBERT model against the regex baseline
on the held-out test set, using the same span-level metrics for a
direct, apples-to-apples comparison.

Usage:
    python scripts/evaluate_bert.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data.generator import Entity
from baseline.regex_detector import detect_pii
from evaluation.metrics import evaluate

MODEL_PATH = "models/pii-distilbert"
TEST_PATH = "data/processed/splits/test.jsonl"

CATEGORIES = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "PERSON_NAME", "ADDRESS", "OVERALL"]


def load_test_set() -> list[dict]:
    with open(TEST_PATH) as f:
        return [json.loads(line) for line in f]


def predict_with_bert(text: str, tokenizer, model) -> list[Entity]:
    """Run the fine-tuned model on `text` and return predicted entities,
    reconstructed from token-level BIO predictions back into spans."""
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=128)
    offset_mapping = inputs.pop("offset_mapping")[0]
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0]

    entities: list[Entity] = []
    current_type = None
    current_start = None
    current_end = None

    for pred_id, (start, end) in zip(predictions, offset_mapping):
        start, end = start.item(), end.item()
        if start == end:
            # special token ([CLS], [SEP], padding)
            continue

        label = model.config.id2label[pred_id.item()]

        if label == "O":
            if current_type is not None:
                entities.append(Entity(type=current_type, start=current_start, end=current_end, value=text[current_start:current_end]))
                current_type = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B" or entity_type != current_type:
            # close out any open entity, start a new one
            if current_type is not None:
                entities.append(Entity(type=current_type, start=current_start, end=current_end, value=text[current_start:current_end]))
            current_type = entity_type
            current_start = start
            current_end = end
        else:
            # continuing the same entity (I- tag matching current type)
            current_end = end

    if current_type is not None:
        entities.append(Entity(type=current_type, start=current_start, end=current_end, value=text[current_start:current_end]))

    return entities


def main():
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    print("Loading test set...")
    items = load_test_set()
    print(f"  {len(items)} examples")

    print("Running regex baseline...")
    true_per_example = []
    regex_pred_per_example = []
    bert_pred_per_example = []

    for i, item in enumerate(items):
        text = item["text"]
        true_entities = [Entity(**e) for e in item["entities"]]
        true_per_example.append(true_entities)

        regex_pred_per_example.append(detect_pii(text))
        bert_pred_per_example.append(predict_with_bert(text, tokenizer, model))

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(items)}")

    regex_results = evaluate(true_per_example, regex_pred_per_example)
    bert_results = evaluate(true_per_example, bert_pred_per_example)

    print()
    print(f'{"Category":15} {"Regex F1":>10} {"BERT F1":>10} {"Regex R":>10} {"BERT R":>10}')
    print("-" * 60)
    for cat in CATEGORIES:
        r = regex_results.get(cat)
        b = bert_results.get(cat)
        r_f1 = r.f1 if r else 0.0
        b_f1 = b.f1 if b else 0.0
        r_r = r.recall if r else 0.0
        b_r = b.recall if b else 0.0
        print(f"{cat:15} {r_f1:>10.3f} {b_f1:>10.3f} {r_r:>10.3f} {b_r:>10.3f}")


if __name__ == "__main__":
    main()