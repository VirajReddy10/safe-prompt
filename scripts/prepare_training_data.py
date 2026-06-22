"""
Combine synthetic and real PII datasets, then split into train/val/test.

Usage:
    python scripts/prepare_training_data.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SYNTHETIC_PATH = "data/processed/synthetic_v1.jsonl"
REAL_PATH = "data/processed/real_v1.jsonl"

OUTPUT_DIR = Path("data/processed/splits")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# remaining 0.1 goes to test

SEED = 42


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(items)} examples to {path}")


def main():
    random.seed(SEED)

    print("Loading datasets...")
    synthetic = load_jsonl(SYNTHETIC_PATH)
    real = load_jsonl(REAL_PATH)
    print(f"  Synthetic: {len(synthetic)} examples")
    print(f"  Real: {len(real)} examples")

    combined = synthetic + real
    random.shuffle(combined)
    print(f"  Combined + shuffled: {len(combined)} examples")

    n = len(combined)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = combined[:n_train]
    val = combined[n_train:n_train + n_val]
    test = combined[n_train + n_val:]

    print()
    print("Saving splits...")
    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")


if __name__ == "__main__":
    main()