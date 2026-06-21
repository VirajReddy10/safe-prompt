"""
Entry point script to generate the synthetic PII dataset.

Usage:
    python scripts/generate_data.py
"""

import sys
from pathlib import Path

# allow running this script directly without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generator import generate_dataset, save_dataset

N_EXAMPLES = 2000
OUTPUT_PATH = "data/processed/synthetic_v1.jsonl"


def main():
    print(f"Generating {N_EXAMPLES} synthetic PII examples...")
    examples = generate_dataset(n_examples=N_EXAMPLES)
    save_dataset(examples, OUTPUT_PATH)

    n_negative = sum(1 for ex in examples if not ex.entities)
    print(f"  {len(examples) - n_negative} positive examples (contain PII)")
    print(f"  {n_negative} negative examples (no PII)")


if __name__ == "__main__":
    main()