# Safe Prompt

A privacy-preserving framework for detecting and anonymizing PII in LLM prompts using fine-tuned BERT, with a detailed comparison to regex baselines.

## Status

✅ **Week 1 & 2 Complete** — Synthetic data generator, regex baseline, real-world dataset integration, DistilBERT fine-tuning, full evaluation.

## Approach

1. ✅ Generate labeled synthetic PII data with span-level annotations
2. ✅ Establish regex baseline for structured PII (email, phone, SSN, credit card)
3. ✅ Integrate real-world PII dataset (ai4privacy/pii-masking-200k) with label mapping
4. ✅ Fine-tune DistilBERT for token classification on combined 10K-example dataset
5. ✅ Compare results and analyze gap closure

## Results

### Test Set Evaluation (1,000 held-out examples, mixed synthetic + real-world data)

| PII Type      | Regex F1 | BERT F1 | Regex Recall | BERT Recall |
|---|---|---|---|---|
| EMAIL         | 1.000    | 1.000   | 1.000        | 1.000       |
| PHONE         | 0.672    | 0.975   | 0.550        | 0.975       |
| SSN           | 0.674    | 0.976   | 0.508        | 0.984       |
| CREDIT_CARD   | 0.818    | 0.897   | 1.000        | 0.938       |
| PERSON_NAME   | 0.000    | 0.972   | 0.000        | 0.972       |
| ADDRESS       | 0.000    | 0.876   | 0.000        | 0.876       |
| **OVERALL**   | **0.429**| **0.948**| **0.284**   | **0.952**   |

### Key Findings

**BERT vs. Regex (121% relative F1 improvement, 0.429 → 0.948)**

- **Structural gap closure**: PERSON_NAME and ADDRESS were completely invisible to regex (0.0 F1), since they require contextual language understanding. BERT achieves 0.972 and 0.876 respectively — this was the gap the project was designed to expose and close.

- **Generalization on format variation**: Regex performs well on clean synthetic data (Week 1: 0.725 overall F1) but degrades significantly on real-world data with non-standard formatting (this evaluation: 0.429 F1). BERT generalizes far better — PHONE recall improves from 0.55 to 0.975, SSN from 0.51 to 0.984 — because it learns patterns from context, not just fixed templates.

- **Semantic ambiguity**: Regex's CREDIT_CARD performance (F1 0.818) is hampered by structural ambiguity: 16-digit patient IDs, account numbers, and other numeric identifiers are indistinguishable from credit card numbers to a length-based pattern. BERT's F1 (0.897) is higher because it uses surrounding context to disambiguate.

See [`notebooks/03_bert_vs_baseline.ipynb`](notebooks/03_bert_vs_baseline.ipynb) for full details.

## Dataset

- **Synthetic**: 2,000 programmatically generated examples with 6 entity types and exact character-span tracking
- **Real-world**: 8,000 examples from `ai4privacy/pii-masking-200k`, mapped to 6 target categories with intelligent sub-component merging (e.g., FIRSTNAME + LASTNAME → single PERSON_NAME span)
- **Combined**: 10,000 examples split 80/10/10 (train/val/test)

Real-world label mapping: fine-grained categories (FIRSTNAME, STREET, CITY, STATE, ZIPCODE) are merged into coarser categories (PERSON_NAME, ADDRESS) when they appear contiguously, with merge gap threshold tuned by inspecting real examples to avoid incorrectly merging non-contiguous references.

## Model

**DistilBERT + BIO Token Classification**

- Base model: `distilbert-base-uncased`
- Labels: 13 (1 "O" + 6 categories × 2 BIO prefixes)
- Training: 3 epochs, lr=2e-5, batch=16, best checkpoint by validation F1
- Colab T4 GPU training time: ~3.5 minutes

Model weights: `models/pii-distilbert/` (not in Git; 265MB)

## Quickstart

```bash
git clone https://github.com/VirajReddy10/safe-prompt.git
cd safe-prompt
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Synthetic data generation + regex baseline

```bash
python scripts/generate_data.py
jupyter notebook notebooks/02_regex_baseline.ipynb
```

### Fine-tuning (requires GPU, ideally Colab)

```bash
# Prepare data
python scripts/prepare_training_data.py

# Train (on Colab, or locally if you have CUDA/MPS)
python scripts/train_bert.py

# Evaluate
python scripts/evaluate_bert.py
```

### View results

```bash
jupyter notebook notebooks/03_bert_vs_baseline.ipynb
```

## Project Structure

safe-prompt/

├── data/

│   ├── raw/                   # (gitignored, not used)

│   └── processed/

│       ├── synthetic_v1.jsonl      # 2000 generated examples

│       ├── real_v1.jsonl           # 8000 real-world examples

│       └── splits/

│           ├── train.jsonl         # 8000 examples

│           ├── val.jsonl           # 1000 examples

│           └── test.jsonl          # 1000 examples

├── models/

│   └── pii-distilbert/        # (gitignored; trained model weights)

│       ├── config.json

│       ├── model.safetensors

│       ├── tokenizer.json

│       └── checkpoint-*

├── notebooks/

│   ├── 02_regex_baseline.ipynb     # Week 1 baseline evaluation

│   └── 03_bert_vs_baseline.ipynb   # Week 2 comparison

├── src/

│   ├── data/

│   │   ├── generator.py            # Synthetic PII data generator

│   │   ├── real_data_loader.py     # Real-world dataset integration

│   │   └── tokenize_align.py       # BIO label alignment for BERT

│   ├── baseline/

│   │   └── regex_detector.py       # Regex-based detector

│   └── evaluation/

│       └── metrics.py              # Span-level precision/recall/F1

├── scripts/

│   ├── generate_data.py            # Entry point: synthetic data generation

│   ├── prepare_training_data.py    # Combine + split datasets

│   ├── train_bert.py               # DistilBERT fine-tuning on Colab

│   └── evaluate_bert.py            # BERT vs. regex comparison

└── tests/                     # (placeholder)

## Key Engineering Decisions

**BIO Tagging with Offset Mapping**: Character-span entity annotations are converted to per-token BIO labels using the tokenizer's offset mapping, correctly handling subword splits (e.g., "Cortez" → `co`/`##rte`/`##z` all tagged `I-PERSON_NAME`) regardless of which model is used.

**Label Mapping & Merging**: The real-world dataset uses fine-grained labels (FIRSTNAME, LASTNAME, STREET, CITY, etc.) while our schema targets 6 coarser categories. Adjacent sub-component spans are merged when the gap between them is ≤2 characters, determined empirically by inspecting actual examples — e.g., "John Smith" (gap: 1 space) merges into one PERSON_NAME, but "John ... (zip: 12345)" (gap: ~10+ characters) correctly doesn't merge, since "zip: 12345" is a parenthetical reference, not part of the contiguous address phrase.

**Evaluation on Real-World Data**: Week 1's regex baseline evaluated on clean synthetic data (0.725 F1). Week 2 evaluates both regex and BERT on real-world test data (0.429 vs. 0.948 F1). The degradation of regex on real data is exactly the point: it exposes why contextual understanding is necessary for PII detection in messy, real-world text.

## Future Directions

- [ ] Production deployment with anonymization pipeline
- [ ] Multi-lingual expansion (the ai4privacy dataset supports 40+ languages)
- [ ] RoBERTa or larger model for even higher accuracy (current: DistilBERT for speed/size trade-off)
- [ ] LLM prompt sanitization as a middleware (e.g., Claude API integration)
- [ ] Evaluation on downstream task: does anonymization preserve utility for classification/summarization?

## License

MIT