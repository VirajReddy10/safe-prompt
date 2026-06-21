# Safe Prompt

A privacy-preserving framework for detecting and anonymizing PII in LLM prompts.

## Status

🚧 **Active rebuild (v2)** — Week 1 complete: synthetic data generator + regex baseline established.

## Approach

1. ✅ Generate labeled synthetic PII data
2. ✅ Establish regex baseline for structured PII (email, phone, SSN, credit card)
3. ⬜ Fine-tune BERT/RoBERTa for token-level PII classification
4. ⬜ Compare results and evaluate anonymization quality

## Baseline Results (Regex Detector)

Evaluated on 2,000 synthetic examples, exact span matching:

| PII Type      | Precision | Recall | F1    |
|----------------|-----------|--------|-------|
| EMAIL          | 1.000     | 1.000  | 1.000 |
| PHONE          | 1.000     | 1.000  | 1.000 |
| SSN            | 1.000     | 1.000  | 1.000 |
| CREDIT_CARD    | 1.000     | 1.000  | 1.000 |
| PERSON_NAME    | 0.000     | 0.000  | 0.000 |
| ADDRESS        | 0.000     | 0.000  | 0.000 |
| **Overall**    | **1.000** | **0.569** | **0.725** |

**Takeaway**: regex achieves perfect precision and recall on structured PII
with rigid formats (email, phone, SSN, credit card numbers), but completely
fails on PERSON_NAME and ADDRESS, which require contextual language
understanding rather than pattern matching. This gap is exactly what the
Week 2 fine-tuned BERT model is designed to close.

See [`notebooks/02_regex_baseline.ipynb`](notebooks/02_regex_baseline.ipynb)
for the full evaluation, including example predictions.

## Quickstart

```bash
git clone https://github.com/VirajReddy10/safe-prompt.git
cd safe-prompt
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Generate the dataset and run the baseline evaluation:
```bash
python scripts/generate_data.py
jupyter notebook notebooks/02_regex_baseline.ipynb
```

## Project Structure

```
safe-prompt/
├── data/              # Generated datasets (gitignored)
├── notebooks/         # Exploration and evaluation notebooks
├── src/
│   ├── data/          # Synthetic PII generator
│   ├── baseline/      # Regex-based detector
│   └── evaluation/    # Span-level precision/recall/F1 metrics
├── scripts/           # Entry points (data generation, etc.)
└── tests/             # Unit tests
```

## Roadmap

- [x] Week 1: Synthetic data generator + regex baseline
- [ ] Week 2: BERT fine-tuning + evaluation against baseline
- [ ] Future: Real-world PII datasets, anonymization pipeline, LLM response quality eval

## License

MIT