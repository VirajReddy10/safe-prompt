# Safe Prompt

A privacy-preserving framework for detecting and anonymizing PII in LLM prompts.

## Status

🚧 **Active rebuild (v2)** — Week 1: synthetic data + regex baseline.

## Approach

1. Generate labeled synthetic PII data
2. Establish regex baseline for structured PII (email, phone, SSN, credit card)
3. Fine-tune BERT/RoBERTa for token-level PII classification
4. Compare results and evaluate anonymization quality

## Quickstart

```bash
git clone https://github.com/VirajReddy10/safe-prompt.git
cd safe-prompt
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Generate data and run the baseline:
```bash
python scripts/generate_data.py
jupyter notebook notebooks/02_regex_baseline.ipynb
```

## Project Structure