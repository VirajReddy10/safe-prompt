"""
Tokenization and BIO label alignment for PII token classification.

Converts character-span entity annotations into per-token BIO labels,
using the target model's own tokenizer so that subword splits are
handled correctly regardless of which model we fine-tune.
"""

from transformers import PreTrainedTokenizerFast

from data.generator import Entity


PII_CATEGORIES = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "PERSON_NAME", "ADDRESS"]

# Build the full BIO label list: O, B-EMAIL, I-EMAIL, B-PHONE, I-PHONE, ...
LABEL_LIST = ["O"] + [f"{prefix}-{cat}" for cat in PII_CATEGORIES for prefix in ("B", "I")]

LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

def align_labels_with_tokens(
    text: str,
    entities: list[Entity],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 128,
) -> dict:
    """
    Tokenize `text` and produce per-token BIO labels aligned to `entities`.

    Uses the tokenizer's offset mapping (character start/end per token)
    to determine which entity, if any, each token falls inside.

    Returns a dict with input_ids, attention_mask, and labels — ready to
    feed directly into a HuggingFace Trainer.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding.pop("offset_mapping")
    labels = []

    # track which entity (if any) we're currently "inside", to know
    # whether to emit B- (first token of entity) or I- (continuation)
    current_entity = None

    for token_start, token_end in offset_mapping:
        # special tokens (like [CLS], [SEP]) have offset (0, 0)
        if token_start == token_end:
            labels.append(-100)  # -100 tells the loss function to ignore this token
            current_entity = None
            continue

        # find which entity (if any) this token falls inside
        matching_entity = None
        for entity in entities:
            if token_start >= entity.start and token_end <= entity.end:
                matching_entity = entity
                break

        if matching_entity is None:
            labels.append(LABEL_TO_ID["O"])
            current_entity = None
        elif matching_entity is current_entity:
            # continuing the same entity as the previous token
            labels.append(LABEL_TO_ID[f"I-{matching_entity.type}"])
        else:
            # first token of a (possibly new) entity
            labels.append(LABEL_TO_ID[f"B-{matching_entity.type}"])
            current_entity = matching_entity

    encoding["labels"] = labels
    return encoding