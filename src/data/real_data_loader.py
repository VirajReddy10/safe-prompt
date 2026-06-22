"""
Loader for the ai4privacy/pii-masking-200k real-world dataset.

Maps the dataset's fine-grained PII labels onto Safe Prompt's 6 target
categories (EMAIL, PHONE, SSN, CREDIT_CARD, PERSON_NAME, ADDRESS),
merging adjacent sub-component spans (e.g. FIRSTNAME + LASTNAME) into
single entities where they appear contiguously in the text.
"""

import json

from datasets import load_dataset

from data.generator import Entity, Example


# Direct one-to-one label mappings (dataset label -> our category)
DIRECT_LABEL_MAP = {
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONE",
    "SSN": "SSN",
    "CREDITCARDNUMBER": "CREDIT_CARD",
}

# Component labels that get merged into a single PERSON_NAME entity
# when they appear contiguously (separated by a short whitespace/punct gap)
NAME_COMPONENT_LABELS = {"FIRSTNAME", "MIDDLENAME", "LASTNAME"}

# Component labels that get merged into a single ADDRESS entity
ADDRESS_COMPONENT_LABELS = {"STREET", "SECONDARYADDRESS", "BUILDINGNUMBER", "CITY", "STATE", "ZIPCODE"}

# Maximum gap (in characters) between adjacent component spans for them
# to be considered "contiguous" and merged into one entity.
MAX_MERGE_GAP = 2

def _merge_component_spans(spans: list[list], component_labels: set[str], merged_type: str) -> list[Entity]:
    """
    Merge contiguous spans whose label is in `component_labels` into a
    single Entity of type `merged_type`.

    `spans` is the dataset's raw [start, end, label] list, already sorted
    by start position. Non-matching spans (label not in component_labels,
    or label == "O") are ignored here; this function only returns the
    merged entities of `merged_type`.
    """
    merged: list[Entity] = []
    current_start = None
    current_end = None

    for start, end, label in spans:
        if label in component_labels:
            if current_start is None:
                # starting a new run
                current_start, current_end = start, end
            elif start - current_end <= MAX_MERGE_GAP:
                # contiguous with the previous component; extend the run
                current_end = end
            else:
                # gap too large; close out the previous run and start a new one
                merged.append(Entity(
                    type=merged_type,
                    start=current_start,
                    end=current_end,
                    value="",  # filled in by caller, which has the source text
                ))
                current_start, current_end = start, end
        else:
            # a non-component span breaks the current run, UNLESS it's a
            # tiny "O" gap (handled by the `start - current_end` check above
            # on the *next* component span) -- so here we only close the run
            # if the gap before the next matching span (if any) was too large.
            # We handle that via the loop logic above; nothing to do here
            # except let larger "O" spans naturally break things on the next
            # matching label.
            pass

    if current_start is not None:
        merged.append(Entity(type=merged_type, start=current_start, end=current_end, value=""))

    return merged

def convert_example(source_text: str, span_labels_raw: str) -> Example:
    """
    Convert one raw dataset row into a Safe Prompt Example.

    Args:
        source_text: the original text.
        span_labels_raw: the dataset's span_labels field (JSON-encoded string).

    Returns:
        Example with entities mapped onto our 6 target categories.
    """
    spans = json.loads(span_labels_raw)

    entities: list[Entity] = []

    # 1. direct mappings (EMAIL, PHONE, SSN, CREDIT_CARD)
    for start, end, label in spans:
        if label in DIRECT_LABEL_MAP:
            entities.append(Entity(
                type=DIRECT_LABEL_MAP[label],
                start=start,
                end=end,
                value=source_text[start:end],
            ))

    # 2. merged name components -> PERSON_NAME
    name_entities = _merge_component_spans(spans, NAME_COMPONENT_LABELS, "PERSON_NAME")
    for e in name_entities:
        e.value = source_text[e.start:e.end]
        entities.append(e)

    # 3. merged address components -> ADDRESS
    address_entities = _merge_component_spans(spans, ADDRESS_COMPONENT_LABELS, "ADDRESS")
    for e in address_entities:
        e.value = source_text[e.start:e.end]
        entities.append(e)

    # sort by start position for consistency with our synthetic generator's output
    entities.sort(key=lambda e: e.start)

    return Example(text=source_text, entities=entities)

def load_real_examples(n_examples: int = 5000, seed: int = 42) -> list[Example]:
    """
    Load and convert a subset of the ai4privacy/pii-masking-200k dataset
    into Safe Prompt's Example format.

    Args:
        n_examples: number of examples to sample (English-only subset).
        seed: random seed for reproducible sampling.

    Returns:
        List of converted Examples.
    """
    print("Loading ai4privacy/pii-masking-200k...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")

    print("Filtering to English...")
    en = ds.filter(lambda x: x["language"] == "en")

    if n_examples < len(en):
        en = en.shuffle(seed=seed).select(range(n_examples))

    print(f"Converting {len(en)} examples...")
    examples = []
    for row in en:
        example = convert_example(row["source_text"], row["span_labels"])
        examples.append(example)

    n_with_pii = sum(1 for e in examples if e.entities)
    print(f"Done. {n_with_pii}/{len(examples)} examples contain at least one of our 6 target PII types.")

    return examples