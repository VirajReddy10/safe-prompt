"""
Synthetic PII data generator for Safe Prompt.

Generates labeled sentences containing PII entities (email, phone, SSN,
credit card, person name, address) using Faker, with exact character-level
span annotations for each entity.
"""

import random
import json
from dataclasses import dataclass, field
from pathlib import Path

from faker import Faker

fake = Faker("en_US")


@dataclass
class Entity:
    """A single labeled PII span within a generated sentence."""
    type: str
    start: int
    end: int
    value: str


@dataclass
class Example:
    """One generated training example: a sentence plus its PII labels."""
    text: str
    entities: list[Entity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entities": [
                {"type": e.type, "start": e.start, "end": e.end, "value": e.value}
                for e in self.entities
            ],
        }


def _fill_template(template: str, fillers: dict[str, tuple[str, str]]) -> Example:
    """
    Fill a template string with values, tracking character spans.

    `fillers` maps placeholder name -> (entity_type, value).
    e.g. {"name": ("PERSON_NAME", "John Smith"), "email": ("EMAIL", "j@x.com")}

    Returns an Example with entities positioned at their actual offsets
    in the final string.
    """
    entities = []
    result = ""
    cursor = 0  # position in the template string we've processed so far

    while cursor < len(template):
        # find the next placeholder, e.g. "{name}"
        open_brace = template.find("{", cursor)
        if open_brace == -1:
            # no more placeholders; append the rest of the template as-is
            result += template[cursor:]
            break

        # append the plain text before the placeholder
        result += template[cursor:open_brace]

        close_brace = template.find("}", open_brace)
        placeholder = template[open_brace + 1:close_brace]

        entity_type, value = fillers[placeholder]
        start = len(result)
        result += value
        end = len(result)

        entities.append(Entity(type=entity_type, start=start, end=end, value=value))

        cursor = close_brace + 1

    return Example(text=result, entities=entities)

def _gen_email() -> str:
    return fake.email()


def _gen_phone() -> str:
    # Faker's phone numbers can be inconsistent in format; pick from a
    # few realistic US patterns ourselves for more reliable regex testing.
    formats = [
        lambda: f"{fake.numerify('###')}-{fake.numerify('###')}-{fake.numerify('####')}",
        lambda: f"({fake.numerify('###')}) {fake.numerify('###')}-{fake.numerify('####')}",
        lambda: f"+1-{fake.numerify('###')}-{fake.numerify('###')}-{fake.numerify('####')}",
    ]
    return random.choice(formats)()


def _gen_ssn() -> str:
    return fake.ssn()


def _gen_credit_card() -> str:
    # Force realistic card lengths (Visa/Mastercard: 16 digits, Amex: 15).
    # Faker's default can sometimes produce shorter (12-digit) numbers
    # depending on card type, which don't reflect real-world card formats.
    return fake.credit_card_number(card_type=random.choice(["visa", "mastercard", "amex"]))


def _gen_name() -> str:
    return fake.name()


def _gen_address() -> str:
    # Single-line address; Faker's default includes newlines, which we
    # don't want inside a sentence.
    return fake.address().replace("\n", ", ")

TEMPLATES_WITH_PII = [
    "Hi, I'm {name}, reach me at {email} or {phone}.",
    "Contact {name} at {email}.",
    "My phone number is {phone}.",
    "Please send the invoice to {email}.",
    "My SSN is {ssn} for verification purposes.",
    "Charge ${amount} to card {credit_card}.",
    "Ship the package to {address}.",
    "You can call {name} on {phone} anytime.",
    "For billing, use card number {credit_card}.",
    "My social security number is {ssn}.",
    "{name} lives at {address}.",
    "Send confirmation to {email} once done.",
    "Reach out to {name} via {email} or {phone}.",
    "The shipping address is {address}, attention {name}.",
]

TEMPLATES_NO_PII = [
    "The weather today is quite pleasant.",
    "I'll have the report ready by Friday.",
    "Let's schedule a meeting for next week.",
    "The project deadline has been extended.",
    "Thanks for your help with this.",
    "Can you review the attached document?",
    "The conference starts at 9 AM tomorrow.",
    "I appreciate your quick response.",
    "Let me know if you have any questions.",
    "The team did a great job on this release.",
]

PLACEHOLDER_GENERATORS = {
    "name": ("PERSON_NAME", _gen_name),
    "email": ("EMAIL", _gen_email),
    "phone": ("PHONE", _gen_phone),
    "ssn": ("SSN", _gen_ssn),
    "credit_card": ("CREDIT_CARD", _gen_credit_card),
    "address": ("ADDRESS", _gen_address),
}

NON_PII_GENERATORS = {
    "amount": lambda: str(fake.random_int(min=10, max=5000)),
}


def generate_example(template: str) -> Example:
    """Generate a single labeled example from a template string."""
    # find all placeholders in the template
    placeholders = []
    cursor = 0
    while True:
        open_brace = template.find("{", cursor)
        if open_brace == -1:
            break
        close_brace = template.find("}", open_brace)
        placeholders.append(template[open_brace + 1:close_brace])
        cursor = close_brace + 1

    fillers = {}
    for ph in placeholders:
        if ph in PLACEHOLDER_GENERATORS:
            entity_type, gen_fn = PLACEHOLDER_GENERATORS[ph]
            fillers[ph] = (entity_type, gen_fn())
        elif ph in NON_PII_GENERATORS:
            # use a sentinel type we filter out before saving entities
            fillers[ph] = ("_IGNORE", NON_PII_GENERATORS[ph]())
        else:
            raise ValueError(f"Unknown placeholder: {ph}")

    example = _fill_template(template, fillers)
    # drop any entities tagged as non-PII filler (e.g. dollar amounts)
    example.entities = [e for e in example.entities if e.type != "_IGNORE"]
    return example


def generate_dataset(n_examples: int, negative_ratio: float = 0.15, seed: int = 42) -> list[Example]:
    """
    Generate a synthetic PII dataset.

    Args:
        n_examples: total number of examples to generate.
        negative_ratio: fraction of examples with no PII at all.
        seed: random seed for reproducibility.
    """
    random.seed(seed)
    Faker.seed(seed)

    n_negative = int(n_examples * negative_ratio)
    n_positive = n_examples - n_negative

    examples = []
    for _ in range(n_positive):
        template = random.choice(TEMPLATES_WITH_PII)
        examples.append(generate_example(template))

    for _ in range(n_negative):
        text = random.choice(TEMPLATES_NO_PII)
        examples.append(Example(text=text, entities=[]))

    random.shuffle(examples)
    return examples

def save_dataset(examples: list[Example], path: str) -> None:
    """Save a dataset as JSONL (one JSON object per line)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")