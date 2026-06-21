"""
Regex-based PII detector — baseline for Safe Prompt.

Deliberately limited to structured PII (email, phone, SSN, credit card).
PERSON_NAME and ADDRESS are intentionally NOT detected here, since they
require contextual understanding that regex can't provide — this gap is
exactly what motivates the BERT-based detector in Week 2.
"""

import re

from data.generator import Entity


# Order matters for credit card vs phone disambiguation in some edge cases,
# but each pattern here is specific enough to avoid most false positives.

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)

SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

CREDIT_CARD_PATTERN = re.compile(
    r"\b\d{13,19}\b"
)

def detect_pii(text: str) -> list[Entity]:
    """
    Run all regex patterns against `text` and return detected PII entities.

    Patterns are applied independently; if two patterns somehow overlap
    on the same span (shouldn't happen with our current patterns, but
    worth guarding against), we keep the first match found and drop
    fully-overlapping duplicates.
    """
    candidates: list[Entity] = []

    for match in EMAIL_PATTERN.finditer(text):
        candidates.append(
            Entity(type="EMAIL", start=match.start(), end=match.end(), value=match.group())
        )

    for match in PHONE_PATTERN.finditer(text):
        candidates.append(
            Entity(type="PHONE", start=match.start(), end=match.end(), value=match.group())
        )

    for match in SSN_PATTERN.finditer(text):
        candidates.append(
            Entity(type="SSN", start=match.start(), end=match.end(), value=match.group())
        )

    for match in CREDIT_CARD_PATTERN.finditer(text):
        candidates.append(
            Entity(type="CREDIT_CARD", start=match.start(), end=match.end(), value=match.group())
        )

    # SSN pattern (###-##-####) can look superficially similar to phone
    # numbers in some edge cases; resolve overlaps by keeping the
    # longest match at each starting position.
    candidates.sort(key=lambda e: (e.start, -(e.end - e.start)))
    resolved: list[Entity] = []
    last_end = -1
    for entity in candidates:
        if entity.start >= last_end:
            resolved.append(entity)
            last_end = entity.end

    return resolved