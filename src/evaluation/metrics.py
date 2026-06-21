"""
Evaluation metrics for PII detection — span-level precision, recall, F1.

Uses exact span matching: a predicted entity counts as correct only if
its type, start, and end all match a ground-truth entity exactly.
"""

from dataclasses import dataclass

from data.generator import Entity


@dataclass
class CategoryMetrics:
    """Precision/recall/F1 for a single PII category."""
    category: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _entity_key(e: Entity) -> tuple[str, int, int]:
    """Hashable key for exact span matching: (type, start, end)."""
    return (e.type, e.start, e.end)

def evaluate(
    true_entities_per_example: list[list[Entity]],
    pred_entities_per_example: list[list[Entity]],
) -> dict[str, CategoryMetrics]:
    """
    Compute per-category precision/recall/F1 across a dataset.

    Args:
        true_entities_per_example: ground-truth entities for each example,
            in the same order as pred_entities_per_example.
        pred_entities_per_example: predicted entities for each example.

    Returns:
        Dict mapping category name -> CategoryMetrics, plus an
        "OVERALL" key aggregating across all categories.
    """
    if len(true_entities_per_example) != len(pred_entities_per_example):
        raise ValueError("true and predicted lists must be the same length")

    # category -> [tp, fp, fn]
    counts: dict[str, list[int]] = {}

    def _ensure(category: str) -> list[int]:
        if category not in counts:
            counts[category] = [0, 0, 0]  # tp, fp, fn
        return counts[category]

    for true_entities, pred_entities in zip(true_entities_per_example, pred_entities_per_example):
        true_keys = {_entity_key(e): e for e in true_entities}
        pred_keys = {_entity_key(e): e for e in pred_entities}

        matched = true_keys.keys() & pred_keys.keys()
        only_true = true_keys.keys() - pred_keys.keys()
        only_pred = pred_keys.keys() - true_keys.keys()

        for key in matched:
            category = key[0]
            _ensure(category)[0] += 1  # true positive

        for key in only_true:
            category = key[0]
            _ensure(category)[2] += 1  # false negative (missed)

        for key in only_pred:
            category = key[0]
            _ensure(category)[1] += 1  # false positive (spurious)

    results = {
        category: CategoryMetrics(category=category, true_positives=tp, false_positives=fp, false_negatives=fn)
        for category, (tp, fp, fn) in counts.items()
    }

    # aggregate overall
    total_tp = sum(c.true_positives for c in results.values())
    total_fp = sum(c.false_positives for c in results.values())
    total_fn = sum(c.false_negatives for c in results.values())
    results["OVERALL"] = CategoryMetrics(
        category="OVERALL", true_positives=total_tp, false_positives=total_fp, false_negatives=total_fn
    )

    return results