"""Set-based precision/recall/F1 for the evaluation harness."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable


def precision_recall_f1(
    expected: Iterable[Hashable], got: Iterable[Hashable]
) -> Dict[str, float]:
    """Precision/recall/F1 over two sets of hashable identities/indices.

    ``tp``/``fp``/``fn`` are counts reported as floats; ``precision``,
    ``recall`` and ``f1`` are in [0, 1]. Both-empty means perfect agreement
    (all 1.0); any other zero-denominator yields 0.0 for that rate.
    """
    e = set(expected)
    g = set(got)
    tp = float(len(e & g))
    fp = float(len(g - e))
    fn = float(len(e - g))
    if tp == 0.0 and fp == 0.0 and fn == 0.0:
        return {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def f1(expected: Iterable[Hashable], got: Iterable[Hashable]) -> float:
    """Convenience helper returning just the F1 of ``precision_recall_f1``."""
    return precision_recall_f1(expected, got)["f1"]
