"""Sequence utilities for label/token series (stdlib only).

``None`` is kept as an ordinary token here (explicit, predictable);
callers that must ignore missing data filter it first. All results are
deterministic (insertion/counter order is normalised by sorting only at
the discovery layer; these primitives preserve input order).
"""

from __future__ import annotations

from collections import Counter


def ngrams(seq, n) -> list[tuple]:
    """Contiguous n-grams as a list of tuples; ``[]`` if ``len(seq) < n``."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive int")
    items = list(seq)
    if len(items) < n:
        return []
    return [tuple(items[i: i + n]) for i in range(len(items) - n + 1)]


def run_length_encode(seq) -> list[tuple]:
    """``[(value, run_length), ...]`` compressing consecutive duplicates."""
    items = list(seq)
    if not items:
        return []
    out: list[tuple] = []
    cur = items[0]
    cnt = 1
    for v in items[1:]:
        if v == cur:
            cnt += 1
        else:
            out.append((cur, cnt))
            cur, cnt = v, 1
    out.append((cur, cnt))
    return out


def transition_counts(seq) -> dict[tuple, int]:
    """``{(a, b): count}`` of adjacent pairs; ``{}`` if ``len < 2``."""
    items = list(seq)
    counts: dict[tuple, int] = {}
    for a, b in zip(items, items[1:]):
        key = (a, b)
        counts[key] = counts.get(key, 0) + 1
    return counts


def transition_probs(seq) -> dict[tuple, float]:
    """``{(a, b): P(b|a)}`` in ``0..1`` (row-normalised counts)."""
    counts = transition_counts(seq)
    totals: dict = {}
    for (a, _), c in counts.items():
        totals[a] = totals.get(a, 0) + c
    return {pair: c / totals[pair[0]] for pair, c in counts.items()}
