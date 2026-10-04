"""Frequency-based pattern discovery (stdlib only).

``None``/``NaN`` entries are treated as missing and skipped (never form
patterns). Patterns are dicts with keys exactly
``id/type/features/sequence/frequency/occurrences/confidence`` (``id``
is ``None``; core assigns it later). ``support``/``confidence`` are in
``0..1``: fraction of valid rows supporting the pattern.
"""

from __future__ import annotations

import math
from collections import Counter


def _pattern(ptype, features, sequence, occurrences, confidence) -> dict:
    return {"id": None, "type": ptype, "features": features,
            "sequence": sequence, "frequency": len(occurrences),
            "occurrences": occurrences, "confidence": confidence}


def find_recurring_values(labels, min_support=2) -> list[dict]:
    """Values occurring ``>= min_support`` times (type ``recurring_value``).

    Features: ``{value, count, support}``; ``sequence=[value]``;
    ``occurrences`` = input indices; ``confidence`` = support.
    Sorted by ``(-count, str(value))`` for determinism.
    """
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    labels = list(labels)
    valid = [(i, v) for i, v in enumerate(labels)
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    total = len(valid)
    if total == 0:
        return []
    counts = Counter(v for _, v in valid)
    out: list[dict] = []
    for value, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if cnt < min_support:
            continue
        occ = [i for i, v in valid if v == value]
        support = cnt / total
        out.append(_pattern("recurring_value",
                            {"value": value, "count": cnt, "support": support},
                            [value], occ, support))
    return out


def find_distribution_modes(values, bins=10) -> dict:
    """Modal-bin pattern of a numeric distribution (type ``distribution``).

    Equal-width ``bins`` over ``[min, max]``; features hold ``edges``,
    ``counts``, ``mode_bin``, ``mode_range``, ``mode_count``/``support``.
    ``occurrences`` = input indices inside the modal bin. A constant
    series yields one bin with ``confidence == 1.0``.
    """
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive int")
    indexed: list[tuple[int, float]] = []
    for i, v in enumerate(list(values)):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if not isinstance(v, (int, float)):
            raise ValueError(f"non-numeric value {v!r}")
        indexed.append((i, float(v)))
    n = len(indexed)
    if n == 0:
        return _pattern("distribution",
                        {"bins": 0, "edges": [], "counts": [],
                         "reason": "no valid values"},
                        [], [], 0.0)
    xs = [v for _, v in indexed]
    lo, hi = min(xs), max(xs)
    if lo == hi:
        occ = [i for i, _ in indexed]
        return _pattern("distribution",
                        {"bins": 1, "edges": [lo, hi], "counts": [n],
                         "mode_bin": 0, "mode_range": [lo, hi],
                         "mode_center": lo, "mode_count": n, "support": 1.0},
                        xs, occ, 1.0)
    width = (hi - lo) / bins
    edges = [lo + k * width for k in range(bins + 1)]
    edges[-1] = hi
    counts = [0] * bins
    which: list[int] = []
    for _, v in indexed:
        b = min(int((v - lo) / width), bins - 1)
        counts[b] += 1
        which.append(b)
    mode = max(range(bins), key=lambda b: counts[b])  # first max: deterministic
    occ = [i for (i, _), b in zip(indexed, which) if b == mode]
    seq = [v for (_, v), b in zip(indexed, which) if b == mode]
    support = counts[mode] / n
    return _pattern("distribution",
                    {"bins": bins, "edges": edges, "counts": counts,
                     "mode_bin": mode, "mode_range": [edges[mode], edges[mode + 1]],
                     "mode_center": (edges[mode] + edges[mode + 1]) / 2.0,
                     "mode_count": counts[mode], "support": support},
                    seq, occ, support)
