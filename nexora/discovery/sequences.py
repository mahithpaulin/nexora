"""Frequent sequential (n-gram) pattern discovery (stdlib only).

Counts n-grams for ``n = 2..max_n`` via the ``ngrams`` primitive, keeps
those with ``count >= min_support``. N-grams containing ``None``/``NaN``
(missing labels) are skipped explicitly. Finds e.g. ``[A -> B -> C]`` in
``A B C A B C`` (count 2). ``support = confidence = count / #n-gram
slots`` (``0..1``); ``occurrences`` = start indices. Deterministic order:
``(-count, n, str(ngram))``.
"""

from __future__ import annotations

import math
from collections import Counter

try:
    from nexora.features.sequence import ngrams
except ImportError:  # fallback when loaded without the package context
    def ngrams(seq, n):
        items = list(seq)
        if n < 1:
            raise ValueError("n must be a positive int")
        return [tuple(items[i: i + n]) for i in range(len(items) - n + 1)]


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def find_frequent_sequences(labels, max_n=3, min_support=2) -> list[dict]:
    """Frequent n-grams (type ``sequential``) with ``{ngram, n, count,
    support, confidence}`` features."""
    if not isinstance(max_n, int) or max_n < 2:
        raise ValueError("max_n must be an int >= 2")
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    labels = list(labels)
    out: list[dict] = []
    for n in range(2, max_n + 1):
        grams = [g for g in ngrams(labels, n)
                 if not any(_is_missing(t) for t in g)]
        slots = len(labels) - n + 1
        if slots <= 0 or not grams:
            continue
        counts = Counter(grams)
        for gram, cnt in sorted(counts.items(),
                                key=lambda kv: (-kv[1], n, str(kv[0]))):
            if cnt < min_support:
                continue
            support = cnt / slots
            occ = [i for i in range(slots)
                   if tuple(labels[i: i + n]) == gram]
            out.append({
                "id": None, "type": "sequential",
                "features": {"ngram": list(gram), "n": n, "count": cnt,
                             "support": support, "confidence": support},
                "sequence": list(gram), "frequency": len(occ),
                "occurrences": occ, "confidence": support,
            })
    out.sort(key=lambda p: (-p["features"]["count"],
                            p["features"]["n"],
                            str(p["features"]["ngram"])))
    return out
