"""Similarity measures in ``0..1``, each returning ``(score, reason)``.

- ``cosine_similarity``: cosine mapped via ``(c+1)/2`` -> ``1`` = same
  direction, ``0.5`` = orthogonal, ``0`` = opposite. Zero vector -> ``0.0``.
- ``pearson_similarity``: Pearson ``r`` mapped via ``(r+1)/2`` -> ``1`` =
  perfectly correlated, ``0.5`` = uncorrelated, ``0`` = anti-correlated.
  Constant series -> ``0.5`` (correlation undefined).
- ``sequence_similarity``: ``difflib.SequenceMatcher`` ratio (order-aware;
  works on strings or token lists). ``1`` = identical, ``0`` = disjoint.
- ``label_overlap``: Jaccard ``|A cap B| / |A cup B|`` over label sets
  (order-insensitive). ``1`` = same label set, ``0`` = no shared labels.

Numeric inputs must be non-empty, equal-length, real numbers without
``None``/``NaN`` (else ``ValueError``). ``None`` sequences/labels raise
``ValueError``.
"""

from __future__ import annotations

import difflib
import math


def _numvec(seq, name: str) -> list[float]:
    if not isinstance(seq, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if len(seq) == 0:
        raise ValueError(f"{name} must be non-empty")
    out: list[float] = []
    for v in seq:
        if v is None or not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must hold only real numbers "
                             f"(None/NaN not allowed), got {v!r}")
        out.append(float(v))
    return out


def cosine_similarity(a, b) -> tuple[float, str]:
    """Cosine similarity mapped to ``0..1`` (see module docstring)."""
    xa, xb = _numvec(a, "a"), _numvec(b, "b")
    if len(xa) != len(xb):
        raise ValueError(f"length mismatch: {len(xa)} != {len(xb)}")
    na = math.sqrt(sum(v * v for v in xa))
    nb = math.sqrt(sum(v * v for v in xb))
    if na == 0.0 or nb == 0.0:
        return 0.0, "zero-magnitude vector: cosine undefined -> score 0.0"
    c = sum(u * v for u, v in zip(xa, xb)) / (na * nb)
    c = max(-1.0, min(1.0, c))
    return (c + 1.0) / 2.0, f"cosine={c:.4f} mapped via (c+1)/2"


def pearson_similarity(a, b) -> tuple[float, str]:
    """Pearson correlation mapped to ``0..1`` via ``(r+1)/2``."""
    xa, xb = _numvec(a, "a"), _numvec(b, "b")
    if len(xa) != len(xb):
        raise ValueError(f"length mismatch: {len(xa)} != {len(xb)}")
    n = len(xa)
    ma, mb = sum(xa) / n, sum(xb) / n
    da = [v - ma for v in xa]
    db = [v - mb for v in xb]
    denom = math.sqrt(sum(v * v for v in da) * sum(v * v for v in db))
    if denom == 0.0:
        return 0.5, "constant series: correlation undefined -> neutral 0.5"
    r = sum(u * v for u, v in zip(da, db)) / denom
    r = max(-1.0, min(1.0, r))
    return (r + 1.0) / 2.0, f"pearson r={r:.4f} mapped via (r+1)/2"


def sequence_similarity(a, b) -> tuple[float, str]:
    """Order-aware similarity via ``difflib.SequenceMatcher.ratio()``."""
    if a is None or b is None:
        raise ValueError("sequence inputs must not be None")
    sa = a if isinstance(a, str) else list(a)
    sb = b if isinstance(b, str) else list(b)
    if len(sa) == 0 and len(sb) == 0:
        return 1.0, "both sequences empty -> identical"
    if len(sa) == 0 or len(sb) == 0:
        return 0.0, "one sequence empty -> disjoint"
    r = float(difflib.SequenceMatcher(None, sa, sb).ratio())
    return r, f"difflib ratio={r:.4f}: 1.0=identical order, 0.0=disjoint"


def label_overlap(a, b) -> tuple[float, str]:
    """Jaccard overlap of label collections (order-insensitive)."""
    if a is None or b is None:
        raise ValueError("label inputs must not be None")
    sa = {a} if isinstance(a, str) else set(a)
    sb = {b} if isinstance(b, str) else set(b)
    if not sa and not sb:
        return 1.0, "both label sets empty -> identical"
    if not sa or not sb:
        return 0.0, "one label set empty -> no overlap"
    inter = len(sa & sb)
    score = inter / len(sa | sb)
    return score, f"jaccard={score:.4f} ({inter} shared of {len(sa | sb)} unique)"
